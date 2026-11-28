"""Thin multi-provider LLM wrapper with tracing.

Uses LiteLLM so one interface can call Claude, GPT, Mistral, etc.
Set MOCK=1 to run the whole pipeline offline with canned answers (plumbing test only:
mock outputs are NOT results and must never be reported as "mesuré").

SEED is only sent to providers that accept it (OpenAI and Mistral do, Anthropic does not: LiteLLM
raises UnsupportedParamsError if it is sent). Each trace records `seed_applied`; calls without a
seed are not reproducible token for token, so their logged outputs are the reference.
Likewise TEMPERATURE is only sent to models that accept it: the Claude 5.5 family has reasoning always on
and only runs at its own sampling settings (`temperature_applied` is false in their traces).
"""
import functools
import json
import os
import re
import time
import uuid
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parent.parent / "experiments" / "traces"
MOCK = os.getenv("MOCK", "0") == "1"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
SEED = int(os.getenv("SEED", "42"))

_run_id = os.getenv("RUN_ID") or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def run_id() -> str:
    return _run_id


def trace(event: dict) -> None:
    """Append one event to experiments/traces/<run_id>.jsonl."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_id": _run_id, **event}
    with open(TRACE_DIR / f"{_run_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_json(text: str):
    """Parse the first JSON object in a model answer."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


@functools.lru_cache(maxsize=None)
def seed_supported(model: str) -> bool:
    """True if LiteLLM can pass `seed` to this model's provider."""
    import litellm
    try:
        params = litellm.get_supported_openai_params(model=model)
    except Exception:
        return False
    return "seed" in (params or [])


@functools.lru_cache(maxsize=None)
def temperature_supported(model: str) -> bool:
    """False for models LiteLLM's model map flags as rejecting sampling params (Claude 5.5 family)."""
    import litellm
    entry = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
    return entry.get("supports_sampling_params") is not False


def call(role: str, model: str, system: str, user: str, prompt_version: str,
         temperature: float | None = None, seed: int | None = None) -> dict:
    """Call a model, return {"text", "json", "latency_s", "cost_usd"} and log a trace.

    temperature / seed override TEMPERATURE / SEED for this call (best-of-N sampling); the trace logs
    the values actually sent.
    """
    temperature = TEMPERATURE if temperature is None else temperature
    seed = SEED if seed is None else seed
    start = time.perf_counter()
    cost = 0.0
    seed_applied = False
    temperature_applied = False
    served = None
    if MOCK:
        from mock import mock_answer
        text = mock_answer(role, user)
    else:
        import litellm
        seed_applied = seed_supported(model)
        temperature_applied = temperature_supported(model)
        extra = {}
        if temperature_applied:
            extra["temperature"] = temperature
        if seed_applied:
            extra["seed"] = seed
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        text = resp.choices[0].message.content
        served = getattr(resp, "model", None)
        try:
            cost = litellm.completion_cost(completion_response=resp) or 0.0
        except Exception:
            cost = 0.0
    latency = time.perf_counter() - start
    parsed = parse_json(text)
    trace({
        "type": "llm_call", "role": role, "model": "mock" if MOCK else model, "model_served": served,
        "prompt_version": prompt_version,
        "temperature": temperature if temperature_applied else None, "temperature_applied": temperature_applied,
        "seed": seed if seed_applied else None, "seed_applied": seed_applied,
        "latency_s": round(latency, 3), "cost_usd": cost, "json_ok": parsed is not None,
        "input": user[:2000], "output": (text or "")[:4000],
    })
    return {"text": text, "json": parsed, "latency_s": latency, "cost_usd": cost}
