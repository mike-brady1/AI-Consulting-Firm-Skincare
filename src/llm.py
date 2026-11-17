"""Thin multi-provider LLM wrapper with tracing.

Uses LiteLLM so one interface can call Claude, GPT, Mistral, etc.
Set MOCK=1 to run the whole pipeline offline with canned answers (plumbing test only:
mock outputs are NOT results and must never be reported as "mesuré").
"""
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


def call(role: str, model: str, system: str, user: str, prompt_version: str) -> dict:
    """Call a model, return {"text", "json", "latency_s", "cost_usd"} and log a trace."""
    start = time.perf_counter()
    cost = 0.0
    if MOCK:
        from mock import mock_answer
        text = mock_answer(role, user)
    else:
        import litellm
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=TEMPERATURE,
            seed=SEED,
        )
        text = resp.choices[0].message.content
        try:
            cost = litellm.completion_cost(completion_response=resp) or 0.0
        except Exception:
            cost = 0.0
    latency = time.perf_counter() - start
    parsed = parse_json(text)
    trace({
        "type": "llm_call", "role": role, "model": "mock" if MOCK else model,
        "prompt_version": prompt_version, "temperature": TEMPERATURE, "seed": SEED,
        "latency_s": round(latency, 3), "cost_usd": cost, "json_ok": parsed is not None,
        "input": user[:2000], "output": (text or "")[:4000],
    })
    return {"text": text, "json": parsed, "latency_s": latency, "cost_usd": cost}
