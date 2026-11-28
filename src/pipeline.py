"""Minimal multi-LLM consulting chain.

objective -> orchestrator -> N generators -> auditor -> personas (bandit) -> best scenario
The auditor runs first, so non-compliant scenarios never use the persona budget.
PERSONA_MODE=bandit (default): top-two Thompson sampling (src/bandit.py) stops once the winner is
BANDIT_CONFIDENCE likely (default 95%) or after PERSONA_BUDGET persona calls per case.
PERSONA_MODE=exhaustive: every compliant scenario meets every persona (reference for src/bandit_eval.py).
No UI and no merge step yet.
"""
import json
import os
import random
from pathlib import Path

import prompts as P
from bandit import reward, run_bandit
from llm import SEED, call
from tools import request_tool

ROOT = Path(__file__).resolve().parent.parent

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "anthropic/claude-sonnet-5-5")
GENERATOR_MODELS = [m.strip() for m in os.getenv(
    "GENERATOR_MODELS",
    "anthropic/claude-sonnet-5-5,openai/gpt-4o-mini,gemini/gemini-flash-latest").split(",") if m.strip()]
PERSONA_MODEL = os.getenv("PERSONA_MODEL", "openai/gpt-4o-mini")
AUDITOR_MODEL = os.getenv("AUDITOR_MODEL", "mistral/mistral-large-latest")
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "openai/gpt-4o-mini")
PERSONA_MODE = os.getenv("PERSONA_MODE", "bandit")
PERSONA_BUDGET = int(os.getenv("PERSONA_BUDGET", "30"))
BANDIT_MIN_PULLS = int(os.getenv("BANDIT_MIN_PULLS", "3"))
BANDIT_CONFIDENCE = float(os.getenv("BANDIT_CONFIDENCE", "0.95"))

if AUDITOR_MODEL in GENERATOR_MODELS:  # no model grades its own work
    raise SystemExit(f"AUDITOR_MODEL ({AUDITOR_MODEL}) must differ from every GENERATOR_MODELS entry.")
if PERSONA_MODE not in ("bandit", "exhaustive"):
    raise SystemExit(f"PERSONA_MODE must be 'bandit' or 'exhaustive', not {PERSONA_MODE!r}.")

PERSONAS = json.loads((ROOT / "data" / "personas.json").read_text(encoding="utf-8"))


def persona_order() -> list[int]:
    """Same seeded persona order for every arm and every case (paired comparison, reproducible)."""
    order = list(range(len(PERSONAS)))
    random.Random(SEED).shuffle(order)
    return order


def _kpi_score(audit: dict) -> float:
    try:
        return min(max(float(audit.get("kpi_score", 0)), 0.0), 10.0)
    except (TypeError, ValueError):
        return 0.0


def _intent(answer) -> float | None:
    value = answer.get("purchase_intent") if isinstance(answer, dict) else None
    if isinstance(value, bool):
        return None
    try:
        return min(max(float(value), 1.0), 10.0)
    except (TypeError, ValueError):
        return None


def _audit(objective: str, kpis: list, scenario: dict) -> tuple[dict, float, float]:
    user = json.dumps({"objective": objective, "kpis": kpis, "scenario": scenario}, ensure_ascii=False)
    r = call("auditor", AUDITOR_MODEL, P.AUDITOR, user, P.PROMPT_VERSION)
    audit = r["json"] if isinstance(r["json"], dict) else {}
    return audit, r["latency_s"], r["cost_usd"]


def run_candidate(objective: str) -> dict:
    spent = {"latency_s": 0.0, "cost_usd": 0.0}

    def track(r: dict) -> dict:
        spent["latency_s"] += r["latency_s"]
        spent["cost_usd"] += r["cost_usd"]
        return r

    o = track(call("orchestrator", ORCHESTRATOR_MODEL, P.ORCHESTRATOR, objective, P.PROMPT_VERSION))
    plan = o["json"] if isinstance(o["json"], dict) else {}
    if plan.get("refuse"):
        return {"refused": True, "refusal_reason": plan.get("refusal_reason", ""), **spent}

    kpis = plan.get("kpis", [])
    scenarios = []
    for i, model in enumerate(GENERATOR_MODELS):
        style = P.GENERATOR_STYLES[i % len(P.GENERATOR_STYLES)]
        user = json.dumps({"objective": objective, "kpis": kpis, "conflicts": plan.get("conflicts", [])},
                          ensure_ascii=False)
        g = track(call(f"generator_{i}", model, P.GENERATOR.replace("{style}", style), user, P.PROMPT_VERSION))
        if not isinstance(g["json"], dict):
            continue
        audit, a_lat, a_cost = _audit(objective, kpis, g["json"])
        spent["latency_s"] += a_lat
        spent["cost_usd"] += a_cost
        scenarios.append({"generator": model, "style": style, "scenario": g["json"], "audit": audit,
                          "compliant": audit.get("compliance") == "pass", "kpi": _kpi_score(audit),
                          "intents": {}})

    arms = [s for s in scenarios if s["compliant"]]
    order = persona_order()

    def pull(arm: int, k: int) -> float | None:
        s, persona = arms[arm], PERSONAS[order[k]]
        r = track(call(f"persona_{persona['id']}", PERSONA_MODEL,
                       P.PERSONA.format(persona=json.dumps(persona, ensure_ascii=False)),
                       json.dumps(s["scenario"], ensure_ascii=False), P.PROMPT_VERSION))
        intent = _intent(r["json"])
        s["intents"][persona["id"]] = intent
        return None if intent is None else reward(s["kpi"], intent)

    exhaustive_calls = len(arms) * len(PERSONAS)
    bandit = None
    if PERSONA_MODE == "exhaustive":
        for a in range(len(arms)):
            for k in range(len(PERSONAS)):
                pull(a, k)
        calls = exhaustive_calls
    else:
        budget = min(PERSONA_BUDGET, exhaustive_calls)
        bandit = run_bandit(len(arms), len(PERSONAS), pull, budget, random.Random(SEED),
                            min_pulls=BANDIT_MIN_PULLS, confidence=BANDIT_CONFIDENCE)
        calls = bandit["calls"]

    for s in scenarios:
        seen = [v for v in s["intents"].values() if v is not None]
        s["persona_n"] = len(s["intents"])
        s["persona_avg"] = sum(seen) / len(seen) if seen else None
        s["combined"] = (s["kpi"] + s["persona_avg"]) / 2 if s["compliant"] and seen else -1

    if bandit is not None and arms:
        best = arms[bandit["choice"]]
    else:
        best = max(scenarios, key=lambda s: s["combined"]) if scenarios else None

    tool_results = [request_tool(a) for a in plan.get("requested_actions", [])]
    return {"refused": False, "plan": plan, "scenarios": scenarios, "best": best,
            "persona_mode": PERSONA_MODE, "persona_calls": calls, "persona_calls_exhaustive": exhaustive_calls,
            "persona_order": [PERSONAS[i]["id"] for i in order], "bandit": bandit,
            "tool_results": tool_results, **spent}


def run_baseline(objective: str) -> dict:
    """Single LLM, single prompt. Scored by the same auditor so the metric is comparable."""
    b = call("baseline", BASELINE_MODEL, P.BASELINE, objective, P.PROMPT_VERSION)
    latency, cost = b["latency_s"], b["cost_usd"]
    out = b["json"] if isinstance(b["json"], dict) else {}
    if out.get("refuse"):
        return {"refused": True, "refusal_reason": out.get("refusal_reason", ""),
                "latency_s": latency, "cost_usd": cost}
    audit, a_lat, a_cost = _audit(objective, [], out)
    return {"refused": False, "best": {"scenario": out, "audit": audit, "persona_avg": None},
            "tool_results": [], "latency_s": latency + a_lat, "cost_usd": cost + a_cost}
