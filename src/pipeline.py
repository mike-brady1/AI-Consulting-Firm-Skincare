"""Minimal multi-LLM consulting chain.

objective -> orchestrator -> N generators -> personas -> auditor -> best scenario
No UI, no bandit and no merge yet (planned for the next sessions).
"""
import json
import os
from pathlib import Path

import prompts as P
from llm import call
from tools import request_tool

ROOT = Path(__file__).resolve().parent.parent

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "anthropic/claude-sonnet-4-5")
GENERATOR_MODELS = [m.strip() for m in os.getenv(
    "GENERATOR_MODELS", "anthropic/claude-sonnet-4-5,openai/gpt-4o-mini").split(",") if m.strip()]
PERSONA_MODEL = os.getenv("PERSONA_MODEL", "openai/gpt-4o-mini")
AUDITOR_MODEL = os.getenv("AUDITOR_MODEL", "mistral/mistral-large-latest")
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "openai/gpt-4o-mini")

PERSONAS = json.loads((ROOT / "data" / "personas.json").read_text(encoding="utf-8"))


def _audit(objective: str, kpis: list, scenario: dict) -> tuple[dict, float, float]:
    user = json.dumps({"objective": objective, "kpis": kpis, "scenario": scenario}, ensure_ascii=False)
    r = call("auditor", AUDITOR_MODEL, P.AUDITOR, user, P.PROMPT_VERSION)
    return (r["json"] or {}), r["latency_s"], r["cost_usd"]


def run_candidate(objective: str) -> dict:
    latency, cost = 0.0, 0.0

    o = call("orchestrator", ORCHESTRATOR_MODEL, P.ORCHESTRATOR, objective, P.PROMPT_VERSION)
    latency += o["latency_s"]; cost += o["cost_usd"]
    plan = o["json"] or {}
    if plan.get("refuse"):
        return {"refused": True, "refusal_reason": plan.get("refusal_reason", ""),
                "latency_s": latency, "cost_usd": cost}

    kpis = plan.get("kpis", [])
    scenarios = []
    for i, model in enumerate(GENERATOR_MODELS):
        style = P.GENERATOR_STYLES[i % len(P.GENERATOR_STYLES)]
        user = json.dumps({"objective": objective, "kpis": kpis, "conflicts": plan.get("conflicts", [])},
                          ensure_ascii=False)
        g = call(f"generator_{i}", model, P.GENERATOR.replace("{style}", style), user, P.PROMPT_VERSION)
        latency += g["latency_s"]; cost += g["cost_usd"]
        if not g["json"]:
            continue
        scenario = g["json"]

        intents = []
        for persona in PERSONAS:
            pr = call(f"persona_{persona['id']}", PERSONA_MODEL,
                      P.PERSONA.format(persona=json.dumps(persona, ensure_ascii=False)),
                      json.dumps(scenario, ensure_ascii=False), P.PROMPT_VERSION)
            latency += pr["latency_s"]; cost += pr["cost_usd"]
            if pr["json"] and isinstance(pr["json"].get("purchase_intent"), (int, float)):
                intents.append(pr["json"]["purchase_intent"])

        audit, a_lat, a_cost = _audit(objective, kpis, scenario)
        latency += a_lat; cost += a_cost
        persona_avg = sum(intents) / len(intents) if intents else None
        compliant = audit.get("compliance") == "pass"
        # Simple combined score for now; the bandit and the merge step replace this later.
        combined = (audit.get("kpi_score", 0) + (persona_avg or 0)) / 2 if compliant else -1
        scenarios.append({"generator": model, "style": style, "scenario": scenario, "audit": audit,
                          "persona_avg": persona_avg, "combined": combined})

    tool_results = [request_tool(a) for a in plan.get("requested_actions", [])]
    best = max(scenarios, key=lambda s: s["combined"]) if scenarios else None
    return {"refused": False, "plan": plan, "scenarios": scenarios, "best": best,
            "tool_results": tool_results, "latency_s": latency, "cost_usd": cost}


def run_baseline(objective: str) -> dict:
    """Single LLM, single prompt. Scored by the same auditor so the metric is comparable."""
    b = call("baseline", BASELINE_MODEL, P.BASELINE, objective, P.PROMPT_VERSION)
    latency, cost = b["latency_s"], b["cost_usd"]
    out = b["json"] or {}
    if out.get("refuse"):
        return {"refused": True, "refusal_reason": out.get("refusal_reason", ""),
                "latency_s": latency, "cost_usd": cost}
    audit, a_lat, a_cost = _audit(objective, [], out)
    return {"refused": False, "best": {"scenario": out, "audit": audit, "persona_avg": None},
            "tool_results": [], "latency_s": latency + a_lat, "cost_usd": cost + a_cost}
