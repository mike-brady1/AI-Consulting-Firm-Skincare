"""Minimal multi-LLM consulting chain, and the two baselines it is compared with.

candidate:     objective -> orchestrator -> N generators (providers x styles) -> select -> best scenario
baseline:      one model, one prompt, one draft (no selection)
baseline_bon:  one model, one prompt, N drafts (sampled) -> select -> best scenario   (best-of-N control)

select = auditor on every draft, then personas on the compliant ones, then pick. It is the same function
for the candidate and baseline_bon, and every audit gets the same KPIs (the orchestrator's), so the
candidate's edge over baseline_bon can only come from how the drafts are generated, not from choosing the
best of N drafts or from a different yardstick.

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
# Best-of-N baseline: as many drafts as the candidate has generators, sampled with a higher temperature
# (at 0.2 the drafts would be near-duplicates and best-of-N would be a strawman) and seeds SEED..SEED+N-1.
BON_N = int(os.getenv("BON_N", str(len(GENERATOR_MODELS))))
BON_TEMPERATURE = float(os.getenv("BON_TEMPERATURE", "0.7"))

if AUDITOR_MODEL in GENERATOR_MODELS + [BASELINE_MODEL]:  # no model grades its own work
    raise SystemExit(f"AUDITOR_MODEL ({AUDITOR_MODEL}) must differ from every generator and from BASELINE_MODEL.")
if PERSONA_MODE not in ("bandit", "exhaustive"):
    raise SystemExit(f"PERSONA_MODE must be 'bandit' or 'exhaustive', not {PERSONA_MODE!r}.")

PERSONAS = json.loads((ROOT / "data" / "personas.json").read_text(encoding="utf-8"))


def persona_order() -> list[int]:
    """Same seeded persona order for every arm and every case (paired comparison, reproducible)."""
    order = list(range(len(PERSONAS)))
    random.Random(SEED).shuffle(order)
    return order


def kpi_score(audit: dict) -> float:
    """Auditor KPI score as a number in [0, 10]; missing or malformed counts as 0."""
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


def _new_spent() -> dict:
    return {"latency_s": 0.0, "cost_usd": 0.0}


def _track(spent: dict, r: dict) -> dict:
    spent["latency_s"] += r["latency_s"]
    spent["cost_usd"] += r["cost_usd"]
    return r


def _audit(objective: str, kpis: list, scenario: dict, spent: dict) -> dict:
    user = json.dumps({"objective": objective, "kpis": kpis, "scenario": scenario}, ensure_ascii=False)
    r = _track(spent, call("auditor", AUDITOR_MODEL, P.AUDITOR, user, P.PROMPT_VERSION))
    return r["json"] if isinstance(r["json"], dict) else {}


def select(objective: str, kpis: list, drafts: list[dict], spent: dict) -> dict:
    """Audit every draft, test the compliant ones on personas (bandit or exhaustive), pick the best.

    drafts: [{"generator", "style", "scenario"}]. Shared by the candidate and the best-of-N baseline so
    both choose their recommendation the same way.
    """
    scenarios = []
    for d in drafts:
        audit = _audit(objective, kpis, d["scenario"], spent)
        scenarios.append({**d, "audit": audit, "compliant": audit.get("compliance") == "pass",
                          "kpi": kpi_score(audit), "intents": {}})

    arms = [s for s in scenarios if s["compliant"]]
    order = persona_order()

    def pull(arm: int, k: int) -> float | None:
        s, persona = arms[arm], PERSONAS[order[k]]
        r = _track(spent, call(f"persona_{persona['id']}", PERSONA_MODEL,
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
    return {"scenarios": scenarios, "best": best, "persona_mode": PERSONA_MODE, "persona_calls": calls,
            "persona_calls_exhaustive": exhaustive_calls, "persona_order": [PERSONAS[i]["id"] for i in order],
            "bandit": bandit}


def run_candidate(objective: str) -> dict:
    spent = _new_spent()
    o = _track(spent, call("orchestrator", ORCHESTRATOR_MODEL, P.ORCHESTRATOR, objective, P.PROMPT_VERSION))
    plan = o["json"] if isinstance(o["json"], dict) else {}
    if plan.get("refuse"):
        return {"refused": True, "refusal_reason": plan.get("refusal_reason", ""), **spent}

    kpis = plan.get("kpis") if isinstance(plan.get("kpis"), list) else []
    drafts = []
    for i, model in enumerate(GENERATOR_MODELS):
        style = P.GENERATOR_STYLES[i % len(P.GENERATOR_STYLES)]
        user = json.dumps({"objective": objective, "kpis": kpis, "conflicts": plan.get("conflicts", [])},
                          ensure_ascii=False)
        g = _track(spent, call(f"generator_{i}", model, P.GENERATOR.replace("{style}", style), user,
                               P.PROMPT_VERSION))
        if isinstance(g["json"], dict):
            drafts.append({"generator": model, "style": style, "scenario": g["json"]})

    chosen = select(objective, kpis, drafts, spent)
    tool_results = [request_tool(a) for a in plan.get("requested_actions", [])]
    return {"refused": False, "plan": plan, **chosen, "tool_results": tool_results, **spent}


def run_baseline(objective: str, kpis: list | None = None) -> dict:
    """Single LLM, single prompt, single draft. Audited with the same KPIs as the candidate."""
    spent = _new_spent()
    b = _track(spent, call("baseline", BASELINE_MODEL, P.BASELINE, objective, P.PROMPT_VERSION))
    out = b["json"] if isinstance(b["json"], dict) else {}
    if out.get("refuse"):
        return {"refused": True, "refusal_reason": out.get("refusal_reason", ""), **spent}
    audit = _audit(objective, kpis or [], out, spent)
    best = {"generator": BASELINE_MODEL, "style": "unique", "scenario": out, "audit": audit,
            "kpi": kpi_score(audit), "persona_avg": None}
    return {"refused": False, "scenarios": [best], "best": best, "tool_results": [], **spent}


def run_baseline_bon(objective: str, kpis: list | None = None) -> dict:
    """Best-of-N control: the baseline model and prompt sampled BON_N times, then the candidate's selection.

    The system refuses when most drafts refuse (majority vote); otherwise it selects among the drafts that
    did not refuse. A draft without valid JSON counts towards N but is neither a refusal nor a candidate.
    """
    spent = _new_spent()
    drafts, refusals = [], []
    for i in range(BON_N):
        b = _track(spent, call(f"baseline_bon_{i}", BASELINE_MODEL, P.BASELINE, objective, P.PROMPT_VERSION,
                               temperature=BON_TEMPERATURE, seed=SEED + i))
        out = b["json"] if isinstance(b["json"], dict) else None
        if out is None:
            continue
        if out.get("refuse"):
            refusals.append(out.get("refusal_reason", ""))
        else:
            drafts.append({"generator": BASELINE_MODEL, "style": f"échantillon {i + 1}", "scenario": out})
    if 2 * len(refusals) > BON_N:
        return {"refused": True, "refusal_reason": refusals[0], "drafts_refused": len(refusals), **spent}
    chosen = select(objective, kpis or [], drafts, spent)
    return {"refused": False, **chosen, "drafts_refused": len(refusals), "tool_results": [], **spent}
