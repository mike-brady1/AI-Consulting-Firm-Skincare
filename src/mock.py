"""Canned answers for MOCK=1 (offline plumbing test). Not a model, not a result."""
import json

FORBIDDEN = ["guérit", "guerit", "faux avis", "avant/après", "avant/apres"]


def mock_answer(role: str, user: str) -> str:
    text = user.lower()
    if role == "orchestrator":
        refuse = any(w in text for w in FORBIDDEN)
        actions = ["publish_ad"] if "publie" in text else []
        return json.dumps({"refuse": refuse, "refusal_reason": "mock: forbidden claim" if refuse else "",
                           "needs_clarification": "faire mieux" in text, "conflicts": [],
                           "kpis": [{"name": "ventes", "target": "+15%", "hypothese": False}],
                           "requested_actions": actions})
    if role.startswith("generator") or role == "baseline":
        if role == "baseline" and any(w in text for w in FORBIDDEN):
            return json.dumps({"refuse": True, "refusal_reason": "mock"})
        return json.dumps({"title": "Mock scenario", "how": ["a"], "who": ["b"],
                           "figures": [{"kpi": "ventes", "estimate": "+12%", "hypothese": True}],
                           "channels": ["instagram"], "claims": [], "tradeoffs": [], "risks": []})
    if role.startswith("persona"):
        return json.dumps({"purchase_intent": 6, "reason": "mock"})
    if role == "auditor":
        return json.dumps({"kpi_score": 6, "compliance": "pass", "issues": [], "mentions_conflict": False})
    return "{}"
