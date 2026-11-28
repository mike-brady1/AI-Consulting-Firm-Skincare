"""Canned answers for MOCK=1 (offline plumbing test). Not a model, not a result.

Persona intents vary deterministically with the persona and the scenario text (hash), so the bandit has
something to tell apart offline. The numbers mean nothing.
"""
import hashlib
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
    if role.startswith("generator") or role.startswith("baseline"):
        if role.startswith("baseline") and any(w in text for w in FORBIDDEN):
            return json.dumps({"refuse": True, "refusal_reason": "mock"})
        return json.dumps({"title": f"Mock scenario ({role})", "how": ["a"], "who": ["b"],
                           "figures": [{"kpi": "ventes", "estimate": "+12%", "hypothese": True}],
                           "channels": ["instagram"], "claims": [], "tradeoffs": [], "risks": []})
    if role.startswith("persona"):
        h = int(hashlib.sha256((role + user).encode("utf-8")).hexdigest(), 16)
        return json.dumps({"purchase_intent": 3 + h % 7, "reason": "mock"})
    if role == "auditor":
        return json.dumps({"kpi_score": 6, "compliance": "pass", "issues": [], "mentions_conflict": False})
    return "{}"
