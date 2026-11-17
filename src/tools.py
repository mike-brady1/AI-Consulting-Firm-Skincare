"""Tools the orchestrator may request, behind a permission gate.

Restricted scope: in this prototype NO tool may act on the outside world.
publish_ad always needs human approval, so an automated request is refused and logged.
"""
from llm import trace

PERMISSIONS = {
    "publish_ad": {"allowed": False, "reason": "human_approval_required"},
    "export_report": {"allowed": True, "reason": "local_file_only"},
}


def request_tool(name: str, args: dict | None = None) -> dict:
    rule = PERMISSIONS.get(name, {"allowed": False, "reason": "unknown_tool"})
    status = "allowed" if rule["allowed"] else "refused"
    result = {"tool": name, "args": args or {}, "status": status, "reason": rule["reason"]}
    trace({"type": "tool_call", **result})
    return result
