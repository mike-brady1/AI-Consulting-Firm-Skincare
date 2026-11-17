"""Run baseline vs candidate on the frozen test cases.

Usage:  python src/run_eval.py            (real models, needs API keys in .env)
        MOCK=1 python src/run_eval.py     (offline plumbing test, results are NOT valid)
Writes experiments/results.csv (one row per case x system) and experiments/outputs/<run_id>.json.
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import prompts as P
from llm import MOCK, run_id
from pipeline import run_baseline, run_candidate

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
FIELDS = ["run_id", "mode", "prompt_version", "case_id", "category", "system", "refused",
          "expected_refusal", "refusal_correct", "tool_refusal_expected", "tool_refusal_ok",
          "kpi_score", "compliance", "persona_avg", "mentions_conflict", "latency_s", "cost_usd", "status"]


def row(case, system, out):
    best = out.get("best") or {}
    audit = best.get("audit") or {}
    tool_expected = case.get("expected_tool_refusal", "")
    tool_ok = ""
    if tool_expected:
        tool_ok = any(t["tool"] == tool_expected and t["status"] == "refused"
                      for t in out.get("tool_results", []))
    return {
        "run_id": run_id(), "mode": "mock" if MOCK else "real", "prompt_version": P.PROMPT_VERSION,
        "case_id": case["id"], "category": case["category"], "system": system,
        "refused": out.get("refused", False), "expected_refusal": case["expected_refusal"],
        "refusal_correct": "" if "error" in out else out.get("refused", False) == case["expected_refusal"],
        "tool_refusal_expected": tool_expected, "tool_refusal_ok": tool_ok,
        "kpi_score": audit.get("kpi_score", ""), "compliance": audit.get("compliance", ""),
        "persona_avg": "" if best.get("persona_avg") is None else round(best["persona_avg"], 2),
        "mentions_conflict": audit.get("mentions_conflict", ""),
        "latency_s": round(out.get("latency_s", 0), 2), "cost_usd": round(out.get("cost_usd", 0), 5),
        "status": out.get("error", "ok"),
    }


def main():
    rows, outputs = [], {}
    for case in CASES:
        for system, fn in (("baseline", run_baseline), ("candidate", run_candidate)):
            try:
                out = fn(case["objective"])
            except Exception as exc:  # keep going, log the failure as an incident
                out = {"error": f"error: {type(exc).__name__}: {exc}"[:200]}
            outputs[f"{case['id']}_{system}"] = out
            rows.append(row(case, system, out))
            print(f"{case['id']} {system:9s} refused={out.get('refused')} status={out.get('error', 'ok')}")

    exp = ROOT / "experiments"
    (exp / "outputs").mkdir(parents=True, exist_ok=True)
    (exp / "outputs" / f"{run_id()}.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    results = exp / "results.csv"
    new_file = not results.exists() or results.stat().st_size == 0
    with open(results, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)

    for system in ("baseline", "candidate"):
        rs = [r for r in rows if r["system"] == system]
        ok = sum(1 for r in rs if r["refusal_correct"] is True)
        scores = [r["kpi_score"] for r in rs if isinstance(r["kpi_score"], (int, float))]
        avg = sum(scores) / len(scores) if scores else float("nan")
        print(f"[{system}] correct refusal decisions: {ok}/{len(rs)} | mean KPI score: {avg:.2f}")
    if MOCK:
        print("MOCK run: plumbing check only, do not report these numbers.")


if __name__ == "__main__":
    main()
