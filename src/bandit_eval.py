"""Offline replay: does the bandit find the exhaustive winner with fewer persona calls?

Usage:
    PERSONA_MODE=exhaustive python src/run_eval.py                 # every compliant scenario x every persona
    python src/bandit_eval.py experiments/outputs/<run_id>.json    # no API call

For each case, the recorded reactions form a scenario x persona reward matrix. The bandit (top-two
Thompson sampling) and a uniform control (round-robin), both with the same warm-up and early-stopping
rule, are replayed on it for several budget caps and seeds. Persona calls are near-deterministic
(temperature 0.2, fixed seed), so a recorded reaction is what a live pull would have returned.

Reported per case and cap: how often each method picks the exhaustive winner and how many calls it used.
Label: "mesuré sur clients simulés (rejeu hors ligne)". From a MOCK run the numbers are plumbing checks only.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandit import CONFIDENCE, MIN_PULLS, reward, run_bandit, run_uniform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TIE = 1e-9


def matrix_for(out: dict) -> tuple[list[list], list[str]] | None:
    arms = [s for s in out.get("scenarios", []) if s.get("compliant")]
    ids = out.get("persona_order") or []
    if len(arms) < 2 or not ids or any(len(s.get("intents", {})) < len(ids) for s in arms):
        return None
    rows = [[None if s["intents"].get(p) is None else reward(s["kpi"], s["intents"][p]) for p in ids]
            for s in arms]
    return rows, [f"{s['generator']} ({s['style']})" for s in arms]


def replay(rows: list[list], budget: int, seeds: int, min_pulls: int, confidence: float) -> dict:
    n_arms, n_p = len(rows), len(rows[0])
    means = []
    for r in rows:
        seen = [v for v in r if v is not None]
        means.append(sum(seen) / len(seen) if seen else 0.0)
    winners = {a for a, m in enumerate(means) if m >= max(means) - TIE}
    stats = {m: {"hits": 0, "calls": 0} for m in ("bandit", "uniform")}
    for seed in range(seeds):
        rng = random.Random(seed)
        order = list(range(n_p))
        rng.shuffle(order)

        def pull(a, k):
            return rows[a][order[k]]

        results = {"bandit": run_bandit(n_arms, n_p, pull, budget, rng, min_pulls, confidence),
                   "uniform": run_uniform(n_arms, n_p, pull, budget, min_pulls, confidence)}
        for method, res in results.items():
            stats[method]["hits"] += res["choice"] in winners
            stats[method]["calls"] += res["calls"]
    ranked = sorted(means, reverse=True)
    return {"winner_margin": ranked[0] - ranked[1],
            **{f"{m}_agreement_pct": 100 * v["hits"] / seeds for m, v in stats.items()},
            **{f"{m}_mean_calls": v["calls"] / seeds for m, v in stats.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", help="experiments/outputs/<run_id>.json from an exhaustive run")
    ap.add_argument("--fractions", default="0.33,0.5,1.0", help="budget caps as fractions of exhaustive testing")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--min-pulls", type=int, default=MIN_PULLS)
    ap.add_argument("--confidence", type=float, default=CONFIDENCE)
    args = ap.parse_args()

    outputs = json.loads(Path(args.outputs).read_text(encoding="utf-8"))
    meta = outputs.get("_meta", {})
    if meta.get("persona_mode") != "exhaustive":
        sys.exit("This run was not exhaustive: rerun with PERSONA_MODE=exhaustive python src/run_eval.py")
    fractions = [float(f) for f in args.fractions.split(",")]

    results = []
    for key, out in outputs.items():
        if not key.endswith("_candidate") or out.get("refused") or "error" in out:
            continue
        case_id = key.removesuffix("_candidate")
        built = matrix_for(out)
        if built is None:
            print(f"{case_id}: skipped (fewer than 2 compliant scenarios, or incomplete reactions)")
            continue
        rows, _ = built
        full = len(rows) * len(rows[0])
        for frac in fractions:
            budget = max(len(rows) * args.min_pulls, round(frac * full))
            r = replay(rows, budget, args.seeds, args.min_pulls, args.confidence)
            results.append({
                "run_id": meta.get("run_id", ""), "mode": meta.get("mode", ""), "case_id": case_id,
                "n_scenarios": len(rows), "n_personas": len(rows[0]), "exhaustive_calls": full,
                "budget_fraction": frac, "budget_cap": budget, "confidence": args.confidence,
                "bandit_agreement_pct": round(r["bandit_agreement_pct"], 1),
                "bandit_mean_calls": round(r["bandit_mean_calls"], 1),
                "bandit_calls_saved_pct": round(100 * (1 - r["bandit_mean_calls"] / full), 1),
                "uniform_agreement_pct": round(r["uniform_agreement_pct"], 1),
                "uniform_mean_calls": round(r["uniform_mean_calls"], 1),
                "winner_margin": round(r["winner_margin"], 4), "seeds": args.seeds,
            })

    if not results:
        sys.exit("No case had 2 or more compliant scenarios with complete reactions: nothing to compare.")
    path = ROOT / "experiments" / f"bandit_{meta.get('run_id', 'run')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    print(f"{'cap':>5} | {'bandit: winner found / calls saved':>35} | {'uniform: winner found / calls saved':>36}")
    for frac in fractions:
        rs = [r for r in results if r["budget_fraction"] == frac]

        def mean(k):
            return sum(r[k] for r in rs) / len(rs)

        u_saved = 100 * (1 - sum(r["uniform_mean_calls"] for r in rs) / sum(r["exhaustive_calls"] for r in rs))
        print(f"{frac:>5.0%} | {mean('bandit_agreement_pct'):>21.0f}% / {mean('bandit_calls_saved_pct'):>9.0f}% | "
              f"{mean('uniform_agreement_pct'):>22.0f}% / {u_saved:>9.0f}%")
    print(f"{len({r['case_id'] for r in results})} cases, {args.seeds} seeds each -> {path.relative_to(ROOT)}")
    if meta.get("mode") == "mock":
        print("MOCK run: plumbing check only, do not report these numbers.")


if __name__ == "__main__":
    main()
