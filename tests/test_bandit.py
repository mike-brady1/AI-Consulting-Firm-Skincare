"""Tests for src/bandit.py on synthetic data where the best arm is known (no LLM, no API key).

Run:  python tests/test_bandit.py      (or pytest tests/test_bandit.py)
Running it directly also prints the synthetic comparison: bandit vs uniform, winner found and calls used.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bandit import reward, run_bandit, run_uniform  # noqa: E402

N_PERSONAS = 30
NEVER_STOP = 1.01  # confidence threshold that can't be reached: spend the whole budget
SETUPS = {"clear": [0.75, 0.6, 0.55], "close": [0.68, 0.62, 0.55],
          "very close": [0.66, 0.63, 0.55], "one bad": [0.7, 0.66, 0.35]}


def synthetic(means, noise=0.12, seed=0):
    """Reward matrix: arm a, persona p -> clipped Gaussian around means[a]."""
    rng = random.Random(seed)
    return [[min(max(rng.gauss(m, noise), 0.0), 1.0) for _ in range(N_PERSONAS)] for m in means]


def evaluate(method, means, budget=30, confidence=0.95, runs=300):
    """(share of runs where the exhaustive winner is chosen, mean calls used)."""
    hits = calls = 0
    for seed in range(runs):
        rows = synthetic(means, seed=seed)
        best = max(range(len(rows)), key=lambda a: sum(rows[a]))
        rng = random.Random(10_000 + seed)
        order = list(range(N_PERSONAS))
        rng.shuffle(order)

        def pull(a, k):
            return rows[a][order[k]]

        if method == "bandit":
            res = run_bandit(len(rows), N_PERSONAS, pull, budget, rng, confidence=confidence)
        else:
            res = run_uniform(len(rows), N_PERSONAS, pull, budget, confidence=confidence)
        hits += res["choice"] == best
        calls += res["calls"]
    return hits / runs, calls / runs


def test_reward_range_and_clamping():
    assert reward(10, 10) == 1.0
    assert reward(0, 1) == 0.05
    assert reward(12, 0) == reward(10, 1)  # out-of-range scores are clamped


def test_budget_is_respected():
    rows = synthetic([0.7, 0.68, 0.66])
    res = run_bandit(3, N_PERSONAS, lambda a, k: rows[a][k], 25, random.Random(0), confidence=NEVER_STOP)
    assert res["calls"] == 25 and sum(res["pulls"]) == 25 and not res["stopped_early"]
    assert all(p >= 3 for p in res["pulls"])  # warm-up: every arm tried min_pulls times


def test_stops_when_every_persona_was_seen():
    res = run_bandit(2, 5, lambda a, k: 0.5, 1000, random.Random(0), confidence=NEVER_STOP)
    assert res["pulls"] == [5, 5] and res["calls"] == 10


def test_failed_calls_count_but_add_no_evidence():
    res = run_bandit(2, N_PERSONAS, lambda a, k: None, 10, random.Random(0))
    assert res["calls"] == 10 and res["means"] == [0.5, 0.5]


def test_stops_early_on_a_clear_winner():
    rows = synthetic([0.85, 0.5, 0.45])
    res = run_bandit(3, N_PERSONAS, lambda a, k: rows[a][k], 30, random.Random(0))
    assert res["choice"] == 0 and res["stopped_early"] and res["calls"] < 30


def test_single_scenario_needs_only_the_warm_up():
    res = run_bandit(1, N_PERSONAS, lambda a, k: 0.6, 30, random.Random(0))
    assert res["choice"] == 0 and res["calls"] == 3


def test_clear_winner_found_with_fewer_calls():
    found, calls = evaluate("bandit", SETUPS["clear"])
    assert found >= 0.95 and calls < 25


def test_not_worse_than_uniform_at_equal_budget():
    for means in SETUPS.values():
        bandit, _ = evaluate("bandit", means, confidence=NEVER_STOP)
        uniform, _ = evaluate("uniform", means, confidence=NEVER_STOP)
        assert bandit >= uniform - 0.03, (means, bandit, uniform)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed\n")
    print(f"Synthetic, 3 scenarios x {N_PERSONAS} personas (90 calls if exhaustive), cap 30 calls, 300 runs:")
    print(f"{'setup':>11} | {'bandit: winner / calls':>23} | {'uniform: winner / calls':>24}")
    for name, means in SETUPS.items():
        (bf, bc), (uf, uc) = evaluate("bandit", means), evaluate("uniform", means)
        print(f"{name:>11} | {bf:>12.0%} / {bc:>4.1f}    | {uf:>13.0%} / {uc:>4.1f}")
