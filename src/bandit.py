"""Top-two Thompson sampling with early stopping, to spend as few persona calls as possible.

Each compliant scenario is an arm; one pull = one simulated customer (persona) reading that scenario.
Reward in [0, 1] = (auditor KPI score + purchase intent) / 20: an arm's mean reward is the pipeline's
combined score / 10, so the bandit looks for the arm the exhaustive mode would rank first.

- Posterior per arm: Gaussian, mean of its rewards, sd = sqrt(max(variance, FLOOR^2) / n). A Beta update
  treats each 0-1 score as a coin flip, which is far too wide for scores that vary as little as these.
- Allocation: top-two Thompson sampling (Russo, 2016). Draw from every posterior and pull the leader or,
  half of the time, the best challenger. Plain Thompson sampling keeps pulling the leader and under-tests
  the runner-up, which is what decides the winner.
- Stop as soon as P(leader is best) >= confidence (Bonferroni lower bound over pairwise Gaussian
  comparisons), or when the budget is spent.

Arms are compared on the same customers in the same order (paired design): the k-th pull of every arm
shows its scenario to persona order[k]. An arm that has seen every persona is closed.
`run_uniform` (round-robin, same warm-up and stopping rule) is the control the bandit has to beat.
No LLM here: a classic algorithm, tested on synthetic data in tests/test_bandit.py.
"""
import math
import random
import statistics
from typing import Callable, Optional

Pull = Callable[[int, int], Optional[float]]  # (arm, k-th persona of that arm) -> reward, or None if the call failed
FLOOR = 0.05        # one purchase-intent point on the reward scale
CONFIDENCE = 0.95
MIN_PULLS = 3


def reward(kpi_score: float, intent: float) -> float:
    """Combine the auditor's KPI score (0-10) and one persona's purchase intent (1-10) into [0, 1]."""
    kpi = min(max(float(kpi_score), 0.0), 10.0)
    intent = min(max(float(intent), 1.0), 10.0)
    return (kpi + intent) / 20.0


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class _Arms:
    def __init__(self, n_arms: int, n_personas: int, pull: Pull):
        self.n_personas, self._pull = n_personas, pull
        self.rewards: list[list[float]] = [[] for _ in range(n_arms)]
        self.pulls, self.calls, self.stopped_early = [0] * n_arms, 0, False

    def open(self) -> list[int]:
        return [a for a in range(len(self.pulls)) if self.pulls[a] < self.n_personas]

    def pull(self, arm: int) -> None:
        r = self._pull(arm, self.pulls[arm])
        self.pulls[arm] += 1
        self.calls += 1
        if r is not None:
            self.rewards[arm].append(r)

    def warm_up(self, min_pulls: int, budget: int) -> None:
        for _ in range(min_pulls):
            for a in range(len(self.pulls)):
                if self.calls < budget and self.pulls[a] < self.n_personas:
                    self.pull(a)

    def mean(self, a: int) -> float:
        x = self.rewards[a]
        return sum(x) / len(x) if x else 0.5

    def sd(self, a: int) -> float:
        x = self.rewards[a]
        if not x:
            return 0.5
        var = statistics.variance(x) if len(x) > 1 else 0.0
        return math.sqrt(max(var, FLOOR ** 2) / len(x))

    def leader(self) -> Optional[int]:
        return max(range(len(self.pulls)), key=self.mean) if self.pulls else None

    def p_best(self) -> float:
        """Lower bound on P(leader is best): 1 - sum of P(other arm beats leader)."""
        lead = self.leader()
        if lead is None:
            return 0.0
        miss = sum(_phi(-(self.mean(lead) - self.mean(b)) / math.hypot(self.sd(lead), self.sd(b)))
                   for b in range(len(self.pulls)) if b != lead)
        return max(0.0, 1.0 - miss)

    def done(self, budget: int, confidence: float) -> bool:
        if self.calls >= budget or not self.open():
            return True
        if self.p_best() >= confidence:
            self.stopped_early = True
            return True
        return False

    def result(self) -> dict:
        return {"pulls": self.pulls, "calls": self.calls, "choice": self.leader(),
                "means": [round(self.mean(a), 4) for a in range(len(self.pulls))],
                "p_best": round(self.p_best(), 4), "stopped_early": self.stopped_early}


def run_bandit(n_arms: int, n_personas: int, pull: Pull, budget: int, rng: random.Random,
               min_pulls: int = MIN_PULLS, confidence: float = CONFIDENCE, top_two: float = 0.5) -> dict:
    """Warm-up round-robin, then top-two Thompson sampling until confident or out of budget."""
    arms = _Arms(n_arms, n_personas, pull)
    arms.warm_up(min_pulls, budget)
    while not arms.done(budget, confidence):
        candidates = arms.open()

        def best_draw() -> int:
            draws = {a: rng.gauss(arms.mean(a), arms.sd(a)) for a in candidates}
            return max(draws, key=draws.get)

        arm = best_draw()
        if len(candidates) > 1 and rng.random() < top_two:
            for _ in range(100):  # resample until another arm wins the draw: the challenger
                challenger = best_draw()
                if challenger != arm:
                    arm = challenger
                    break
        arms.pull(arm)
    return arms.result()


def run_uniform(n_arms: int, n_personas: int, pull: Pull, budget: int,
                min_pulls: int = MIN_PULLS, confidence: float = CONFIDENCE) -> dict:
    """Control: same warm-up and stopping rule, but the budget is spread evenly (round-robin)."""
    arms = _Arms(n_arms, n_personas, pull)
    arms.warm_up(min_pulls, budget)
    while not arms.done(budget, confidence):
        arms.pull(min(arms.open(), key=lambda a: arms.pulls[a]))
    return arms.result()
