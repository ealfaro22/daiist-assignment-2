"""
The cost model, shared by train.py (to sweep thresholds) and app.py (to
recompute everything live when the thresholds move).

Each fruit has a rot score in [0, 1]. Two thresholds split it three ways:

    score <  t_low            -> pass   (packed)
    t_low <= score < t_high   -> review (a person decides)
    score >= t_high           -> fail   (discarded)

With t_low == t_high there is no review band and the model decides alone.
The reviewer is assumed to always decide correctly.

Rates are measured per true class on the test split and then re-weighted to
the production base rate, because the test split is ~44% rotten while the
line is assumed to be ~5% rotten.
"""

import numpy as np

# Figures and sources are argued in REPORT.md.
COST_FP = 0.18        # good fruit discarded, euros
COST_FN = 10.0        # rotten fruit packed (carton-level damage), euros
COST_TP = 0.0         # rotten fruit discarded: it had no value anyway
COST_TN = 0.0         # good fruit packed
BASE_RATE = 0.05      # share of fruit on the line that is rotten

REVIEWER_EUR_PER_HOUR = 12.0
REVIEW_SECONDS = 5.0
COST_REVIEW = REVIEWER_EUR_PER_HOUR * REVIEW_SECONDS / 3600

FRUITS_PER_DAY = 288_000           # ~10 fruits/s over an 8 h shift
REVIEWER_HOURS_PER_DAY = 7.0       # productive hours in an 8 h shift

FN_SCENARIOS = {"A: consumer complaint": 1.5, "B: carton damage": 10.0, "C: load rejection": 20.0}
BASE_RATE_SCENARIOS = (0.02, 0.05, 0.10, 0.15)

# Finer steps below 0.1, where a high FN cost pushes the optimum.
THRESHOLDS = np.unique(np.round(np.concatenate([np.linspace(0, 0.1, 51), np.linspace(0.1, 1, 91)]), 3))


def outcome_rates(scores, is_rotten, t_low, t_high) -> dict:
    """Share of rotten and of good fruit that ends up in pass / review / fail."""
    scores = np.asarray(scores, dtype=float)
    is_rotten = np.asarray(is_rotten, dtype=bool)
    rates = {}
    for name, mask in (("rotten", is_rotten), ("good", ~is_rotten)):
        s = scores[mask]
        n = max(len(s), 1)
        fail = np.count_nonzero(s >= t_high) / n
        passed = np.count_nonzero(s < t_low) / n
        rates[name] = {"pass": passed, "review": 1 - fail - passed, "fail": fail}
    return rates


def per_1000(rates: dict, base_rate: float = BASE_RATE) -> dict:
    """Expected outcome counts per 1,000 fruits at the production base rate."""
    rot, good = rates["rotten"], rates["good"]
    n_rot, n_good = 1000 * base_rate, 1000 * (1 - base_rate)
    return {
        "TP": n_rot * rot["fail"],
        "FN": n_rot * rot["pass"],
        "review_rotten": n_rot * rot["review"],
        "FP": n_good * good["fail"],
        "TN": n_good * good["pass"],
        "review_good": n_good * good["review"],
    }


def cost_per_1000(rates: dict, base_rate: float = BASE_RATE, cost_fn: float = COST_FN,
                  cost_fp: float = COST_FP, cost_review: float = COST_REVIEW) -> float:
    c = per_1000(rates, base_rate)
    return (c["TP"] * COST_TP + c["FN"] * cost_fn + c["FP"] * cost_fp + c["TN"] * COST_TN
            + (c["review_rotten"] + c["review_good"]) * cost_review)


def review_share(rates: dict, base_rate: float = BASE_RATE) -> float:
    return base_rate * rates["rotten"]["review"] + (1 - base_rate) * rates["good"]["review"]


def reviewers_needed(share: float) -> float:
    hours = FRUITS_PER_DAY * share * REVIEW_SECONDS / 3600
    return hours / REVIEWER_HOURS_PER_DAY


def sweep(scores, is_rotten, cost_fn: float = COST_FN, base_rate: float = BASE_RATE) -> np.ndarray:
    """Cost per 1,000 fruits at each single threshold (no review band)."""
    return np.array([cost_per_1000(outcome_rates(scores, is_rotten, t, t), base_rate, cost_fn)
                     for t in THRESHOLDS])


# A threshold is only trusted if the cost stays low when it is nudged: on ~700
# test fruits the literal minimum can hinge on a single fruit (here a mislabelled
# banana with rot score 0.004). Each candidate is judged by the worst cost
# within +/- ROBUST_WINDOW of it, and ties go to the middle of the tied range.
ROBUST_WINDOW = 0.05


def _neighbours(i: int) -> np.ndarray:
    return np.where(np.abs(THRESHOLDS - THRESHOLDS[i]) <= ROBUST_WINDOW + 1e-9)[0]


def _middle_of(indices: np.ndarray) -> int:
    midpoint = (THRESHOLDS[indices].min() + THRESHOLDS[indices].max()) / 2
    return int(indices[np.argmin(np.abs(THRESHOLDS[indices] - midpoint))])


def robust_threshold(costs_by_threshold: np.ndarray) -> dict:
    """Threshold with the lowest worst-case cost in its neighbourhood."""
    worst = np.array([costs_by_threshold[_neighbours(i)].max() for i in range(len(THRESHOLDS))])
    tied = np.where(np.isclose(worst, worst.min()))[0]
    i = _middle_of(tied)
    return {"threshold": float(THRESHOLDS[i]), "cost_per_1000": float(costs_by_threshold[i]),
            "worst_cost_in_window": float(worst[i]), "window": ROBUST_WINDOW,
            "tied_range": [float(THRESHOLDS[tied].min()), float(THRESHOLDS[tied].max())],
            "literal_minimum": float(THRESHOLDS[np.argmin(costs_by_threshold)]),
            "literal_minimum_cost": float(costs_by_threshold.min())}


def best_band(scores, is_rotten, max_review_share: float = 1.0) -> dict:
    """(t_low, t_high) with the lowest worst-case cost when both edges are nudged,
    among bands whose review share stays within the cap."""
    n = len(THRESHOLDS)
    cost = np.full((n, n), np.inf)
    share = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            rates = outcome_rates(scores, is_rotten, min(THRESHOLDS[i], THRESHOLDS[j]),
                                  max(THRESHOLDS[i], THRESHOLDS[j]))
            cost[i, j], share[i, j] = cost_per_1000(rates), review_share(rates)
    neighbours = [_neighbours(i) for i in range(n)]
    worst = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i, n):
            if share[i, j] <= max_review_share + 1e-12:
                worst[i, j] = cost[np.ix_(neighbours[i], neighbours[j])].max()
    tied = np.argwhere(np.isclose(worst, worst.min()))
    # Among ties, the band that sends fewest fruit to a person, then the middle one.
    fewest = tied[np.isclose(share[tied[:, 0], tied[:, 1]], share[tied[:, 0], tied[:, 1]].min())]
    i, j = fewest[len(fewest) // 2]
    return {"t_low": float(THRESHOLDS[i]), "t_high": float(THRESHOLDS[j]),
            "cost_per_1000": float(cost[i, j]), "worst_cost_in_window": float(worst[i, j]),
            "review_share": float(share[i, j])}
