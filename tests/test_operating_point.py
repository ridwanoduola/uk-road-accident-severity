import numpy as np
from src.operating_point import COST, cost_optimal, decide_with_threshold, ece, threshold_for_recall


def test_ece_small_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.03


def test_ece_large_for_overconfident_probabilities():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.2).astype(int)
    assert ece(y, np.full(5000, 0.9) + rng.random(5000) * 1e-6) > 0.6


def test_threshold_meets_target_recall():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 3000)
    p = np.where(y == 2, rng.beta(5, 2, 3000), rng.beta(2, 5, 3000))
    t = threshold_for_recall(p, y, 0.95)
    assert (p[y == 2] >= t).mean() >= 0.95


def test_decide_with_threshold():
    proba = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.45, 0.45, 0.10]])
    assert decide_with_threshold(proba, 0.35).tolist() == [0, 2, 0]


def test_cost_optimal_escalates_only_when_costs_justify_it():
    proba = np.array([[0.95, 0.04, 0.01],     # clearly slight  -> stay at slight
                      [0.30, 0.30, 0.40],     # 40% fatal       -> escalate to fatal
                      [0.70, 0.20, 0.10]])    # 10% fatal: under-triaging costs 10x, so "serious" is the cheaper call
    assert cost_optimal(proba).tolist() == [0, 2, 1]
