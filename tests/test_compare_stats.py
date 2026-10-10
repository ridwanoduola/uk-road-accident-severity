import numpy as np
from src.compare_stats import paired_tests


def test_corrected_test_is_more_conservative_than_naive():
    d = np.array([0.010, 0.008, 0.012, 0.006, 0.011])
    mean, sd, p_naive, p_corr = paired_tests(d, test_over_train=0.25)
    assert mean > 0 and p_corr > p_naive


def test_identical_models_give_p_one():
    _, _, p_naive, p_corr = paired_tests(np.zeros(5), 0.25)
    assert p_naive == 1.0 and p_corr == 1.0
