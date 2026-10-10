"""Exact Shapley-value explanations for a model with only a handful of inputs (here 5 -> 32 coalitions).

For one case x and a target quantity f (e.g. P(fatal)):
    v(S) = average of f over background cases whose inputs in S are replaced by x's values.
    phi_i = weighted average of v(S + i) - v(S) over all subsets S that exclude input i.
Properties: sum(phi) = f(x) - v(empty) (the case's probability minus the average background probability).
No external SHAP dependency is needed because the number of inputs is tiny.
"""
from itertools import combinations
from math import factorial


def shapley_values(predict_fn, x, background, variables):
    """predict_fn(DataFrame of raw inputs) -> 1-D array of the quantity to explain.
    x: mapping variable -> value. background: DataFrame containing at least `variables`.
    Returns (phi: dict, base_value, prediction)."""
    n = len(variables)
    cache = {}

    def v(S):
        S = tuple(sorted(S))
        if S not in cache:
            df = background[variables].copy()
            for j in S:
                df[variables[j]] = x[variables[j]]
            cache[S] = float(predict_fn(df).mean())
        return cache[S]

    phi = {}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        total = 0.0
        for k in range(len(others) + 1):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                total += w * (v(S + (i,)) - v(S))
        phi[variables[i]] = total
    return phi, v(()), v(tuple(range(n)))
