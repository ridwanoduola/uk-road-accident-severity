"""Paired comparison of models from fold-level CV results.
Run:  python -m src.compare_stats [--mode random|grouped] [--metric macro_F1] [--ref "<model name>"]

Uses the SAME folds for every model, so differences are paired. With only 5 folds the naive paired t-test
is over-confident (training sets overlap), so we also report the Nadeau-Bengio corrected resampled t-test."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from . import config as cfg


def paired_tests(d: np.ndarray, test_over_train: float):
    k = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:
        return mean, d.std(ddof=1), 1.0, 1.0
    t_naive = mean / np.sqrt(var / k)
    t_corr = mean / np.sqrt((1 / k + test_over_train) * var)
    return mean, np.sqrt(var), 2 * t_dist.sf(abs(t_naive), k - 1), 2 * t_dist.sf(abs(t_corr), k - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "grouped"])
    ap.add_argument("--metric", default="macro_F1")
    ap.add_argument("--ref", default=None, help="reference model (default: best mean)")
    ap.add_argument("--with-nn", action="store_true", help="also include reports/nn_results_<mode>.csv (same folds)")
    args = ap.parse_args()

    path = cfg.ROOT / "reports" / f"model_comparison_{args.mode}.csv"
    res = pd.read_csv(path)
    if args.with_nn:
        res = pd.concat([res, pd.read_csv(cfg.ROOT / "reports" / f"nn_results_{args.mode}.csv")], ignore_index=True)
    wide = res.pivot(index="fold", columns="model", values=args.metric)
    k = len(wide)
    ref = args.ref or wide.mean().idxmax()
    ratio = 1 / (k - 1)  # n_test / n_train for k-fold CV

    rows = []
    for m in wide.columns:
        if m == ref:
            continue
        mean, sd, p_naive, p_corr = paired_tests((wide[ref] - wide[m]).to_numpy(), ratio)
        rows.append({"model": m, f"mean {args.metric} diff (ref - model)": round(mean, 4), "sd of diff": round(sd, 4),
                     "p (naive paired t)": round(p_naive, 3), "p (corrected t)": round(p_corr, 3),
                     "ref better in folds": f"{int((wide[ref] > wide[m]).sum())}/{k}"})
    out = pd.DataFrame(rows).sort_values("p (corrected t)")
    text = (f"Reference model: {ref}  (mean {args.metric} = {wide[ref].mean():.4f}) | mode={args.mode} | {k} folds\n"
            f"Positive diff = reference is better. Corrected t-test is the one to quote.\n\n{out.to_string(index=False)}")
    print(text)
    (cfg.ROOT / "reports" / f"paired_tests_{args.mode}{'_with_nn' if args.with_nn else ''}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
