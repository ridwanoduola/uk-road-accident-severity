"""Phase 9c: which inputs does the model actually need?  (development-set cross-validation only; test set untouched)

1) Re-trains the tuned HistGradientBoosting on smaller sets of input variables and compares it with the full model
   on the SAME folds (paired, corrected t-test).
2) 'Lookup-table' benchmark: predict the most common severity seen in training for each exact combination of the
   chosen variables. If the model does no better than the lookup table, it has extracted all the information those
   variables contain; the remaining errors are then limited by the data, not by the algorithm.
Run:  python -m src.ablation          Output: reports/ablation.txt
"""
import json
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from . import config as cfg
from .compare_stats import paired_tests
from .experiments import fold_metrics, get_splits, load_dev

CORE5 = ["speed_limit", "light_conditions", "vehicle_type", "weather_conditions", "road_surface_conditions"]
ALL_VARS = ["speed_limit", "age_of_oldest_driver"] + list(cfg.CATEGORICAL_COLS)
SCENE = [v for v in ALL_VARS if v not in cfg.POST_CRASH_COLS]
SETS = {
    "A  all 12 variables": ALL_VARS,
    "B  scene-only (no 4 post-crash variables)": SCENE,
    "C  B without driver sex and age": [v for v in SCENE if v not in ("sex_of_driver", "age_of_oldest_driver")],
    "D  core five": CORE5,
    "E  top three (vehicle, light, speed)": ["speed_limit", "light_conditions", "vehicle_type"],
}


def columns_for(variables):
    """Original variables -> (numeric, flag, categorical) model columns, including only derived features
    that can be computed from the chosen variables."""
    v = set(variables)
    num, flag, cat = [], [], []
    if "speed_limit" in v:
        num.append("speed_limit"); flag += ["speed_missing", "high_speed"]
    if "age_of_oldest_driver" in v:
        num.append("age_of_oldest_driver"); flag.append("age_missing")
    if "light_conditions" in v:
        flag.append("is_dark")
    if {"speed_limit", "light_conditions"} <= v:
        num.append("dark_x_speed")
    if set(cfg.POST_CRASH_COLS) <= v:
        num.append("n_crash_events")
    if v >= set(ALL_VARS):
        num.append("n_missing_fields")           # counts 'missing' across ALL categorical columns -> only valid for the full set
    if {"sex_of_driver", "age_of_oldest_driver"} <= v:
        flag.append("driver_unknown")
    cat = [c for c in cfg.CATEGORICAL_COLS if c in v]
    return num, flag, cat


def _key(df, cols):
    return df[cols].astype("string").fillna("NA").agg("|".join, axis=1).to_numpy()


def lookup_predict(train, y_train, val, cols):
    counts = {}
    for k, y in zip(_key(train, cols), y_train):
        counts.setdefault(k, np.zeros(3, dtype=int))[y] += 1
    fallback = int(np.bincount(y_train, minlength=3).argmax())
    pred = np.array([counts[k].argmax() if k in counts else fallback for k in _key(val, cols)])
    return pred, counts


def main():
    from .tune import DEFAULT, build
    dev, X, y = load_dev()
    splits = get_splits(dev, X, y)
    pf = cfg.MODEL_DIR / "hgb_best_params.json"
    params = {**DEFAULT, **json.loads(pf.read_text())} if pf.exists() else dict(DEFAULT)
    print(f"Development rows: {len(dev):,} | folds: {len(splits)} | HGB parameters: {'tuned' if pf.exists() else 'default'}\n")

    res, f1 = {}, {}
    for name, vars_ in SETS.items():
        num, flag, cat = columns_for(vars_)
        rows = []
        for tr, va in splits:
            pipe = build(params, num, flag, cat)
            pipe.fit(X.iloc[tr], y[tr], clf__sample_weight=compute_sample_weight("balanced", y[tr]))
            rows.append(fold_metrics(y[va], pipe.predict(X.iloc[va])))
        res[name] = pd.DataFrame(rows)
        f1[name] = res[name]["macro_F1"].to_numpy()
        print(f"  done: {name}", flush=True)

    ref = "A  all 12 variables"
    k = len(splits)
    lines = [f"ABLATION (tuned HGB, balanced weights, 5-fold CV on {len(dev):,} development rows; mean ± sd across folds)", "",
             f"{'input set':<46}{'#vars':>6}{'macro-F1':>16}{'accuracy':>16}{'fatal recall':>16}{'fatal prec.':>16}{'diff vs A':>11}{'p (corr.)':>11}"]
    for name, vars_ in SETS.items():
        d = res[name]
        mean, sd, _, p = paired_tests(f1[ref] - f1[name], 1 / (k - 1)) if name != ref else (0.0, 0.0, 1.0, 1.0)
        fmt = lambda c: f"{d[c].mean():.3f} ± {d[c].std(ddof=1):.3f}"
        lines.append(f"{name:<46}{len(vars_):>6}{fmt('macro_F1'):>16}{fmt('accuracy'):>16}{fmt('recall_fatal'):>16}"
                     f"{fmt('precision_fatal'):>16}{mean:>+11.4f}{p:>11.3f}")
    lines += ["", "diff vs A = macro-F1 of A minus macro-F1 of the smaller set (positive = the smaller set is worse).", ""]

    lines += ["LOOKUP-TABLE BENCHMARK (most common training label for each exact combination of the variables; same folds)",
              f"{'variables used':<46}{'#profiles (dev)':>16}{'macro-F1':>16}{'accuracy':>16}{'in-sample acc.':>16}"]
    for name in ["C  B without driver sex and age", "D  core five", "E  top three (vehicle, light, speed)"]:
        cols = SETS[name]
        rows = []
        for tr, va in splits:
            pred, _ = lookup_predict(dev.iloc[tr], y[tr], dev.iloc[va], cols)
            rows.append(fold_metrics(y[va], pred))
        d = pd.DataFrame(rows)
        _, counts = lookup_predict(dev, y, dev.iloc[:1], cols)
        insample = sum(a.max() for a in counts.values()) / len(dev)
        fmt = lambda c: f"{d[c].mean():.3f} ± {d[c].std(ddof=1):.3f}"
        lines.append(f"{name:<46}{len(counts):>16,}{fmt('macro_F1'):>16}{fmt('accuracy'):>16}{insample:>16.3f}")
    lines += ["", "in-sample acc. = accuracy of the lookup table on the very rows it was built from (an upper bound for these variables)."]
    text = "\n".join(lines)
    print("\n" + text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "ablation.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
