"""Phase 6: Optuna tuning of HistGradientBoosting (DEVELOPMENT set only; test set untouched).

Run:  python -m src.tune --trials 40        (try --trials 5 first to check everything works)
Objective: mean macro-F1 over the same 5 stratified folds used in experiments.py, with balanced sample weights.
Saves models/hgb_best_params.json and reports/tuning_hgb.txt.

Caution: the best CV score of a tuned model is slightly optimistic (it was selected on those folds).
The honest number is the single final evaluation on the held-out test set.
"""
import argparse
import json
import time
import warnings
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from . import config as cfg
from .experiments import RS, fold_metrics, get_splits, load_dev
from .features import CAT_FEATURES, FLAG_FEATURES, NUM_FEATURES, build_preprocessor

DEFAULT = dict(learning_rate=0.08, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20,
               l2_regularization=0.0, max_depth=None, native_cat=False)


def build(p, num=None, flag=None, cat=None):
    num = NUM_FEATURES if num is None else list(num)
    flag = FLAG_FEATURES if flag is None else list(flag)
    cat = CAT_FEATURES if cat is None else list(cat)
    n_front = len(num) + len(flag)
    mask = np.array([False] * n_front + [True] * len(cat)) if p["native_cat"] else None
    clf = HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        max_depth=p["max_depth"], categorical_features=mask, random_state=RS)
    return Pipeline([("prep", build_preprocessor("tree", num, flag, cat)), ("clf", clf)])


def evaluate(p, X, y, splits, trial=None, full=False):
    scores, rows = [], []
    for i, (tr, va) in enumerate(splits):
        pipe = build(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X.iloc[tr], y[tr], clf__sample_weight=compute_sample_weight("balanced", y[tr]))
            pred = pipe.predict(X.iloc[va])
        scores.append(f1_score(y[va], pred, average="macro"))
        if full:
            rows.append(fold_metrics(y[va], pred))
        if trial is not None:
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return (scores, rows) if full else scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dev, X, y = load_dev()
    splits = get_splits(dev, X, y)

    def objective(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 127),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_depth=trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
            native_cat=trial.suggest_categorical("native_cat", [False, True]),
        )
        return float(np.mean(evaluate(p, X, y, splits, trial)))

    def progress(study, trial):
        best = study.best_value if study.best_trial is not None else float("nan")
        print(f"  trial {trial.number + 1:>3}/{args.trials}  state={trial.state.name:<9} "
              f"value={trial.value if trial.value is not None else float('nan'):.4f}  best so far={best:.4f}", flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, callbacks=[progress])
    best = {**DEFAULT, **study.best_params}

    d_scores, d_rows = evaluate(DEFAULT, X, y, splits, full=True)
    b_scores, b_rows = evaluate(best, X, y, splits, full=True)
    diff = np.array(b_scores) - np.array(d_scores)

    def fmt(rows, key):
        v = np.array([r[key] for r in rows]); return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"

    lines = [f"HistGradientBoosting tuning | {len(dev):,} dev rows | {args.trials} trials | {time.time() - t0:.0f}s", "",
             "Best parameters:", json.dumps(best, indent=2), "",
             f"{'':<10}{'macro_F1':<18}{'accuracy':<18}{'recall_fatal':<18}{'precision_fatal':<18}"]
    for name, rows, sc in [("default", d_rows, d_scores), ("tuned", b_rows, b_scores)]:
        lines.append(f"{name:<10}{fmt(rows, 'macro_F1'):<18}{fmt(rows, 'accuracy'):<18}"
                     f"{fmt(rows, 'recall_fatal'):<18}{fmt(rows, 'precision_fatal'):<18}")
    lines += ["", f"Paired fold-level macro-F1 gain (tuned - default): mean {diff.mean():+.4f}, sd {diff.std(ddof=1):.4f}, "
              f"positive in {int((diff > 0).sum())}/{len(diff)} folds",
              "(optimistic: parameters were selected on these same folds; use the test set for the final number)"]
    text = "\n".join(lines)
    print("\n" + text)
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    (cfg.MODEL_DIR / "hgb_best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    (cfg.ROOT / "reports" / "tuning_hgb.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
