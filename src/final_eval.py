"""Phase 8: FINAL evaluation on the locked test set.  *** Run ONCE. ***

Models are fitted on the full development set (85%) and scored a single time on the held-out test set (15%).
Reports bootstrap 95% confidence intervals, paired bootstrap differences and McNemar tests.

Run:  python -m src.final_eval                     (needs PyTorch for the neural network)
      python -m src.final_eval --skip-nn           (traditional models only)
      python -m src.final_eval --nn-config "MLP embeddings [256,128]" --seeds 5
The script refuses to run a second time unless --force is given, to prevent repeatedly peeking at the test set.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from . import config as cfg
from .data import load_raw, clean
from .experiments import fold_metrics
from .features import ALL_FEATURES, add_features, build_preprocessor
from .split import make_split

N_BOOT = 2000
HGB_NAME = "HistGradientBoosting (tuned)"


# ----------------------------------------------------------------------------- fast metrics for bootstrapping
def cm_from(y, pred):
    return np.bincount(np.asarray(y) * 3 + np.asarray(pred), minlength=9).reshape(3, 3)


def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(0), cm.sum(1)
    prec = np.divide(tp, pred_n, out=np.zeros(3), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.zeros(3), where=true_n > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros(3), where=den > 0)
    return float(f1.mean())


def bootstrap(y, preds, n_boot=N_BOOT, seed=cfg.RANDOM_STATE):
    """Paired bootstrap: the same resampled rows are used for every model."""
    rng = np.random.default_rng(seed)
    n = len(y)
    f1 = {k: np.empty(n_boot) for k in preds}
    fatal = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, p in preds.items():
            cm = cm_from(y[idx], p[idx])
            f1[k][b] = macro_f1_from_cm(cm)
            fatal[k][b] = cm[2, 2] / max(cm[2].sum(), 1)
    return f1, fatal


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-config", default="MLP one-hot [256,128,64]")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    report_path = cfg.ROOT / "reports" / "final_test_results.txt"
    if report_path.exists() and not args.force:
        print(f"{report_path} already exists. The test set should be evaluated ONCE.\n"
              "Re-run with --force only if you are certain (and say so in your write-up).")
        return

    labeled, _, _ = clean(load_raw())
    df = add_features(labeled)
    dev_idx, test_idx = make_split(df, mode="random")
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    Xd, Xt = dev[ALL_FEATURES], test[ALL_FEATURES]
    m = {c: i for i, c in enumerate(cfg.CLASS_ORDER)}
    yd, yt = dev[cfg.TARGET].map(m).to_numpy(), test[cfg.TARGET].map(m).to_numpy()
    print(f"dev={len(dev):,}  test={len(test):,}")

    preds, probas = {}, {}
    preds["Majority class"] = np.full(len(yt), np.bincount(yd).argmax())

    lr = Pipeline([("prep", build_preprocessor("linear")),
                   ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]).fit(Xd, yd)
    preds["LogReg (balanced)"] = lr.predict(Xt)

    from .tune import DEFAULT, build as build_hgb          # lazy: tune imports optuna
    pfile = cfg.MODEL_DIR / "hgb_best_params.json"
    params = {**DEFAULT, **json.loads(pfile.read_text())} if pfile.exists() else dict(DEFAULT)
    print("HGB parameters:", "tuned (models/hgb_best_params.json)" if pfile.exists() else "default (no tuning file found)")
    hgb = build_hgb(params)
    hgb.fit(Xd, yd, clf__sample_weight=compute_sample_weight("balanced", yd))
    probas[HGB_NAME] = hgb.predict_proba(Xt)
    preds[HGB_NAME] = probas[HGB_NAME].argmax(1)

    nn_name, single_seed = None, None
    if not args.skip_nn:
        from .nn import CONFIGS, fit_predict, prepare
        c = CONFIGS[args.nn_config]
        data_tr, (data_te,), n_num, cards = prepare(c["kind"], Xd, [Xt])
        plist = []
        for s in range(args.seeds):
            print(f"  training neural network, seed {s + 1}/{args.seeds}", flush=True)
            plist.append(fit_predict(c, data_tr, yd, data_te, n_num, cards, seed=cfg.RANDOM_STATE + s, return_proba=True))
        single_seed = [fold_metrics(yt, p.argmax(1))["macro_F1"] for p in plist]
        nn_name = f"Neural net {args.nn_config} ({args.seeds}-seed ensemble)"
        probas[nn_name] = np.mean(plist, axis=0)
        preds[nn_name] = probas[nn_name].argmax(1)

    f1_b, fat_b = bootstrap(yt, preds)
    lines = [f"FINAL TEST-SET EVALUATION (evaluated once) | dev {len(dev):,} rows | test {len(test):,} rows | "
             f"bootstrap {N_BOOT} resamples", ""]
    hdr = f"{'model':<58}{'accuracy':>9}{'macro-F1 [95% CI]':>28}{'recall fatal [95% CI]':>28}{'prec fatal':>11}{'rec slight':>11}{'rec serious':>12}"
    lines.append(hdr)
    for k, p in preds.items():
        fm = fold_metrics(yt, p)
        lo, hi = ci(f1_b[k]); flo, fhi = ci(fat_b[k])
        lines.append(f"{k:<58}{fm['accuracy']:>9.3f}{fm['macro_F1']:>10.3f} [{lo:.3f}, {hi:.3f}]"
                     f"{fm['recall_fatal']:>12.3f} [{flo:.3f}, {fhi:.3f}]{fm['precision_fatal']:>11.3f}"
                     f"{fm['recall_slight']:>11.3f}{fm['recall_serious']:>12.3f}")
    if single_seed is not None:
        lines += ["", f"Neural net single-seed test macro-F1 over {len(single_seed)} seeds: "
                      f"mean {np.mean(single_seed):.3f}, sd {np.std(single_seed, ddof=1) if len(single_seed) > 1 else float('nan'):.4f}"]

    lines += ["", "PAIRED COMPARISONS vs " + HGB_NAME + " (positive = HGB better)"]
    for k in preds:
        if k in (HGB_NAME, "Majority class"):
            continue
        d = f1_b[HGB_NAME] - f1_b[k]
        lo, hi = ci(d)
        ok_h, ok_k = preds[HGB_NAME] == yt, preds[k] == yt
        b, c_ = int((ok_h & ~ok_k).sum()), int((~ok_h & ok_k).sum())
        p = binomtest(b, b + c_, 0.5).pvalue if b + c_ > 0 else 1.0
        lines.append(f"  vs {k}: macro-F1 diff {np.mean(d):+.4f} [95% CI {lo:+.4f}, {hi:+.4f}] | "
                     f"McNemar: HGB-only-correct={b}, other-only-correct={c_}, p={p:.3f}")

    lines += ["", "CONFUSION MATRICES (rows = true, cols = predicted; order: " + ", ".join(cfg.CLASS_ORDER) + ")"]
    for k, p in preds.items():
        if k != "Majority class":
            lines += [k, str(cm_from(yt, p)), ""]
    text = "\n".join(lines)
    print("\n" + text)

    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    out = pd.DataFrame({"y_true": [cfg.CLASS_ORDER[i] for i in yt]})
    for k, p in preds.items():
        out["pred_" + k] = [cfg.CLASS_ORDER[i] for i in p]
    for k, pr in probas.items():
        for j, cl in enumerate(cfg.CLASS_ORDER):
            out[f"proba_{cl}__{k}"] = pr[:, j]
    out.to_csv(cfg.ROOT / "reports" / "test_predictions.csv", index=False)
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": hgb, "features": ALL_FEATURES, "classes": cfg.CLASS_ORDER, "params": params},
                cfg.MODEL_DIR / "hgb_final.joblib")
    print("\nSaved reports/final_test_results.txt, reports/test_predictions.csv, models/hgb_final.joblib")


if __name__ == "__main__":
    main()
