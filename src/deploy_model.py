"""Phase 10: build the DEPLOYMENT ("triage") model.   *** Run ONCE (it reports test-set results). ***

Design decisions (all made from development-set cross-validation, see reports/ablation.txt):
  * Inputs: the five core variables only (speed limit, light, vehicle type, weather, road surface). All are known
    at the scene; driver sex/age and post-crash variables are excluded.
  * Model: tuned HistGradientBoosting WITHOUT class weights, so its probabilities are calibrated to the dataset.
  * Sensitivity levels: probability thresholds for 'fatal' chosen on development out-of-fold predictions so that
    recall reaches 90% / 95% / 98%. The test set is only used to report what each level delivers.
Run:  python -m src.deploy_model            (refuses to run twice unless --force)
Outputs: models/triage_model.joblib, reports/deploy_model_test_results.txt
"""
import argparse
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from . import config as cfg
from .ablation import CORE5, columns_for
from .data import load_raw, clean
from .experiments import fold_metrics
from .features import core_features
from .operating_point import decide_with_threshold, ece, summarise, threshold_for_recall
from .split import make_split

LEVELS = {"Standard": 0.90, "High": 0.95, "Maximum": 0.98}      # target recall for fatal accidents (on dev)
SPEED_CHOICES = [20, 30, 40, 50, 60, 70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    report_path = cfg.ROOT / "reports" / "deploy_model_test_results.txt"
    if report_path.exists() and not args.force:
        print(f"{report_path} exists: the test set was already used for this model. Use --force only if you must.")
        return
    from .tune import DEFAULT, build

    labeled, _, _ = clean(load_raw())
    dev_idx, test_idx = make_split(labeled, mode="random")        # identical split to every earlier phase
    dev, test = labeled.iloc[dev_idx].reset_index(drop=True), labeled.iloc[test_idx].reset_index(drop=True)
    m = {c: i for i, c in enumerate(cfg.CLASS_ORDER)}
    yd, yt = dev[cfg.TARGET].map(m).to_numpy(), test[cfg.TARGET].map(m).to_numpy()
    Xd, Xt = core_features(dev), core_features(test)
    num, flag, cat = columns_for(CORE5)
    pf = cfg.MODEL_DIR / "hgb_best_params.json"
    params = {**DEFAULT, **json.loads(pf.read_text())} if pf.exists() else dict(DEFAULT)

    print("out-of-fold probabilities on the development set ...")
    oof = np.zeros((len(yd), 3))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE).split(Xd, yd):
        oof[va] = build(params, num, flag, cat).fit(Xd.iloc[tr], yd[tr]).predict_proba(Xd.iloc[va])
    print("fitting the deployment model on the full development set ...")
    pipe = build(params, num, flag, cat).fit(Xd, yd)
    p_test = pipe.predict_proba(Xt)

    levels, lines = {}, []
    yb = (yt == 2).astype(int)
    lines += [f"DEPLOYMENT MODEL | inputs: {', '.join(CORE5)}", f"train {len(dev):,} rows | test {len(test):,} rows (fatal share {yb.mean():.1%})", "",
              f"Test-set quality of P(fatal): ROC-AUC {roc_auc_score(yb, p_test[:, 2]):.3f} | Brier {brier_score_loss(yb, p_test[:, 2]):.4f} | "
              f"ECE {ece(yb, p_test[:, 2]):.4f}   (out-of-fold on dev: ECE {ece((yd == 2).astype(int), oof[:, 2]):.4f})", "",
              f"{'sensitivity level':<22}{'threshold':>10}{'fatal recall':>14}{'fatal prec.':>13}{'flagged':>9}{'false alarms/fatal caught':>27}{'accuracy':>10}{'macro-F1':>10}"]
    for name, r in {"Argmax (no threshold)": None, **LEVELS}.items():
        if r is None:
            pred, t = p_test.argmax(1), None
        else:
            t = threshold_for_recall(oof[:, 2], yd, r)
            pred = decide_with_threshold(p_test, t)
        s = summarise(yt, pred)
        s["macro_F1"] = fold_metrics(yt, pred)["macro_F1"]
        lines.append(f"{name:<22}{('%.3f' % t) if t is not None else '-':>10}{s['recall_fatal']:>14.3f}{s['precision_fatal']:>13.3f}"
                     f"{s['flagged_share']:>9.3f}{s['false_alarms_per_fatal_caught']:>27.2f}{s['accuracy']:>10.3f}{s['macro_F1']:>10.3f}")
        if r is not None:
            levels[name] = {"target_recall": r, "threshold": t,
                            "test": {k: float(s[k]) for k in ["recall_fatal", "precision_fatal", "flagged_share",
                                                              "false_alarms_per_fatal_caught", "accuracy", "macro_F1"]}}
    # transparent lookup table: observed outcomes for each exact combination of the five inputs (development data)
    lookup = (dev.assign(n=1).pivot_table(index=CORE5, columns=cfg.TARGET, values="n", aggfunc="sum", fill_value=0, dropna=False)
              .reindex(columns=cfg.CLASS_ORDER, fill_value=0).reset_index())
    lookup = lookup[lookup[cfg.CLASS_ORDER].sum(axis=1) > 0]
    lookup["speed_limit"] = lookup["speed_limit"].astype(float)
    choices = {c: sorted(dev[c].dropna().unique().tolist()) for c in CORE5 if c != "speed_limit"}
    choices["speed_limit"] = SPEED_CHOICES
    artifact = {
        "pipeline": pipe, "inputs": CORE5, "classes": cfg.CLASS_ORDER, "levels": levels, "choices": choices, "lookup": lookup,
        "background": dev[CORE5].sample(500, random_state=cfg.RANDOM_STATE).reset_index(drop=True),
        "train_fatal_share": float((yd == 2).mean()), "n_train": len(dev), "n_test": len(test), "params": params,
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "python": platform.python_version()},
    }
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, cfg.MODEL_DIR / "triage_model.joblib", compress=3)
    lines += ["", f"saved models/triage_model.joblib | lookup table rows: {len(lookup):,} | versions: {artifact['versions']}"]
    text = "\n".join(lines)
    print("\n" + text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
