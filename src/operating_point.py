"""Phase 9b: probability calibration, decision thresholds and cost-based decisions.

Design: thresholds are chosen on DEVELOPMENT data only (5-fold out-of-fold probabilities); the test set is used
only to report what those choices deliver. The class-weighted final model is compared with an UNWEIGHTED
HistGradientBoosting whose probabilities reflect the dataset's own class mix.
Run:  python -m src.operating_point        (needs models/hgb_final.joblib from src.final_eval)
Outputs: reports/operating_point.txt, reports/figures/60_calibration_fatal.png, 61_pr_curve_fatal.png
IMPORTANT: probabilities describe THIS dataset (about 20% fatal), not real-world UK prevalence.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from . import config as cfg
from .data import load_raw, clean
from .features import ALL_FEATURES, add_features
from .split import make_split

TARGET_RECALLS = [0.90, 0.95, 0.98]
PREVALENCES = [0.20, 0.05, 0.02, 0.01]
# ILLUSTRATIVE cost of predicting column k when the truth is row j (order: slight, serious, fatal).
# Replace with the emergency service's real costs. Under-triaging a fatal accident is the worst error.
COST = np.array([[0.0, 1.0, 2.0],
                 [3.0, 0.0, 1.0],
                 [10.0, 4.0, 0.0]])


def ece(y_bin, p, bins=10):
    """Expected calibration error with equal-frequency bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(p[chunk].mean() - y_bin[chunk].mean())
    return float(err)


def reliability_points(y_bin, p, bins=10):
    order = np.argsort(p)
    return [(p[c].mean(), y_bin[c].mean()) for c in np.array_split(order, bins) if len(c)]


def threshold_for_recall(p_fatal, y, target):
    """Largest threshold whose recall on these data is >= target (flag fatal when p >= threshold)."""
    return float(np.quantile(p_fatal[y == 2], 1 - target, method="lower"))


def decide_with_threshold(proba, t):
    pred = proba[:, :2].argmax(1)
    pred[proba[:, 2] >= t] = 2
    return pred


def cost_optimal(proba, cost=COST):
    return (proba @ cost).argmin(1)


def summarise(y, pred):
    tp = int(((pred == 2) & (y == 2)).sum()); fp = int(((pred == 2) & (y != 2)).sum())
    fn = int(((pred != 2) & (y == 2)).sum()); tn = int(((pred != 2) & (y != 2)).sum())
    return {"recall_fatal": tp / max(tp + fn, 1), "precision_fatal": tp / max(tp + fp, 1),
            "flagged_share": (tp + fp) / len(y), "false_alarms_per_fatal_caught": fp / max(tp, 1),
            "specificity": tn / max(tn + fp, 1), "accuracy": float((pred == y).mean()),
            "mean_cost": float(COST[y, pred].mean())}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .tune import DEFAULT, build as build_hgb

    labeled, _, _ = clean(load_raw())
    df = add_features(labeled)
    dev_idx, test_idx = make_split(df, mode="random")
    dev, test = df.iloc[dev_idx], df.iloc[test_idx]
    m = {c: i for i, c in enumerate(cfg.CLASS_ORDER)}
    Xd, Xt = dev[ALL_FEATURES], test[ALL_FEATURES]
    yd, yt = dev[cfg.TARGET].map(m).to_numpy(), test[cfg.TARGET].map(m).to_numpy()

    pf = cfg.MODEL_DIR / "hgb_best_params.json"
    params = {**DEFAULT, **json.loads(pf.read_text())} if pf.exists() else dict(DEFAULT)
    weighted = joblib.load(cfg.MODEL_DIR / "hgb_final.joblib")["pipeline"]
    p_w = weighted.predict_proba(Xt)

    print("fitting unweighted model and 5-fold out-of-fold probabilities on the development set ...")
    unw = build_hgb(params).fit(Xd, yd)
    p_u = unw.predict_proba(Xt)
    oof = np.zeros((len(yd), 3))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE).split(Xd, yd):
        oof[va] = build_hgb(params).fit(Xd.iloc[tr], yd[tr]).predict_proba(Xd.iloc[va])
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[:, 2], (yd == 2).astype(int))
    p_iso = iso.predict(p_u[:, 2])

    yb = (yt == 2).astype(int)
    out = [f"OPERATING POINT & CALIBRATION | dev {len(dev):,} | test {len(test):,} | fatal share in test {yb.mean():.1%}", ""]

    # 1) ranking quality and calibration of P(fatal)
    out += ["1) HOW WELL DOES THE MODEL RANK AND CALIBRATE P(fatal)?  (test set)",
            f"   ROC-AUC (fatal vs rest) {roc_auc_score(yb, p_u[:, 2]):.3f} | PR-AUC {average_precision_score(yb, p_u[:, 2]):.3f} "
            f"(a random ranking would give PR-AUC {yb.mean():.3f}) | 3-class one-vs-rest AUC {roc_auc_score(yt, p_u, multi_class='ovr'):.3f}",
            f"   {'probabilities':<42}{'Brier(fatal)':>14}{'ECE(fatal)':>12}{'3-class log-loss':>18}",
            f"   {'class-weighted final model':<42}{brier_score_loss(yb, p_w[:, 2]):>14.4f}{ece(yb, p_w[:, 2]):>12.4f}{log_loss(yt, p_w):>18.4f}",
            f"   {'unweighted model':<42}{brier_score_loss(yb, p_u[:, 2]):>14.4f}{ece(yb, p_u[:, 2]):>12.4f}{log_loss(yt, p_u):>18.4f}",
            f"   {'unweighted + isotonic (fitted on dev OOF)':<42}{brier_score_loss(yb, p_iso):>14.4f}{ece(yb, p_iso):>12.4f}{'':>18}",
            "   (lower is better; class weighting distorts probabilities, so use the unweighted model's probabilities)", ""]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for lab, p in [("class-weighted", p_w[:, 2]), ("unweighted", p_u[:, 2]), ("unweighted + isotonic", p_iso)]:
        pts = np.array(reliability_points(yb, p))
        ax.plot(pts[:, 0], pts[:, 1], "o-", label=lab)
    ax.set_xlabel("predicted P(fatal)"); ax.set_ylabel("observed fatal share"); ax.legend(); ax.set_title("Reliability of P(fatal), test set")
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIG_DIR / "60_calibration_fatal.png", bbox_inches="tight"); plt.close(fig)

    # 2) thresholds chosen on dev, evaluated on test
    rows = []
    ref = summarise(yt, p_w.argmax(1)); ref["rule"] = "class-weighted model, argmax (the final model)"; rows.append(ref)
    unw_s = summarise(yt, p_u.argmax(1)); unw_s["rule"] = "unweighted model, argmax"; rows.append(unw_s)
    marks = []
    for r in TARGET_RECALLS:
        t = threshold_for_recall(oof[:, 2], yd, r)
        s = summarise(yt, decide_with_threshold(p_u, t))
        s["rule"] = f"unweighted, flag fatal if P>={t:.3f} (dev recall target {r:.0%})"
        rows.append(s); marks.append((s["recall_fatal"], s["precision_fatal"], f"{r:.0%}"))
    tab = pd.DataFrame(rows).set_index("rule")
    out += ["2) DECISION THRESHOLDS FOR 'FATAL' (chosen on dev out-of-fold probabilities, reported on test)",
            tab[["recall_fatal", "precision_fatal", "flagged_share", "false_alarms_per_fatal_caught", "accuracy"]].round(3).to_string(), ""]

    prec, rec, _ = precision_recall_curve(yb, p_u[:, 2])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rec, prec, color="#2c7fb8", label="unweighted model")
    ax.scatter([ref["recall_fatal"]], [ref["precision_fatal"]], color="#c0392b", zorder=3, label="final (class-weighted) model")
    for rr, pp, lab in marks:
        ax.scatter([rr], [pp], color="k", s=18, zorder=3); ax.annotate(lab, (rr, pp), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("recall (fatal)"); ax.set_ylabel("precision (fatal)"); ax.set_title("Precision-recall for fatal, test set"); ax.legend()
    fig.savefig(cfg.FIG_DIR / "61_pr_curve_fatal.png", bbox_inches="tight"); plt.close(fig)

    # 3) what precision would look like at lower prevalence (illustration only)
    out += ["3) ILLUSTRATION: fatal precision if the same sensitivity/specificity held at other fatal shares",
            "   (assumption-heavy: real-world performance would differ; cite official statistics for the true prevalence)",
            f"   {'rule':<66}" + "".join(f"{'prev ' + format(p, '.0%'):>10}" for p in PREVALENCES)]
    for rule, r in tab.iterrows():
        sens, spec = r["recall_fatal"], r["specificity"]
        ppv = [sens * p / (sens * p + (1 - spec) * (1 - p)) for p in PREVALENCES]
        out.append(f"   {rule[:64]:<66}" + "".join(f"{v:>10.3f}" for v in ppv))
    out.append("")

    # 4) cost-based decisions
    rows = []
    for name, pred in [("unweighted argmax", p_u.argmax(1)), ("class-weighted final model", p_w.argmax(1)),
                       ("minimum expected cost (unweighted probabilities)", cost_optimal(p_u))]:
        s = summarise(yt, pred); s["rule"] = name; rows.append(s)
    ct = pd.DataFrame(rows).set_index("rule")
    out += ["4) COST-BASED DECISIONS  (ILLUSTRATIVE cost matrix, rows = truth, cols = prediction; order slight, serious, fatal)",
            "   " + str(COST.tolist()),
            ct[["mean_cost", "recall_fatal", "precision_fatal", "accuracy"]].round(3).to_string(), ""]
    text = "\n".join(out)
    print("\n" + text)
    (cfg.ROOT / "reports" / "operating_point.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
