"""Phase 7: neural networks (PyTorch) evaluated exactly like the traditional models.

Same development set, same 5 folds as src/experiments.py (so results can be paired), same metrics,
same balanced class weighting. Each configuration is trained with several random seeds per fold.

Run:  python -m src.nn                 (default: 3 seeds, 4 architectures)
      python -m src.nn --seeds 1 --only "one-hot [128,64]"     (quick check)
      python -m src.nn --grouped       (stricter profile-grouped folds)
Outputs: reports/nn_results_<mode>.csv (fold-level, same schema as model_comparison_<mode>.csv)
         reports/nn_seed_level_<mode>.csv, reports/nn_summary_<mode>.txt
The test set is NOT touched.
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from . import config as cfg
from .experiments import fold_metrics, get_splits, load_dev
from .features import FLAG_FEATURES, NUM_FEATURES, build_preprocessor

N_FRONT = len(NUM_FEATURES) + len(FLAG_FEATURES)
CONFIGS = {
    "MLP one-hot [128,64]":        dict(kind="onehot", hidden=[128, 64], dropout=0.2),
    "MLP one-hot [256,128,64]":    dict(kind="onehot", hidden=[256, 128, 64], dropout=0.3),
    "MLP embeddings [128,64]":     dict(kind="embed", hidden=[128, 64], dropout=0.2),
    "MLP embeddings [256,128]":    dict(kind="embed", hidden=[256, 128], dropout=0.3),
}


# ----------------------------------------------------------------------------- models
def _body(n_in, hidden, dropout, n_out=3):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class TabularMLP(nn.Module):
    """x_num: float features (one-hot block or scaled numerics); x_cat: integer codes (0 = unseen) or empty."""

    def __init__(self, n_num, cardinalities, hidden, dropout, emb_max=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, min(emb_max, max(2, (c + 1) // 2))) for c in cardinalities])
        n_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.net = _body(n_in, hidden, dropout)

    def forward(self, x_num, x_cat):
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat(parts, dim=1))


# ----------------------------------------------------------------------------- preprocessing (fit on training fold only)
def prepare(kind, X_train, X_others):
    """Returns (train_tuple, [other_tuples], n_num, cardinalities); each tuple = (x_num float32, x_cat int64)."""
    if kind == "onehot":
        pre = build_preprocessor("linear").fit(X_train)
        make = lambda X: (pre.transform(X).astype(np.float32), np.zeros((len(X), 0), dtype=np.int64))
        tr = make(X_train)
        return tr, [make(X) for X in X_others], tr[0].shape[1], []
    pre = build_preprocessor("tree").fit(X_train)
    scaler = StandardScaler().fit(pre.transform(X_train)[:, :N_FRONT])

    def make(X):
        Z = pre.transform(X)
        return (scaler.transform(Z[:, :N_FRONT]).astype(np.float32),
                np.clip(Z[:, N_FRONT:] + 1, 0, None).astype(np.int64))      # +1 so that unseen (-1) -> 0
    cards = [len(c) for c in pre.named_transformers_["cat"].categories_]
    return make(X_train), [make(X) for X in X_others], N_FRONT, cards


# ----------------------------------------------------------------------------- training
def fit_predict(cfg_, data_tr, y_tr, data_va, n_num, cards, seed, max_epochs=80, patience=8, lr=1e-3, wd=1e-4, bs=256,
                return_proba=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # inner stratified split used ONLY for early stopping
    idx_a, idx_b = train_test_split(np.arange(len(y_tr)), test_size=0.1, stratify=y_tr, random_state=seed)
    T = lambda a, dt: torch.as_tensor(a, dtype=dt)
    xa_n, xa_c, ya = T(data_tr[0][idx_a], torch.float32), T(data_tr[1][idx_a], torch.long), T(y_tr[idx_a], torch.long)
    xb_n, xb_c, yb = T(data_tr[0][idx_b], torch.float32), T(data_tr[1][idx_b], torch.long), T(y_tr[idx_b], torch.long)

    w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_tr[idx_a])
    crit = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    model = TabularMLP(n_num, cards, cfg_["hidden"], cfg_["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loader = DataLoader(TensorDataset(xa_n, xa_c, ya), batch_size=bs, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))

    best, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for n_, c_, y_ in loader:
            opt.zero_grad()
            crit(model(n_, c_), y_).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = crit(model(xb_n, xb_c), yb).item()
        sched.step(val)
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(T(data_va[0], torch.float32), T(data_va[1], torch.long))
    if return_proba:
        return torch.softmax(logits, dim=1).numpy()
    return logits.argmax(dim=1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default=None, help="run only configs whose name contains this text")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--smoke", action="store_true", help="30-second self-test: 1 config, 1 fold, 1 seed, 3 epochs; writes nothing")
    args = ap.parse_args()
    mode = "grouped" if args.grouped else "random"

    dev, X, y = load_dev()
    splits = get_splits(dev, X, y, grouped=args.grouped)
    print(f"Development rows: {len(dev):,} | folds ({mode}): {len(splits)} | seeds per fold: {args.seeds}\n")

    if args.smoke:
        tr, va = splits[0]
        for name, c in [(k, v) for k, v in CONFIGS.items() if k.endswith("[128,64]")]:
            data_tr, (data_va,), n_num, cards = prepare(c["kind"], X.iloc[tr], [X.iloc[va]])
            pred = fit_predict(c, data_tr, y[tr], data_va, n_num, cards, seed=0, max_epochs=3)
            m = fold_metrics(y[va], pred)
            print(f"smoke OK | {name}: input dims={n_num}+{len(cards)} embeddings | macro-F1 after 3 epochs={m['macro_F1']:.3f}")
        print("PyTorch pipeline works. Now run: python -m src.nn")
        return

    seed_rows = []
    for name, c in CONFIGS.items():
        if args.only and args.only.lower() not in name.lower():
            continue
        t0 = time.time()
        for fold, (tr, va) in enumerate(splits):
            data_tr, (data_va,), n_num, cards = prepare(c["kind"], X.iloc[tr], [X.iloc[va]])
            for s in range(args.seeds):
                pred = fit_predict(c, data_tr, y[tr], data_va, n_num, cards, seed=cfg.RANDOM_STATE + s, max_epochs=args.epochs)
                seed_rows.append({"model": name, "fold": fold, "seed": s, **fold_metrics(y[va], pred)})
        print(f"  done: {name:<28} {time.time() - t0:6.1f}s", flush=True)

    seed_df = pd.DataFrame(seed_rows)
    metrics = [m for m in seed_df.columns if m not in ("model", "fold", "seed")]
    fold_df = seed_df.groupby(["model", "fold"], as_index=False)[metrics].mean()      # average over seeds within a fold
    g = fold_df.groupby("model")[metrics]
    mean, std = g.mean(), g.std()
    seed_sd = seed_df.groupby(["model", "fold"])["macro_F1"].std().groupby("model").mean()
    table = pd.DataFrame({m: [f"{mean.loc[i, m]:.3f} ± {std.loc[i, m]:.3f}" for i in mean.index] for m in metrics}, index=mean.index)
    table["seed-to-seed sd (macro_F1)"] = seed_sd.round(4)
    table = table.loc[mean.sort_values("macro_F1", ascending=False).index]
    text = (f"Neural networks | {len(dev):,} dev rows | 5-fold CV ({mode}) | fold metrics = mean over {args.seeds} seeds; "
            f"± = sd across folds\n\n{table.to_string()}")
    print("\n" + text)
    (cfg.ROOT / "reports").mkdir(exist_ok=True)
    fold_df.to_csv(cfg.ROOT / "reports" / f"nn_results_{mode}.csv", index=False)
    seed_df.to_csv(cfg.ROOT / "reports" / f"nn_seed_level_{mode}.csv", index=False)
    (cfg.ROOT / "reports" / f"nn_summary_{mode}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
