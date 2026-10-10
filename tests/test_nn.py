import numpy as np
import pytest

torch = pytest.importorskip("torch")   # skipped automatically where PyTorch is not installed

from src import config as cfg
from src.features import ALL_FEATURES, add_features
from src.nn import CONFIGS, TabularMLP, fit_predict, prepare


def _xy(labeled_df):
    df = add_features(labeled_df)
    X = df[ALL_FEATURES]
    y = df[cfg.TARGET].map({c: i for i, c in enumerate(cfg.CLASS_ORDER)}).to_numpy()
    return X, y


@pytest.mark.parametrize("name", ["MLP one-hot [128,64]", "MLP embeddings [128,64]"])
def test_prepare_and_forward_shapes(labeled_df, name):
    X, y = _xy(labeled_df)
    c = CONFIGS[name]
    tr, (va,), n_num, cards = prepare(c["kind"], X.iloc[:400], [X.iloc[400:]])
    assert tr[0].dtype == np.float32 and tr[1].dtype == np.int64
    assert not np.isnan(tr[0]).any() and not np.isnan(va[0]).any()
    model = TabularMLP(n_num, cards, c["hidden"], c["dropout"]).eval()
    out = model(torch.as_tensor(va[0]), torch.as_tensor(va[1]))
    assert out.shape == (len(X) - 400, 3)


def test_fit_predict_runs_and_returns_valid_labels(labeled_df):
    X, y = _xy(labeled_df)
    c = CONFIGS["MLP embeddings [128,64]"]
    tr, (va,), n_num, cards = prepare(c["kind"], X.iloc[:450], [X.iloc[450:]])
    pred = fit_predict(c, tr, y[:450], va, n_num, cards, seed=0, max_epochs=2)
    assert pred.shape == (len(X) - 450,) and set(pred) <= {0, 1, 2}
