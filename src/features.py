"""Phase 4: row-wise feature engineering + leakage-safe preprocessing.

`add_features` only uses each row's own values (no statistics learned from data), so it is safe
to apply BEFORE splitting. Anything that learns from data (imputation medians, scaling, encoders)
lives in `build_preprocessor` and must be fitted inside a Pipeline on training data only.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from . import config as cfg

DARK = "darkness"
CRASH_EVENT_VALUES = {
    "skidding_and_overturning": "at least one vehicle skidded or overturned",
    "vehicle_leaving_carriageway": "at least one vehicle leaving carriageway",
    "hit_object_off_carriageway": "at least one vehicle hit an object",
}

CAT_FEATURES = list(cfg.CATEGORICAL_COLS)
NUM_FEATURES = ["speed_limit", "age_of_oldest_driver", "dark_x_speed", "n_crash_events", "n_missing_fields"]
FLAG_FEATURES = ["age_missing", "speed_missing", "is_dark", "high_speed", "driver_unknown"]
ALL_FEATURES = NUM_FEATURES + FLAG_FEATURES + CAT_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in cfg.NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out[CAT_FEATURES] = out[CAT_FEATURES].astype(object)
    if "age_missing" not in out:
        out["age_missing"] = out["age_of_oldest_driver"].isna().astype(int)
    if "speed_missing" not in out:
        out["speed_missing"] = out["speed_limit"].isna().astype(int)

    speed = out["speed_limit"]
    out["is_dark"] = (out["light_conditions"] == DARK).astype(int)
    out["high_speed"] = (speed >= 60).astype(int)
    out["dark_x_speed"] = out["is_dark"] * speed            # NaN stays NaN -> imputed in pipeline
    out["n_crash_events"] = sum((out[c] == v).astype(int) for c, v in CRASH_EVENT_VALUES.items())
    out["n_missing_fields"] = (out[CAT_FEATURES] == cfg.MISSING_LABEL).sum(axis=1).astype(int)
    out["driver_unknown"] = ((out["sex_of_driver"] == cfg.MISSING_LABEL) & (out["age_missing"] == 1)).astype(int)
    return out


def build_preprocessor(kind: str = "linear", num=None, flag=None, cat=None) -> ColumnTransformer:
    """kind='linear': impute + scale numerics, one-hot categoricals (logistic regression, neural nets).
    kind='tree'  : impute numerics, ordinal-encode categoricals (random forest, boosting)."""
    if kind not in {"linear", "tree"}:
        raise ValueError("kind must be 'linear' or 'tree'")
    num = NUM_FEATURES if num is None else list(num)
    flag = FLAG_FEATURES if flag is None else list(flag)
    cat = CAT_FEATURES if cat is None else list(cat)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if kind == "linear":
        num_steps.append(("scale", StandardScaler()))
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num),
         ("flag", "passthrough", flag),
         ("cat", cat_enc, cat)],
        verbose_feature_names_out=False,
    )


def core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for the deployed five-input model, computed from the five raw inputs ONLY
    (speed_limit, light_conditions, vehicle_type, weather_conditions, road_surface_conditions).
    Must stay identical to add_features() for these columns (tested), so training and serving agree."""
    out = pd.DataFrame(index=df.index)
    speed = pd.to_numeric(df["speed_limit"], errors="coerce").astype(float)
    out["speed_limit"] = speed
    out["speed_missing"] = speed.isna().astype(int)
    out["high_speed"] = (speed >= 60).astype(int)
    out["is_dark"] = (df["light_conditions"] == DARK).astype(int)
    out["dark_x_speed"] = out["is_dark"] * speed
    for c in ["light_conditions", "weather_conditions", "road_surface_conditions", "vehicle_type"]:
        out[c] = df[c].astype(object)
    return out
