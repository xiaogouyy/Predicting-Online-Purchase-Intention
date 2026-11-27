"""Model pipelines of the final model, moved out of notebooks/09_fine_tuning.ipynb.

src/train.py and src/evaluate.py both import from here, so the saved model can be loaded again
(pickle needs to find SessionFeatures and the column helpers in a normal module).
All settings come from configs/training.yaml.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler,
                                   StandardScaler)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "training.yaml"

# The 17 input columns, same names and order as in the notebooks
COUNTS = ["Administrative", "Informational", "ProductRelated"]
DURATIONS = [c + "_Duration" for c in COUNTS]
RAW_NUM = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated",
           "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
CATS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
FEATURES = RAW_NUM + CATS
# long-tailed columns that get log1p (KNN only)
LOG_SET = set(COUNTS + DURATIONS + ["PageValues", "TotalPages", "TotalDuration", "AvgTimePerPage", "AvgProductTime",
                                    "AvgAdminTime", "AvgInfoTime"])

SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


def load_config(path=CONFIG_FILE):
    """Read the YAML config into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


class SessionFeatures(BaseEstimator, TransformerMixin):
    """Row-wise step: categories as strings, optional engineered features, drop unused inputs.

    Nothing is learned in fit, so this step cannot leak information between folds.
    """

    def __init__(self, engineered=False, drop=()):
        self.engineered = engineered
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        z = X[FEATURES].copy()
        z[CATS] = z[CATS].astype(str).astype(object)
        if self.engineered:
            pages, dur = z[COUNTS].sum(axis=1), z[DURATIONS].sum(axis=1)
            z["TotalPages"], z["TotalDuration"] = pages, dur
            z["AvgTimePerPage"] = safe_div(dur, pages)
            z["AvgProductTime"] = safe_div(z["ProductRelated_Duration"], z["ProductRelated"])
            z["AvgAdminTime"] = safe_div(z["Administrative_Duration"], z["Administrative"])
            z["AvgInfoTime"] = safe_div(z["Informational_Duration"], z["Informational"])
            z["ProductPageShare"] = safe_div(z["ProductRelated"], pages)
            z["ProductTimeShare"] = safe_div(z["ProductRelated_Duration"], dur)
            z["NoTrackedPages"] = (pages == 0).astype(int)
            z["ZeroDurationSession"] = ((pages > 0) & (dur == 0)).astype(int)
            z["AnyInformational"] = (z["Informational"] > 0).astype(int)
            z["AnyAdministrative"] = (z["Administrative"] > 0).astype(int)
        return z.drop(columns=[c for c in self.drop if c in z.columns])


# column selectors for the ColumnTransformer (they look at the columns left after SessionFeatures)
def log_cols(df):
    return [c for c in df.columns if c in LOG_SET]


def lin_cols(df):
    return [c for c in df.select_dtypes("number").columns if c not in LOG_SET]


def num_cols(df):
    return list(df.select_dtypes("number").columns)


def cat_cols(df):
    return list(df.select_dtypes(exclude="number").columns)


def scale_by(X, factor=1.0):
    return X * factor


def cat_encoder(kind="onehot", rare=False):
    kw = {"min_frequency": 0.01} if rare else {}
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="infrequent_if_exist" if rare else "ignore", sparse_output=False, **kw)
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **kw)


def tree_prep(cat_encoding, rare):
    """Tree models: numbers unchanged, categories encoded."""
    return ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat_encoder(cat_encoding, rare), cat_cols)])


def scaled_prep(log, scaler, rare, cat_weight=1.0):
    """KNN: distances need every column on a similar scale."""
    return ColumnTransformer([
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p) if log else "passthrough"),
                          ("scale", SCALERS[scaler]())]), log_cols),
        ("linear", SCALERS[scaler](), lin_cols),
        ("cat", Pipeline([("encode", cat_encoder("onehot", rare)),
                          ("weight", FunctionTransformer(scale_by, kw_args={"factor": cat_weight}))]), cat_cols)])


def build_model(name, config):
    """Return one unfitted pipeline (features -> preprocessing -> classifier) for a model in the config."""
    spec = config["models"][name]
    seed = config["seed"]
    params = spec["params"]
    frame = SessionFeatures(engineered=spec["engineered"], drop=tuple(spec["drop"]))

    if name == "knn":
        prep = scaled_prep(spec["log"], spec["scaler"], spec["rare"], spec["cat_weight"])
        clf = KNeighborsClassifier(**params, n_jobs=1)
    else:
        prep = tree_prep(spec["cat_encoding"], spec["rare"])
        if name == "random_forest":
            clf = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
        elif name == "xgboost":
            clf = xgb.XGBClassifier(**params, random_state=seed, n_jobs=1)
        elif name == "lightgbm":
            clf = lgb.LGBMClassifier(**params, random_state=seed, n_jobs=1, verbose=-1)
        else:
            raise ValueError(f"Unknown model name: {name}")

    return Pipeline([("frame", frame), ("prep", prep), ("model", clf)])


def build_stacking(config):
    """The final model: the tuned members, combined by a logistic regression on their probabilities."""
    stack = config["stacking"]
    members = [(name, build_model(name, config)) for name in stack["members"]]
    return StackingClassifier(
        estimators=members,
        final_estimator=LogisticRegression(max_iter=stack["final_max_iter"]),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=stack["inner_cv_splits"], shuffle=True, random_state=config["seed"]),
    )
