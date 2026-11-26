"""Train the project's models, log them to MLflow, and register the final stacking ensemble.

Usage (from the repo root, after running 01_eda.ipynb so data/processed/train_raw.csv exists):
    python src/train_mlflow.py            # full run, a few minutes
    python src/train_mlflow.py --quick    # smaller forests, for a fast check
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000

Everything is stored locally (mlflow.db + mlruns/), so no account is needed.
Only the training set is used (5-fold CV). The test set stays untouched, as in 09_fine_tuning.
Hyperparameters are the tuned values printed in 09_fine_tuning.ipynb; the preprocessing is the shared
session_preprocessing pipeline, so CV scores can differ slightly from the notebook.
"""
import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from session_preprocessing import make_preprocessing_pipeline  # noqa: E402

SEED = 42
TARGET = "Revenue"
EXPERIMENT = "purchase-intent"
REGISTERED_NAME = "purchase-intent-stacking"


def base_estimators(quick=False):
    """Tuned hyperparameters from 09_fine_tuning.ipynb (quick=True shrinks the forests)."""
    k = 0.2 if quick else 1.0
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=int(500 * k), max_depth=10, min_samples_leaf=20, max_features=0.3,
            max_samples=0.6, criterion="entropy", random_state=SEED, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=int(175 * k), learning_rate=0.02, max_depth=6, min_child_weight=9.04,
            gamma=2.196, subsample=0.8006, colsample_bytree=0.8576, reg_alpha=0.0042,
            reg_lambda=0.6533, scale_pos_weight=2.32, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=65, p=1, weights="uniform"),
        "lightgbm": LGBMClassifier(
            n_estimators=int(350 * k), learning_rate=0.01, num_leaves=38, min_child_samples=100,
            min_split_gain=0.4833, subsample=0.9256, subsample_freq=1, colsample_bytree=0.9672,
            reg_alpha=2.5825, reg_lambda=0.1228, random_state=SEED, verbose=-1, n_jobs=-1),
    }


def build_model(name, quick=False):
    """Return an unfitted sklearn Pipeline: shared preprocessing + one model (or the stacking ensemble)."""
    est = base_estimators(quick)
    if name == "stacking":
        members = [(m, est[m]) for m in ["random_forest", "xgboost", "knn", "lightgbm"]]
        model = StackingClassifier(members, final_estimator=LogisticRegression(max_iter=1000),
                                   stack_method="predict_proba",
                                   cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    else:
        model = est[name]
    return Pipeline([("prep", make_preprocessing_pipeline(include_pagevalues=True)), ("model", model)])


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "knn", "lightgbm", "stacking"]


def run(train_path, quick=False, tracking_uri=f"sqlite:///{ROOT / 'mlflow.db'}"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    train = pd.read_csv(train_path)
    X = train.drop(columns=[c for c in [TARGET, "source_row_id"] if c in train])
    y = train[TARGET].astype(int).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for name in MODEL_NAMES:
        pipe = build_model(name, quick)
        with mlflow.start_run(run_name=name) as active:
            scores = cross_validate(pipe, X, y, cv=cv, scoring=["average_precision", "roc_auc"])
            mlflow.log_params({"model": name, "quick": quick, "cv_folds": 5, "include_pagevalues": True})
            mlflow.log_params({f"model__{k}": v for k, v in pipe.named_steps["model"].get_params().items()
                               if isinstance(v, (int, float, str, bool)) and not k.startswith("cv")})
            mlflow.log_metrics({
                "cv_ap_mean": float(np.mean(scores["test_average_precision"])),
                "cv_ap_sd": float(np.std(scores["test_average_precision"])),
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            })
            pipe.fit(X, y)
            p = pipe.predict_proba(X)[:, 1]
            mlflow.log_metrics({"train_ap": average_precision_score(y, p), "train_roc_auc": roc_auc_score(y, p)})
            mlflow.set_tag("stage", "final-model" if name == "stacking" else "baseline")
            mlflow.sklearn.log_model(
                pipe, name="model", input_example=X.head(3),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                code_paths=[str(ROOT / "notebooks" / "session_preprocessing.py")],  # model loads on its own
                signature=infer_signature(X.head(50), pipe.predict_proba(X.head(50))[:, 1]),
                registered_model_name=REGISTERED_NAME if name == "stacking" else None)
            print(f"{name:20s} CV AP = {np.mean(scores['test_average_precision']):.3f}  (run {active.info.run_id[:8]})")

    client = MlflowClient()
    latest = max(client.search_model_versions(f"name='{REGISTERED_NAME}'"), key=lambda v: int(v.version))
    client.set_registered_model_alias(REGISTERED_NAME, "champion", latest.version)
    print(f"Registered {REGISTERED_NAME} v{latest.version} with alias 'champion'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(ROOT / "data" / "processed" / "train_raw.csv"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.train, quick=args.quick)
