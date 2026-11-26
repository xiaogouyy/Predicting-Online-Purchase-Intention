"""Log our models to MLflow and register the final stacking ensemble.

How to run (from the repo root, after 01_eda.ipynb has written data/processed/train_raw.csv):
    python src/train_mlflow.py --quick    # small models, about 1 minute: checks that everything works
    python src/train_mlflow.py            # tuned models, a few minutes
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000

What it does:
    1. For each of the 6 models: 5-fold cross-validation on the TRAINING set only,
       then one MLflow "run" with the parameters, the CV scores and the fitted model.
    2. Registers the stacking model in the Model Registry and gives it the alias "champion".

Everything is stored locally in mlflow.db (runs, metrics, registry) and mlruns/ (model files),
so no account is needed. The test set is NOT used here: it is evaluated once, in 09_fine_tuning.
"""
import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# The shared preprocessing lives in notebooks/, so we add that folder to Python's import path.
ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_FILE = ROOT / "notebooks" / "session_preprocessing.py"
sys.path.insert(0, str(PREPROCESSING_FILE.parent))
from session_preprocessing import make_preprocessing_pipeline  # noqa: E402

SEED = 42
TARGET = "Revenue"
EXPERIMENT_NAME = "purchase-intent"
REGISTERED_MODEL_NAME = "purchase-intent-stacking"
TRAIN_FILE = ROOT / "data" / "processed" / "train_raw.csv"

# The order in which the models are trained and logged. Stacking (the final model) comes last.
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "knn", "lightgbm", "stacking"]
STACKING_MEMBERS = ["random_forest", "xgboost", "knn", "lightgbm"]

# Tuned hyperparameters, copied from the outputs of notebooks/09_fine_tuning.ipynb.
# (Logistic regression was not tuned there, so it keeps simple defaults.)
PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "random_forest": {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 20, "max_features": 0.3,
                      "max_samples": 0.6, "criterion": "entropy"},
    "xgboost": {"n_estimators": 175, "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 9.04,
                "gamma": 2.196, "subsample": 0.8006, "colsample_bytree": 0.8576, "reg_alpha": 0.0042,
                "reg_lambda": 0.6533, "scale_pos_weight": 2.32},
    "knn": {"n_neighbors": 65, "p": 1, "weights": "uniform"},
    "lightgbm": {"n_estimators": 350, "learning_rate": 0.01, "num_leaves": 38, "min_child_samples": 100,
                 "min_split_gain": 0.4833, "subsample": 0.9256, "subsample_freq": 1, "colsample_bytree": 0.9672,
                 "reg_alpha": 2.5825, "reg_lambda": 0.1228},
}
QUICK_N_ESTIMATORS = 50  # with --quick, every tree model uses only 50 trees


def get_params(name, quick=False):
    """Return the hyperparameters for one model (fewer trees if quick=True)."""
    params = dict(PARAMS[name])  # a copy, so we never change PARAMS itself
    if quick and "n_estimators" in params:
        params["n_estimators"] = QUICK_N_ESTIMATORS
    return params


def make_classifier(name, quick=False):
    """Create one unfitted classifier from its name."""
    if name == "stacking":
        # Stacking: the 4 shortlisted models make predictions, and a logistic regression combines them.
        members = [(member, make_classifier(member, quick)) for member in STACKING_MEMBERS]
        return StackingClassifier(
            estimators=members,
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        )

    params = get_params(name, quick)
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)
    if name == "xgboost":
        return XGBClassifier(**params, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "lightgbm":
        return LGBMClassifier(**params, random_state=SEED, verbose=-1, n_jobs=-1)
    raise ValueError(f"Unknown model name: {name}")


def build_model(name, quick=False):
    """Return an unfitted pipeline: shared preprocessing, then the classifier."""
    return Pipeline([
        ("prep", make_preprocessing_pipeline(include_pagevalues=True)),
        ("model", make_classifier(name, quick)),
    ])


def params_to_log(name, quick):
    """The parameters we show in MLflow for one run."""
    if name == "stacking":
        params = {"members": ", ".join(STACKING_MEMBERS), "final_estimator": "logistic_regression"}
    else:
        params = get_params(name, quick)
    params["model"] = name
    params["quick"] = quick
    params["cv_folds"] = 5
    params["include_pagevalues"] = True
    return params


def main(quick=False):
    # Step 1: tell MLflow where to store everything, and which experiment to use (created if missing).
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Step 2: load the TRAINING data only. X = the 17 session features, y = 1 if the visitor bought.
    train = pd.read_csv(TRAIN_FILE)
    X = train.drop(columns=[TARGET, "source_row_id"], errors="ignore")
    y = train[TARGET].astype(int)

    # The same 5 folds for every model, so the scores are comparable.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    stacking_model_uri = None
    for name in MODEL_NAMES:
        print(f"Training {name} ...")
        model = build_model(name, quick)

        # Step 3: one MLflow run per model. Everything logged inside the "with" block belongs to this run.
        with mlflow.start_run(run_name=name):
            # 3a. Cross-validation: train on 4 folds, score on the 5th, 5 times.
            scores = cross_validate(model, X, y, cv=folds, scoring=["average_precision", "roc_auc"])
            ap = scores["test_average_precision"]
            auc = scores["test_roc_auc"]

            # 3b. Log the parameters and the CV metrics.
            mlflow.log_params(params_to_log(name, quick))
            mlflow.log_metric("cv_ap_mean", ap.mean())
            mlflow.log_metric("cv_ap_sd", ap.std())
            mlflow.log_metric("cv_roc_auc_mean", auc.mean())
            mlflow.log_metric("cv_roc_auc_sd", auc.std())

            # 3c. Fit on the whole training set and save the fitted pipeline as an MLflow model.
            #     cloudpickle + code_paths: our pipeline uses a custom class from session_preprocessing.py,
            #     so we save that file next to the model, and the model can be loaded anywhere.
            model.fit(X, y)
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                code_paths=[str(PREPROCESSING_FILE)],
                input_example=X.head(5),
            )
            if name == "stacking":
                stacking_model_uri = model_info.model_uri

        print(f"  CV AP = {ap.mean():.4f} (sd {ap.std():.4f}) | CV ROC-AUC = {auc.mean():.4f}")

    # Step 4: put the stacking model in the Model Registry and point the alias "champion" at it.
    version = mlflow.register_model(stacking_model_uri, REGISTERED_MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {version.version} with alias 'champion'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log the project's models to MLflow.")
    parser.add_argument("--quick", action="store_true", help="use small models for a fast test run")
    args = parser.parse_args()
    main(quick=args.quick)
