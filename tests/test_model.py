"""Unit tests for the model factory in src/train_mlflow.py."""
import numpy as np
import pytest

from train_mlflow import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost", "knn", "lightgbm"])
def test_each_model_fits_and_returns_probabilities(name, X_y):
    X, y = X_y
    model = build_model(name, quick=True).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_stacking_ensemble_fits_and_predicts(X_y):
    X, y = X_y
    model = build_model("stacking", quick=True).fit(X, y)
    assert model.predict(X.head(10)).tolist() == model.predict(X.head(10)).tolist()  # deterministic
    assert set(np.unique(model.predict(X))) <= {0, 1}


def test_final_model_is_in_the_experiment_list():
    assert MODEL_NAMES[-1] == "stacking"
