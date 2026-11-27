"""Unit tests for the model factory in src/train_mlflow.py."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from train_mlflow import MODEL_NAMES, PARAMS, QUICK_N_ESTIMATORS, build_model, get_params, params_to_log


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost", "knn", "lightgbm"])
def test_each_model_fits_and_returns_probabilities(name, X_y):
    X, y = X_y
    model = build_model(name, quick=True).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_array_equal(model.classes_, [0, 1])


def test_stacking_ensemble_fits_and_predicts(X_y):
    X, y = X_y
    model = build_model("stacking", quick=True).fit(X, y)
    repeated = clone(model).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X.head(10)), repeated.predict_proba(X.head(10)))
    assert set(np.unique(model.predict(X))) <= {0, 1}


def test_final_model_is_in_the_experiment_list():
    assert MODEL_NAMES[-1] == "stacking"


@pytest.mark.parametrize("name", ["knn", "random_forest"])
def test_knn_and_rf_prediction_contract(name, X_y):
    X, y = X_y
    model = build_model(name, quick=True)
    with pytest.raises(NotFittedError):
        model.predict_proba(X.head(1))
    model.fit(X, y)
    # Single-session and reordered-column serving must preserve the same prediction.
    expected = model.predict_proba(X.head(1))
    actual = model.predict_proba(X.head(1).loc[:, list(reversed(X.columns))])
    assert actual.shape == (1, 2)
    np.testing.assert_allclose(actual, expected)
    with pytest.raises((KeyError, ValueError)):
        model.predict_proba(X.head(1).drop(columns="ProductRelated"))
    with pytest.raises(ValueError):
        model.predict_proba(X.iloc[:0])


def test_unknown_model_fails_before_training():
    with pytest.raises(KeyError):
        build_model("not_a_model")


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "lightgbm"])
def test_quick_parameters_do_not_mutate_tuned_configuration(name):
    original = dict(PARAMS[name])
    quick = get_params(name, quick=True)
    assert quick["n_estimators"] == QUICK_N_ESTIMATORS
    quick["n_estimators"] = 1
    assert PARAMS[name] == original
    assert get_params(name) == original


def test_logged_parameters_do_not_pollute_model_configuration():
    logged = params_to_log("knn", quick=True)
    assert logged["cv_folds"] == 5
    assert logged["include_pagevalues"] is True
    assert "model" not in get_params("knn")
    ensemble = params_to_log("stacking", quick=False)
    assert ensemble["final_estimator"] == "logistic_regression"
    assert ensemble["members"] == "random_forest, xgboost, knn, lightgbm"
