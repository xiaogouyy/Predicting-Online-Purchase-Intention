"""Training orchestration unit tests with isolated I/O and MLflow calls."""
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

import train_mlflow as training


@pytest.mark.parametrize("quick", [False, True])
def test_training_only_cv_and_champion_registration(monkeypatch, sessions, tmp_path, quick):
    # The real classifiers are tested separately. Here no dataset, model binary,
    # tracking database or registry is read/written by the orchestration test.
    train = sessions.assign(source_row_id=np.arange(len(sessions)))
    expected_X = sessions.drop(columns="Revenue")
    read_csv = Mock(return_value=train)
    monkeypatch.setattr(training.pd, "read_csv", read_csv)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    training_path = tmp_path / "train_raw.csv"
    monkeypatch.setattr(training, "TRAIN_FILE", training_path)

    models = {name: MagicMock(name=name) for name in training.MODEL_NAMES}
    build_model = Mock(side_effect=lambda name, quick: models[name])
    monkeypatch.setattr(training, "build_model", build_model)
    scores = {
        "test_average_precision": np.array([0.6, 0.65, 0.7, 0.75, 0.8]),
        "test_roc_auc": np.array([0.8, 0.82, 0.84, 0.86, 0.88]),
    }
    cross_validate = Mock(return_value=scores)
    monkeypatch.setattr(training, "cross_validate", cross_validate)

    # Replace the module reference as one boundary. MLflow's lazy sklearn loader
    # can otherwise replace a patched attribute when it first imports the flavor.
    effects = {}
    for name in ["set_tracking_uri", "set_experiment", "start_run", "log_params", "log_metric"]:
        effects[name] = MagicMock()
    uris = [f"runs:/synthetic/{name}" for name in training.MODEL_NAMES]
    log_model = Mock(side_effect=[SimpleNamespace(model_uri=uri) for uri in uris])
    register = Mock(return_value=SimpleNamespace(version="7"))
    client = Mock()
    monkeypatch.setattr(training, "mlflow", SimpleNamespace(
        **effects,
        sklearn=SimpleNamespace(log_model=log_model, SERIALIZATION_FORMAT_CLOUDPICKLE="cloudpickle"),
        register_model=register,
        MlflowClient=Mock(return_value=client),
    ))

    training.main(quick=quick)

    # One training file only; target and source identifiers never reach CV/fit.
    read_csv.assert_called_once_with(training_path)
    assert cross_validate.call_count == len(training.MODEL_NAMES)
    for name, call in zip(training.MODEL_NAMES, cross_validate.call_args_list):
        assert call.args[0] is models[name]
        pd.testing.assert_frame_equal(call.args[1], expected_X)
        pd.testing.assert_series_equal(call.args[2], sessions["Revenue"])
        folds = call.kwargs["cv"]
        assert folds.n_splits == 5
        assert folds.shuffle is True
        assert folds.random_state == training.SEED
        assert call.kwargs["scoring"] == ["average_precision", "roc_auc"]
        models[name].fit.assert_called_once()
        pd.testing.assert_frame_equal(models[name].fit.call_args.args[0], expected_X)
        pd.testing.assert_series_equal(models[name].fit.call_args.args[1], sessions["Revenue"])
    assert [call.args for call in build_model.call_args_list] == [
        (name, quick) for name in training.MODEL_NAMES
    ]

    assert [call.kwargs["run_name"] for call in effects["start_run"].call_args_list] == training.MODEL_NAMES
    effects["set_experiment"].assert_called_once_with(training.EXPERIMENT_NAME)
    effects["set_tracking_uri"].assert_called_once_with(f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert len(effects["log_params"].call_args_list) == len(training.MODEL_NAMES)
    for name, call in zip(training.MODEL_NAMES, effects["log_params"].call_args_list):
        assert call.args[0]["model"] == name
        assert call.args[0]["quick"] is quick
        assert call.args[0]["cv_folds"] == 5
    for metric, expected in [
        ("cv_ap_mean", scores["test_average_precision"].mean()),
        ("cv_ap_sd", scores["test_average_precision"].std()),
        ("cv_roc_auc_mean", scores["test_roc_auc"].mean()),
        ("cv_roc_auc_sd", scores["test_roc_auc"].std()),
    ]:
        values = [call.args[1] for call in effects["log_metric"].call_args_list if call.args[0] == metric]
        np.testing.assert_allclose(values, np.full(len(training.MODEL_NAMES), expected))
    for name, call in zip(training.MODEL_NAMES, log_model.call_args_list):
        assert call.args[0] is models[name]
        assert call.kwargs["code_paths"] == [str(training.PREPROCESSING_FILE)]
        pd.testing.assert_frame_equal(call.kwargs["input_example"], expected_X.head(5))
    register.assert_called_once_with(uris[-1], training.REGISTERED_MODEL_NAME)
    client.set_registered_model_alias.assert_called_once_with(training.REGISTERED_MODEL_NAME, "champion", "7")
    assert not list(tmp_path.iterdir())
