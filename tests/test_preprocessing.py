"""Unit tests for notebooks/session_preprocessing.py (the shared feature pipeline)."""
import numpy as np
import pandas as pd
import pytest

from session_preprocessing import (SessionFeatureEngineer, make_preprocessing_pipeline,
                                   safe_ratio)


def test_safe_ratio_returns_zero_when_denominator_is_zero():
    out = safe_ratio([10, 5, 0], [2, 0, 0])
    assert out.tolist() == [5.0, 0.0, 0.0]


def test_safe_ratio_keeps_missing_values_missing():
    out = safe_ratio([np.nan, 4], [2, 2])
    assert np.isnan(out[0]) and out[1] == 2.0


def test_feature_engineer_rejects_target_column(sessions):
    with pytest.raises(ValueError):
        SessionFeatureEngineer().fit(sessions)  # sessions still contains Revenue


def test_pagevalues_is_dropped_by_default(X_y):
    X, _ = X_y
    out = SessionFeatureEngineer().fit(X).transform(X)
    assert "PageValues" not in out.columns


def test_pagevalues_kept_when_requested(X_y):
    X, _ = X_y
    out = SessionFeatureEngineer(include_pagevalues=True).fit(X).transform(X)
    assert "PageValues" in out.columns


def test_engineered_totals_are_correct(X_y):
    X, _ = X_y
    out = SessionFeatureEngineer().fit(X).transform(X)
    expected = X["Administrative"] + X["Informational"] + X["ProductRelated"]
    assert np.allclose(out["TotalPages"], expected)


def test_pipeline_output_has_no_missing_values_and_fixed_width(X_y):
    X, y = X_y
    pipe = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    Xt = pipe.transform(X)
    assert not np.isnan(Xt).any()
    # an unseen category at prediction time must not change the number of columns
    X_new = X.head(5).copy()
    X_new["VisitorType"] = "Unseen_Type"
    assert pipe.transform(X_new).shape[1] == Xt.shape[1]
