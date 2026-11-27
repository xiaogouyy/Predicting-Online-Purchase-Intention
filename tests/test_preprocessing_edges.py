"""Missing data, zero activity and prediction-time category regressions."""
import numpy as np
import pytest

from session_preprocessing import SessionFeatureEngineer, make_preprocessing_pipeline, safe_ratio


def test_safe_ratio_preserves_missing_denominators_and_zero_over_missing():
    result = safe_ratio([4, 0, np.nan, 5], [np.nan, np.nan, 0, 2])
    np.testing.assert_allclose(result, [np.nan, np.nan, np.nan, 2.5], equal_nan=True)


def test_safe_ratio_accepts_empty_arrays():
    result = safe_ratio([], [])
    assert result.shape == (0,)


def test_zero_activity_has_finite_ratios_and_correct_flags(X_y):
    X, _ = X_y
    X = X.head(2).copy()
    for column in ["Administrative", "Informational", "ProductRelated",
                   "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]:
        X[column] = 0
    # Second row has pages but no recorded duration.
    X.loc[X.index[1], "ProductRelated"] = 2
    out = SessionFeatureEngineer().fit_transform(X)
    ratios = ["AvgTimePerPage", "AvgProductTime", "AvgAdminTime", "AvgInfoTime",
              "ProductPageShare", "ProductTimeShare"]
    assert np.isfinite(out[ratios].to_numpy()).all()
    assert out.iloc[0][ratios].eq(0).all()
    assert out["NoTrackedPages"].tolist() == [1, 0]
    assert out["ZeroDurationSession"].tolist() == [0, 1]
    assert out["ProductPageShare"].tolist() == [0, 1]


def test_missing_components_do_not_silently_become_zero_totals(X_y):
    X, _ = X_y
    X = X.head(2).copy()
    X["Administrative"] = X["Administrative"].astype(float)
    X.loc[X.index[0], "Administrative"] = np.nan
    X.loc[X.index[1], "ProductRelated_Duration"] = np.nan
    out = SessionFeatureEngineer().fit_transform(X)
    assert np.isnan(out.iloc[0]["TotalPages"])
    assert np.isnan(out.iloc[0]["AvgTimePerPage"])
    assert np.isnan(out.iloc[1]["TotalDuration"])
    assert np.isnan(out.iloc[1]["ProductTimeShare"])


@pytest.mark.parametrize("include_pagevalues", [False, True])
def test_partial_missing_values_are_imputed_with_finite_output(include_pagevalues, X_y):
    X, y = X_y
    X = X.copy()
    for name in ["Administrative", "ProductRelated_Duration", "BounceRates", "PageValues"]:
        X[name] = X[name].astype(float)
        X.loc[X.index[:3], name] = np.nan
    for name in ["Month", "VisitorType", "Weekend"]:
        X[name] = X[name].astype(object)
        X.loc[X.index[:3], name] = np.nan
    pipeline = make_preprocessing_pipeline(include_pagevalues=include_pagevalues).fit(X, y)
    result = pipeline.transform(X.head(5))
    assert result.shape[0] == 5
    assert np.isfinite(result).all()


def test_all_nominal_columns_accept_unseen_levels_without_changing_width(X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    batch = X.head(1).copy()
    for name in ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]:
        batch[name] = "unseen_category"
    expected_width = pipeline.transform(X.head(1)).shape[1]
    result = pipeline.transform(batch)
    assert result.shape == (1, expected_width)
    assert np.isfinite(result).all()


def test_prediction_does_not_refit_scaling_on_new_sessions(X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    scaler = pipeline.named_steps["columns"].named_transformers_["log"].named_steps["scale"]
    fitted_mean = scaler.mean_.copy()
    original = pipeline.transform(X.head(1))
    changed = X.head(1).copy()
    changed["ProductRelated_Duration"] = 1e10
    pipeline.transform(changed)
    np.testing.assert_array_equal(scaler.mean_, fitted_mean)
    np.testing.assert_allclose(pipeline.transform(X.head(1)), original)
