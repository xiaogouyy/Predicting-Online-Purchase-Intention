"""Contract tests for the raw DataFrame consumed by training and future serving."""
import numpy as np
import pandas as pd
import pytest

from session_preprocessing import SessionFeatureEngineer, make_preprocessing_pipeline


RAW_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend",
]


@pytest.mark.parametrize("column", RAW_FEATURES)
def test_prediction_requires_every_feature_in_pagevalues_variant(column, X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    with pytest.raises((KeyError, ValueError)):
        pipeline.transform(X.head(2).drop(columns=column))


@pytest.mark.parametrize("include_pagevalues", [False, True])
def test_prediction_column_order_does_not_change_features(include_pagevalues, X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline(include_pagevalues=include_pagevalues).fit(X, y)
    expected = pipeline.transform(X.head(3))
    reordered = X.head(3).loc[:, list(reversed(X.columns))]
    np.testing.assert_allclose(pipeline.transform(reordered), expected)


def test_without_pagevalues_variant_does_not_require_pagevalues(X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline().fit(X.drop(columns="PageValues"), y)
    np.testing.assert_allclose(
        pipeline.transform(X.head(3)),
        pipeline.transform(X.head(3).drop(columns="PageValues")),
    )


@pytest.mark.parametrize("forbidden", ["Revenue", "source_row_id"])
@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_labels_and_identifiers_are_rejected_at_both_stages(forbidden, stage, X_y):
    X, _ = X_y
    engineer = SessionFeatureEngineer().fit(X)
    invalid = X.assign(**{forbidden: 0})
    with pytest.raises(ValueError, match="target and row identifiers"):
        getattr(engineer, stage)(invalid)


@pytest.mark.parametrize("invalid", [np.zeros((2, 17)), [[0] * 17], {"Month": ["Nov"]}])
def test_fit_requires_a_named_dataframe(invalid):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        SessionFeatureEngineer().fit(invalid)


@pytest.mark.parametrize("include_pagevalues", [False, True])
def test_feature_names_match_transformed_columns(include_pagevalues, X_y):
    X, _ = X_y
    engineer = SessionFeatureEngineer(include_pagevalues=include_pagevalues).fit(X)
    transformed = engineer.transform(X)
    assert engineer.get_feature_names_out().tolist() == transformed.columns.tolist()
    assert engineer.get_feature_names_out(X.columns).tolist() == transformed.columns.tolist()
    assert engineer.n_features_in_ == 17


def test_category_codes_are_encoded_as_labels(X_y):
    X, _ = X_y
    transformed = SessionFeatureEngineer().fit_transform(X)
    for name in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        assert transformed[name].tolist() == X[name].astype(str).tolist()
    np.testing.assert_array_equal(transformed["Weekend"], X["Weekend"].astype(float))


def test_unused_extra_column_is_not_a_model_feature(X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    np.testing.assert_allclose(
        pipeline.transform(X.head(3).assign(unused_metadata="ignored")),
        pipeline.transform(X.head(3)),
    )


@pytest.mark.parametrize("column", ["Administrative", "ProductRelated_Duration", "BounceRates", "Weekend"])
def test_nonnumeric_measurements_are_rejected(column, X_y):
    X, y = X_y
    pipeline = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    invalid = X.head(1).copy()
    invalid[column] = "not_a_number"
    with pytest.raises((TypeError, ValueError)):
        pipeline.transform(invalid)


def test_preprocessing_preserves_source_dataframe_and_row_order(X_y):
    X, y = X_y
    original = X.copy(deep=True)
    pipeline = make_preprocessing_pipeline(include_pagevalues=True).fit(X, y)
    batch = X.iloc[[4, 1, 4]].copy()
    batch.index = ["duplicate", "other", "duplicate"]
    transformed = pipeline.transform(batch)
    expected = np.vstack([pipeline.transform(X.iloc[[i]]) for i in [4, 1, 4]])
    np.testing.assert_allclose(transformed, expected)
    pd.testing.assert_frame_equal(X, original)
