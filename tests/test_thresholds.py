"""Exercise the real notebook helpers without executing data or training cells.

Only the named function's AST is compiled. The implementation remains in the
notebook; a removed or renamed helper fails the test instead of using a copy.
Notebook execution as a whole is outside Python module coverage.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score,
)


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def notebook_function(filename, function_name, **bindings):
    notebook = json.loads((NOTEBOOKS / filename).read_text(encoding="utf-8"))
    matches = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                matches.append(node)
    assert len(matches) == 1, f"Expected one {function_name} in {filename}"
    namespace = {
        "np": np, "precision_recall_curve": precision_recall_curve,
        "confusion_matrix": confusion_matrix, "precision_score": precision_score,
        "recall_score": recall_score, "f1_score": f1_score, "accuracy_score": accuracy_score,
        **bindings,
    }
    module = ast.Module(body=matches, type_ignores=[])
    exec(compile(module, str(NOTEBOOKS / filename), "exec"), namespace)
    return namespace[function_name]


@pytest.fixture
def threshold_metrics():
    return notebook_function("09_fine_tuning.ipynb", "threshold_metrics")


def test_probability_equal_to_threshold_is_positive(threshold_metrics):
    p = np.array([np.nextafter(0.36, 0), 0.36, np.nextafter(0.36, 1)])
    result = threshold_metrics(np.array([0, 1, 1]), p, 0.36)
    assert [result[k] for k in ["TN", "FP", "FN", "TP"]] == [1, 0, 0, 2]
    assert result["Precision"] == result["Recall"] == result["F1"] == 1


def test_confusion_counts_and_metrics_match_hand_calculation(threshold_metrics):
    result = threshold_metrics(np.array([0, 0, 1, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8, 0.7]), 0.5)
    assert [result[k] for k in ["TN", "FP", "FN", "TP"]] == [1, 1, 1, 2]
    assert result["Precision"] == pytest.approx(2 / 3)
    assert result["Recall"] == pytest.approx(2 / 3)
    assert result["F1"] == pytest.approx(2 / 3)
    assert result["Accuracy"] == pytest.approx(3 / 5)


@pytest.mark.parametrize("threshold, expected", [(0.0, [0, 2, 0, 2]), (1.0, [2, 0, 1, 1])])
def test_threshold_endpoints(threshold_metrics, threshold, expected):
    result = threshold_metrics(np.array([0, 1, 0, 1]), np.array([0, 0.2, 0.9, 1]), threshold)
    assert [result[k] for k in ["TN", "FP", "FN", "TP"]] == expected


def test_no_positive_predictions_returns_zero_precision_and_f1(threshold_metrics):
    result = threshold_metrics(np.array([0, 1]), np.array([0.1, 0.2]), 0.5)
    assert result["Precision"] == result["Recall"] == result["F1"] == 0
    assert [result[k] for k in ["TN", "FP", "FN", "TP"]] == [1, 0, 1, 0]


def test_single_class_batch_still_has_all_confusion_counts(threshold_metrics):
    result = threshold_metrics(np.array([1, 1]), np.array([0.1, 0.9]), 0.5)
    assert [result[k] for k in ["TN", "FP", "FN", "TP"]] == [0, 0, 1, 1]


def test_max_f1_threshold_selects_expected_training_cutoff():
    labels = np.array([0, 1, 0, 1])
    choose = notebook_function("08_model_comparison.ipynb", "max_f1_threshold", y=labels)
    threshold, score = choose(np.array([0.1, 0.4, 0.3, 0.8]))
    assert threshold == pytest.approx(0.4)
    assert score == pytest.approx(1)


def test_max_f1_tie_uses_lowest_candidate_threshold():
    labels = np.array([1, 0, 0, 1])
    choose = notebook_function("08_model_comparison.ipynb", "max_f1_threshold", y=labels)
    threshold, score = choose(np.array([0.1, 0.2, 0.3, 0.4]))
    # Predicting all rows and predicting only the final buyer both have F1 = 2/3.
    assert threshold == pytest.approx(0.1)
    assert score == pytest.approx(2 / 3)


def test_constant_scores_have_a_defined_threshold():
    labels = np.array([0, 0, 1, 1])
    choose = notebook_function("08_model_comparison.ipynb", "max_f1_threshold", y=labels)
    threshold, score = choose(np.full(4, 0.5))
    assert threshold == pytest.approx(0.5)
    assert score == pytest.approx(2 / 3)
