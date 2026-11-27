"""Evaluate the saved final model on the test set (notebooks/09_fine_tuning.ipynb, section 8).

How to run (from the repo root, after src/train.py):
    python src/evaluate.py
    python src/evaluate.py --config configs/training.yaml

What it does:
    1. Scores the test sessions with the saved model and its threshold (both fixed in src/train.py).
    2. Compares it with two baselines that need no fitting: the purchase rate, and "rank by PageValues".
    3. 95 % bootstrap intervals for the final model's metrics.
Results are written to test_results.csv and test_ci.csv in the results folder from the config.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from model_pipeline import CONFIG_FILE, FEATURES, ROOT, load_config


def threshold_metrics(y, p, t):
    """Metrics after turning probabilities into yes/no with threshold t."""
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred, zero_division=0), "Accuracy": accuracy_score(y, pred),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def bootstrap_ci(y, p, t, n_boot, seed):
    """95 % intervals from resampling the test sessions with replacement."""
    rng = np.random.default_rng(seed)
    boot = defaultdict(list)
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        m = threshold_metrics(y[i], p[i], t)
        boot["AP"].append(average_precision_score(y[i], p[i]))
        boot["ROC-AUC"].append(roc_auc_score(y[i], p[i]))
        for k in ["Precision", "Recall", "F1"]:
            boot[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in boot.items()}


def main(config_path=CONFIG_FILE):
    config = load_config(config_path)
    target = config["target"]

    # Step 1: load the saved model, and the train/test files
    saved = joblib.load(ROOT / config["output"]["model_file"])
    model, threshold = saved["model"], saved["threshold"]
    train = pd.read_csv(ROOT / config["data"]["train_file"])
    test = pd.read_csv(ROOT / config["data"]["test_file"])
    assert not set(train["source_row_id"]) & set(test["source_row_id"]), "train and test share sessions"
    X_test, y_test = test[FEATURES], test[target].astype(int).to_numpy()
    purchase_rate = train[target].astype(int).mean()  # dummy baseline uses the TRAINING rate

    # Step 2: final model vs. the two baselines
    p_final = model.predict_proba(X_test)[:, 1]
    scores = {"Dummy (purchase rate)": np.full(len(y_test), purchase_rate),
              "Rule: rank by PageValues": X_test["PageValues"].to_numpy(),
              "Final model (stacking)": p_final}
    cutoffs = {"Dummy (purchase rate)": 0.5, "Rule: rank by PageValues": 1e-9, "Final model (stacking)": threshold}
    table = pd.DataFrame({name: {"AP": average_precision_score(y_test, p), "ROC-AUC": roc_auc_score(y_test, p),
                                 "threshold": cutoffs[name], **threshold_metrics(y_test, p, cutoffs[name])}
                          for name, p in scores.items()}).T

    # Step 3: bootstrap intervals for the final model
    intervals = bootstrap_ci(y_test, p_final, threshold, config["evaluation"]["n_bootstrap"], config["seed"])
    ci = pd.DataFrame({k: {"test": table.loc["Final model (stacking)", k], "ci95_low": low, "ci95_high": high}
                       for k, (low, high) in intervals.items()}).T
    ci.loc["AP", "CV estimate"] = saved["cv_ap"]

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200, "display.max_columns", None):
        print(table)
        print()
        print(ci)
    at_half = threshold_metrics(y_test, p_final, 0.5)
    print("At threshold 0.5 instead:", {k: round(float(v), 4) for k, v in at_half.items()})

    results_dir = ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "test_results.csv")
    ci.to_csv(results_dir / "test_ci.csv")
    print(f"Saved test results to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the final model on the test set.")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="path to the YAML config")
    args = parser.parse_args()
    main(args.config)
