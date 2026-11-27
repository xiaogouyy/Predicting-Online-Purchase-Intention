"""Train the final model of notebooks/09_fine_tuning.ipynb, with the settings in configs/training.yaml.

How to run (from the repo root, after src/data_validation.py has written data/processed/train_raw.csv):
    python src/train.py
    python src/train.py --config configs/training.yaml

What it does (09 sections 6-7; the random searches stay in the notebook, their results are in the config):
    1. Out-of-fold predictions of each tuned model on 15 folds (3 x 5-fold, seed 2026),
       then the stacking logistic regression on top of them -> CV estimate of the final model's AP.
    2. Decision threshold = the one with the highest F1 on the pooled out-of-fold predictions.
    3. Refit the stacking model on all training sessions and save it together with the threshold.
The test set is NOT read here. It is used once, in src/evaluate.py.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.parallel import Parallel, delayed
from threadpoolctl import threadpool_limits

from model_pipeline import CONFIG_FILE, FEATURES, ROOT, build_model, build_stacking, load_config


def load_train(config):
    """Training sessions only: X = the 17 inputs, y = 1 if the session bought."""
    train = pd.read_csv(ROOT / config["data"]["train_file"])
    return train[FEATURES], train[config["target"]].astype(int).to_numpy()


def fit_predict_fold(model, X, y, train_idx, val_idx):
    # one core per job, so running folds in parallel gives the same numbers as running them one by one
    with threadpool_limits(limits=1):
        m = clone(model).fit(X.iloc[train_idx], y[train_idx])
        return m.predict_proba(X.iloc[val_idx])[:, 1]


def member_oof(model, X, y, folds, n_splits, n_jobs):
    """Out-of-fold probabilities, one row per repeat of the K-fold split."""
    preds = Parallel(n_jobs=n_jobs)(delayed(fit_predict_fold)(model, X, y, tr, va) for tr, va in folds)
    oof = np.full((len(folds) // n_splits, len(y)), np.nan)
    for j, ((_, va), p) in enumerate(zip(folds, preds)):
        oof[j // n_splits, va] = p
    return oof


def stack_oof(member_oofs, y, folds, n_splits, max_iter):
    """Stacking on the members' out-of-fold probabilities: the logistic regression is also fitted
    on 4 folds and scored on the 5th, so the stacker is cross-validated too."""
    first = next(iter(member_oofs.values()))
    out = np.full_like(first, np.nan)
    for r in range(out.shape[0]):
        Z = np.column_stack([oof[r] for oof in member_oofs.values()])
        for j in range(n_splits):
            tr, va = folds[r * n_splits + j]
            out[r, va] = LogisticRegression(max_iter=max_iter).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
    return out


def fold_aps(oof, y, folds, n_splits):
    return np.array([average_precision_score(y[va], oof[j // n_splits, va]) for j, (_, va) in enumerate(folds)])


def best_f1_threshold(y, scores):
    """Threshold with the highest F1 (balance between missed buyers and false alarms)."""
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])


def main(config_path=CONFIG_FILE):
    config = load_config(config_path)
    # KNN computes distances in chunks: 256 MB per worker instead of 1 GB, so parallel folds fit in memory
    sklearn.set_config(working_memory=256)

    # Step 1: training data and the 15 folds
    X, y = load_train(config)
    cv = config["cv"]
    folds = list(RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"],
                                         random_state=cv["random_state"]).split(X, y))
    print(f"Training sessions: {len(X)} | purchase rate {y.mean():.4f} | {len(folds)} folds")

    # Step 2: out-of-fold predictions of each tuned model
    members = config["stacking"]["members"]
    oofs = {}
    for name in members:
        oofs[name] = member_oof(build_model(name, config), X, y, folds, cv["n_splits"], config["n_jobs"])
        print(f"  {name:<14} CV AP = {fold_aps(oofs[name], y, folds, cv['n_splits']).mean():.4f}")

    # Step 3: CV estimate of the stacking model and the decision threshold
    final_oof = stack_oof(oofs, y, folds, cv["n_splits"], config["stacking"]["final_max_iter"])
    aps = fold_aps(final_oof, y, folds, cv["n_splits"])
    pooled_y = np.concatenate([y] * final_oof.shape[0])
    threshold, prec, rec, f1 = best_f1_threshold(pooled_y, final_oof.ravel())
    print(f"Stacking CV AP = {aps.mean():.4f} +- {aps.std(ddof=1):.4f}")
    print(f"Threshold (max F1 on OOF) = {threshold:.4f} (precision {prec:.3f}, recall {rec:.3f}, F1 {f1:.3f})")

    # Step 4: refit on all training sessions and save the model with its threshold
    print("Fitting the final model on all training sessions ...")
    final_model = build_stacking(config).fit(X, y)
    weights = dict(zip(members, final_model.final_estimator_.coef_.ravel().round(3).tolist()))
    print("Stacking weights:", weights)

    model_file = ROOT / config["output"]["model_file"]
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "threshold": threshold, "cv_ap": float(aps.mean())}, model_file)

    summary = {
        "cv_ap_mean": float(aps.mean()),
        "cv_ap_sd": float(aps.std(ddof=1)),
        "member_cv_ap": {name: float(fold_aps(oofs[name], y, folds, cv["n_splits"]).mean()) for name in members},
        "threshold": threshold,
        "oof_precision": prec,
        "oof_recall": rec,
        "oof_f1": f1,
        "stacking_weights": weights,
    }
    results_dir = ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model to {model_file} and CV summary to {results_dir / 'cv_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final stacking model.")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="path to the YAML config")
    args = parser.parse_args()
    main(args.config)
