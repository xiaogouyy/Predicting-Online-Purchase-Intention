## Logistic Regression (owner: Yorkin Ma)

Notebook: `notebooks/02_logistic_regression.ipynb`. It reads `data/processed/train_raw.csv` / `test_raw.csv` and writes everything to `results/logistic_regression/` (same layout as `results/random_forest/`). A full run takes about 9 minutes.

Same split, same folds (`StratifiedKFold(5, shuffle=True, random_state=42)`) and same primary metric (average precision) as the RF notebook.

| Model | CV AP (mean ± SD) | Test AP | Test ROC-AUC |
|---|---|---|---|
| Random ranking | 0.156 | 0.156 | 0.50 |
| LR, **without** PageValues (EDA policy) | 0.347 ± 0.026 | 0.347 | 0.760 |
| LR, **with** PageValues | 0.684 ± 0.011 | 0.699 | 0.925 |

- Preprocessing: the 12 engineered features from the EDA, log1p + scaling, one-hot encoding, all fitted inside the CV folds. Tuned by grid search over L1 / L2 / elastic-net, C and class_weight.
- Threshold: picked on out-of-fold training predictions (max F1). `lr_test_comparison_table.csv` also reports the fixed 0.5 threshold used in the RF notebook.
- `lr_oof_predictions.csv` holds the out-of-fold probabilities by `source_row_id`, for the ensemble step.
