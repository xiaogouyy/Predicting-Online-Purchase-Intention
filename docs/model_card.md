# Model card: purchase-intent-stacking

A short summary of what the model is, what it is for, how well it works and where it should not be trusted. Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*. Full details are in the [README](../README.md).

## Model details

| | |
|---|---|
| **Name in the MLflow registry** | `purchase-intent-stacking`, alias `champion` |
| **Type** | Stacking ensemble for binary classification |
| **Members** | Tuned random forest, XGBoost, KNN and LightGBM; a logistic regression combines their probabilities |
| **Input** | The 17 session features of the UCI dataset (page counts and durations, bounce and exit rates, `PageValues`, special day, month, OS, browser, region, traffic type, visitor type, weekend) |
| **Output** | Probability that the session ends in a purchase (`Revenue = 1`); decision threshold 0.36 |
| **Preprocessing** | Shared pipeline in `notebooks/session_preprocessing.py`, fitted inside each CV fold |
| **Code** | `notebooks/09_fine_tuning.ipynb` (selection and test evaluation), `src/train_mlflow.py` (MLflow training and registration) |
| **Owners** | INSY 674 Group 3, McGill University, Fall 2026 |

Load it with:

```python
import mlflow
model = mlflow.sklearn.load_model("models:/purchase-intent-stacking@champion")
proba = model.predict_proba(sessions)[:, 1]
```

## Intended use

- **Primary use:** rank completed shopping sessions by purchase probability, so that marketing and CRM teams can study high-intent sessions and plan follow-up (for example post-session retargeting).
- **Secondary use:** show analysts which browsing behaviours separate buyers from browsers.
- **Users:** marketing / CRM teams and e-commerce analysts.

## Out-of-scope uses

- **Real-time decisions during a visit.** The features describe the whole session, so the model can only score a session after it ends.
- **Decisions about individual people** (pricing, credit, eligibility). The data have no user identity and were not collected for this.
- **Other websites or other years** without re-validation. All data come from one retailer over one year.
- **Measuring the effect of a campaign.** The model predicts who is likely to buy, not who would buy *because of* an offer.

## Training data

- [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI, CC BY 4.0): 12,330 sessions from one year; January and April are missing.
- 125 exact duplicate rows removed, leaving 12,205 sessions.
- Stratified 80/20 split (`random_state=42`): **9,764 training** sessions and **2,441 test** sessions, both about 15.6% purchases.

## Metrics

- **Average precision (AP)** is the main metric, because only 15.6% of sessions buy and the use case is ranking. Random ranking scores AP 0.156.
- **ROC-AUC** as a second ranking metric; **precision, recall and F1** at the chosen threshold.

## Performance

**Cross-validation (training set only):** AP **0.757 ± 0.013** over 15 folds (3 × 5-fold) in `09_fine_tuning.ipynb`. The MLflow run, which uses the shared preprocessing for every member, scores AP 0.754 ± 0.026 over 5 folds.

**Held-out test set (2,441 sessions)**, evaluated once in `09_fine_tuning.ipynb`:

| Model | AP | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (purchase rate) | 0.156 | 0.500 | 0.000 | 0.000 | 0.000 |
| Rule: rank by `PageValues` | 0.622 | 0.866 | 0.571 | 0.812 | 0.670 |
| **Stacking ensemble** (threshold 0.36) | **0.763** (95% CI 0.722–0.804) | **0.938** | **0.686** | **0.704** | **0.695** |

At the 0.36 threshold the model flags 392 of 2,441 sessions (16%) and finds 269 of 382 buyers (70%), with 123 false alarms.

**Most important inputs** (permutation importance): `PageValues` (0.556), then `Month` (0.046) and `ExitRates` (0.030).

## Limitations and risks

| Limitation | What it means |
|---|---|
| **`PageValues` may leak the outcome.** It is not a copy of the label (19.5% of buyers have a value of 0), but page values were probably computed over the same period. | Without it, CV AP falls from 0.758 to **0.383**. Treat 0.383 as the lower bound of what browsing alone can predict. |
| **The test set is not fully fresh.** Some individual-model notebooks evaluated it earlier. | The test score may be slightly optimistic; use the CV estimate as the main figure. |
| **Offline, whole-session model.** | It cannot trigger an action during a visit. |
| **Sessions assumed independent.** The split is random, and repeat visitors may appear in both train and test. | Future performance may be lower; validate by time and by user. |
| **Threshold maximises F1, not profit.** | Set it from the value of a sale and the cost of an offer before use. |
| **Prediction is not causation.** | High-intent visitors may buy anyway; run an A/B test or uplift model before spending on offers. |

## Ethical considerations

- The data hold no names or user IDs, only session behaviour and coarse codes (region, browser, OS).
- `Region` adds almost nothing to the model (importance about 0), but it is still an input; check results by region before use so that no region is targeted unfairly.
- The model should support human marketing decisions, not make automatic decisions about individuals.

## Recommendations

1. Confirm with the data owner how `PageValues` is calculated, or rebuild it from an earlier period.
2. Validate on new sessions from a later time period before any deployment.
3. Choose the threshold from business costs.
4. Monitor the share of flagged sessions and the AP on new data, and retrain when they drift.
