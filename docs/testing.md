# Unit tests and coverage

Eric Sun owns the expanded regression tests and coverage reporting on
`feature/unit-tests`. The tests protect preprocessing, model input contracts and
threshold decisions. GitHub Actions runs them before a pull request can be merged.

## Run the checks

From the repository root, with Python 3.12 and a virtual environment activated:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v --cov --cov-report=term-missing \
  --cov-report=xml:coverage.xml --cov-report=html:htmlcov \
  --junitxml=junit.xml
```

Open `htmlcov/index.html` for a local report. On GitHub, open the CI run to see
the coverage summary and download the `test-and-coverage-reports` artifact
(HTML, XML and JUnit results). Reports are uploaded even when tests fail, if
they were generated. Generated reports are ignored by Git.

## What is tested

| Area | Contract checked |
| --- | --- |
| Thresholds | Equality uses `>=`; endpoint cutoffs; hand-calculated confusion counts and metrics; no positive predictions; single-class batches; max-F1 selection, ties and constant scores |
| Raw input schema | All 17 features in the PageValues variant; PageValues optional in the alternative variant; named DataFrames; target and row ID exclusion; invalid numeric values; column order invariance |
| Preprocessing edges | Zero-page and zero-duration sessions; missing components; numeric and categorical imputation; unseen categories; feature names; preserved input data and row order; fitted scaling remains unchanged during prediction |
| Classifiers | Fit and probability output for five model families; independent stacking refit; focused KNN and RF checks for single rows, missing fields, reordered columns, empty input and use before fitting |
| Training workflow | Synthetic training file only; labels and IDs removed from model inputs; five-fold CV; logged metrics; final stacking URI registered as `purchase-intent-stacking@champion` |

The fixtures generate deterministic synthetic sessions. No UCI CSV, fitted model,
tracking server, local MLflow database or registry is needed. The training workflow
test mocks file loading and MLflow effects; it is a unit test, not a live registry
integration test. Existing model tests fit small real classifiers.

Threshold helpers currently live in `08_model_comparison.ipynb` and
`09_fine_tuning.ipynb`. Their actual function definitions are loaded with Python's
AST so the tests do not run notebook data-loading or training cells. The loader
fails if a helper disappears or is renamed. It does not duplicate the helper's
implementation. These tests exercise `max_f1_threshold` and `threshold_metrics`;
the separate inline final-threshold selection cell in notebook 09 is not executed.

Coverage measures lines and branches in the Python modules under `src/` and
`notebooks/`; `.ipynb` files are explicitly excluded. This percentage does not
represent coverage of entire notebooks, the future API or Streamlit application.
Coverage is reported without introducing a new percentage gate; pytest failures
continue to fail the existing `tests` job.

## Interfaces with the team

Yanjie's training refactor can reuse the synthetic fixtures and model contracts.
Vincent owns full raw-data validation, including allowed ranges and checksums;
these tests cover the current model input behavior. Yorkin can reuse the
probability and threshold expectations when implementing `/predict`. These tests
do not claim that the API or Streamlit UI already exists.

## Check in explanation

> I am responsible for unit tests and coverage. I expanded the synthetic-data
> tests to cover threshold boundaries, missing input columns, unseen categories,
> missing values, and zero-activity sessions. I also strengthened the KNN and
> random forest prediction checks. GitHub Actions now runs the tests and publishes
> coverage reports. My contribution sits in the CI part of our architecture,
> checking the preprocessing and model pipeline before changes reach main.

For a short demonstration, open this branch's draft pull request, show the green
`tests` check, then open its Actions run and the coverage summary. An example
to explain is `test_probability_equal_to_threshold_is_positive`: a probability
of exactly 0.36 is positive, while the nearest lower floating-point value is not.
