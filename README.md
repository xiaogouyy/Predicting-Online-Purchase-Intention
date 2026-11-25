# MGSC674-Group-Project

Predicting whether an online shopping session ends in a purchase (UCI Online Shoppers Purchasing Intention dataset).

## Repository structure

```
├── data/                  # not committed; see data/README.md for how to get it
├── notebooks/             # numbered in the order to run them
│   ├── 01_eda.ipynb                    # EDA + train/test split (run first)
│   ├── 02_logistic_regression.ipynb
│   ├── 03_naive_bayes.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_knn.ipynb                    # compares itself with results/random_forest/ if it exists
│   ├── 06_xgboost.ipynb
│   ├── 07_lightgbm.ipynb
│   ├── 08_model_comparison.ipynb       # compares the six models, writes the shortlist
│   └── 09_fine_tuning.ipynb            # tunes the shortlist, final test evaluation
├── results/               # outputs, one folder per notebook (same name without the number)
│   ├── eda/
│   ├── logistic_regression/            # includes README.md with the LR notes
│   ├── naive_bayes/
│   ├── random_forest/
│   ├── knn/
│   ├── xgboost/
│   ├── lightgbm/
│   ├── model_comparison/
│   └── fine_tuning/
└── requirements.txt
```

Fitted model files (`*.pkl`, `*.joblib`) are not committed. Running a notebook writes them to its `results/` folder.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
```

Then get the data (see [data/README.md](data/README.md)) and run the notebooks in number order. `01_eda.ipynb` must come first, because it writes `data/processed/train_raw.csv` and `test_raw.csv`. `09_fine_tuning.ipynb` needs `results/model_comparison/shortlist.json` from `08_model_comparison.ipynb`.

The notebooks work when opened from `notebooks/` or from the repo root.
