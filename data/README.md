# Data

The data files are not stored in this repository (see `.gitignore`). Follow the steps below to get them. Only this README is tracked.

## Source

**Online Shoppers Purchasing Intention Dataset**, UCI Machine Learning Repository
<https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>

- 12,330 sessions, 17 features, binary target `Revenue` (1,908 purchases, about 15.5%)
- License: CC BY 4.0
- Citation: Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31, 6893–6908.

## Expected layout

```
data/
├── README.md
├── raw/
│   └── online_shoppers_intention.csv   # downloaded from UCI (step 1)
└── processed/                          # written by the EDA notebook (step 2)
    ├── train_raw.csv                   # 9,764 sessions, used by every model notebook
    ├── test_raw.csv                    # 2,441 sessions, held-out test set
    └── ...                             # engineered / prepared matrices, split manifest
```

## Step 1: download the raw file

Option A, by hand: open the UCI page above, click **Download**, unzip, and move `online_shoppers_intention.csv` to `data/raw/`.

Option B, from the repo root:

```bash
mkdir -p data/raw
curl -L -o /tmp/online_shoppers.zip \
  https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip
unzip -o /tmp/online_shoppers.zip online_shoppers_intention.csv -d data/raw/
```

Check that you have the same file the team used:

```bash
shasum -a 256 data/raw/online_shoppers_intention.csv
# b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9
```

## Step 2: build the train/test split

Run `notebooks/01_eda.ipynb` from start to finish. It removes the 125 exact duplicate sessions, makes a stratified 80/20 split (`random_state=42`), and writes `train_raw.csv` and `test_raw.csv` to `data/processed/`. The other notebooks read these two files, so run the EDA notebook first.

The EDA notebook imports `session_preprocessing.py` from the repo root (included in the repo).

Check the split:

```bash
shasum -a 256 data/processed/train_raw.csv data/processed/test_raw.csv
# 5f0f12cf13dc203981e358de484d546a567de9792676b858faaebce29d0ce933  data/processed/train_raw.csv
# 1e326e7e2479f3ae457f24dd91b100947ecc9c8a8f5a5272ce70702d24d1233c  data/processed/test_raw.csv
```

If the hashes differ, the split is different and your results will not match the ones in `results/`.
