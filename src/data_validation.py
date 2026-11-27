"""Download, validate and split the project's original UCI dataset.

Run from the repository root: python src/data_validation.py
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
CSV_NAME = "online_shoppers_intention.csv"
RAW_PATH = ROOT / "data" / "raw" / CSV_NAME
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
EXPECTED_SHA256 = (
    "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
)


def download_data():
    """Return the CSV path after checking the team's reference SHA-256."""
    if RAW_PATH.exists():
        print("Using existing dataset.")
        csv_bytes = RAW_PATH.read_bytes()
    else:
        print("Downloading dataset from UCI...")
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            zip_bytes = response.read()
        with ZipFile(BytesIO(zip_bytes)) as archive:
            csv_bytes = archive.read(CSV_NAME)

    actual_hash = sha256(csv_bytes).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch. Expected {EXPECTED_SHA256}, got {actual_hash}."
        )

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        RAW_PATH.write_bytes(csv_bytes)

    print("SHA-256 check passed.")
    return RAW_PATH


def validate_data(df):
    """Reject invalid raw data and report duplicates before splitting."""
    counts = ["Administrative", "Informational", "ProductRelated"]
    durations = [f"{name}_Duration" for name in counts]
    codes = ["OperatingSystems", "Browser", "Region", "TrafficType"]
    rates = ["BounceRates", "ExitRates", "SpecialDay"]
    numeric = counts + durations + codes + rates + ["PageValues"]
    expected = set(numeric + ["Month", "VisitorType", "Weekend", "Revenue"])

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    if set(df.columns) != expected:
        missing = sorted(expected - set(df.columns))
        extra = sorted(set(df.columns) - expected)
        raise ValueError(f"Schema mismatch. Missing: {missing}; extra: {extra}")
    if df.empty:
        raise ValueError("The dataset is empty.")

    missing_cells = int(df.isna().sum().sum())
    if missing_cells:
        raise ValueError(f"Found {missing_cells} missing cells.")

    for col in numeric:
        if (
            not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            raise ValueError(f"{col} must contain numeric values.")

    for col in ["Weekend", "Revenue"]:
        if not pd.api.types.is_bool_dtype(df[col]):
            raise ValueError(f"{col} must contain boolean values.")

    values = df[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Numeric values must be finite and non-negative.")
    if (df[counts + codes] % 1 != 0).any().any():
        raise ValueError("Page counts and category codes must be whole numbers.")
    if (df[codes] < 1).any().any():
        raise ValueError("Category codes must be positive integers.")
    if (df[rates] > 1).any().any():
        raise ValueError("BounceRates, ExitRates and SpecialDay must be in [0, 1].")

    months = {
        "Jan", "Feb", "Mar", "Apr", "May", "June",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    }
    visitors = {"Returning_Visitor", "New_Visitor", "Other"}
    if not df["Month"].isin(months).all():
        raise ValueError("Month contains an unknown category.")
    if not df["VisitorType"].isin(visitors).all():
        raise ValueError("VisitorType contains an unknown category.")
    if df["Revenue"].nunique() != 2:
        raise ValueError("Revenue must contain both purchase classes.")

    print("Schema, types, and value checks passed.")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Missing cells: {missing_cells}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    return df


def create_splits(df, output_dir=None):
    """Reproduce the EDA split and retain source-row IDs for auditing.

    Input is the validated raw frame in its original CSV row order.
    Source-row IDs are audit keys, never model input columns.
    """
    raw = df.reset_index(drop=True)
    duplicate_mask = raw.duplicated(keep="first")
    removed = raw.loc[duplicate_mask].copy()
    removed.insert(0, "source_row_id", removed.index)

    clean = raw.loc[~duplicate_mask].copy()
    clean.index.name = "source_row_id"
    clean["Revenue"] = clean["Revenue"].astype(int)
    X = clean.drop(columns="Revenue")
    y = clean["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if not set(X_train.index).isdisjoint(X_test.index):
        raise ValueError("Training and test source-row IDs overlap.")
    if len(X_train) + len(X_test) != len(clean):
        raise ValueError("The split lost or duplicated rows.")
    if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
        raise ValueError("Features and labels are not aligned.")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Each split must contain both purchase classes.")

    # Match the EDA integrity check for both PageValues feature settings.
    train_keys = pd.util.hash_pandas_object(
        X_train.drop(columns="PageValues"), index=False
    )
    test_keys = pd.util.hash_pandas_object(
        X_test.drop(columns="PageValues"), index=False
    )
    if not set(train_keys).isdisjoint(test_keys):
        raise ValueError("The same session feature values occur in both splits.")

    train = X_train.join(y_train).sort_index()
    test = X_test.join(y_test).sort_index()
    manifest = pd.concat([
        pd.DataFrame({"source_row_id": X_train.index, "partition": "train"}),
        pd.DataFrame({"source_row_id": X_test.index, "partition": "test"}),
    ]).sort_values("source_row_id")

    output_dir = Path(output_dir) if output_dir is not None else ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train_raw.csv")
    test.to_csv(output_dir / "test_raw.csv")
    removed.to_csv(output_dir / "removed_exact_duplicates.csv", index=False)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    print(f"Duplicates removed: {int(duplicate_mask.sum()):,}")
    print(f"Rows after deduplication: {len(clean):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print("Split integrity checks passed.")
    print(f"Saved split files to: {output_dir}")
    return train, test


if __name__ == "__main__":
    path = download_data()
    df = pd.read_csv(path)
    validate_data(df)
    create_splits(df)
