"""Shared test fixtures. Tests use a small synthetic dataset, so CI never needs the UCI file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
# session_preprocessing.py lives in notebooks/ and train_mlflow.py in src/
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src"))

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def make_sessions(n=200, seed=0):
    """Fake sessions with the same 17 columns + Revenue as the UCI file."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Administrative": rng.integers(0, 10, n),
        "Administrative_Duration": rng.exponential(60, n),
        "Informational": rng.integers(0, 4, n),
        "Informational_Duration": rng.exponential(20, n),
        "ProductRelated": rng.integers(0, 80, n),
        "ProductRelated_Duration": rng.exponential(900, n),
        "BounceRates": rng.uniform(0, 0.2, n),
        "ExitRates": rng.uniform(0, 0.2, n),
        "PageValues": rng.exponential(5, n) * rng.integers(0, 2, n),
        "SpecialDay": rng.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0], n),
        "Month": rng.choice(MONTHS, n),
        "OperatingSystems": rng.integers(1, 9, n),
        "Browser": rng.integers(1, 14, n),
        "Region": rng.integers(1, 10, n),
        "TrafficType": rng.integers(1, 21, n),
        "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor", "Other"], n),
        "Weekend": rng.choice([True, False], n),
    })
    # make sessions with zero pages have zero duration, like the real data
    df.loc[df["Informational"] == 0, "Informational_Duration"] = 0.0
    df["Revenue"] = (df["PageValues"] + rng.normal(0, 2, n) > 4).astype(int)
    return df


@pytest.fixture
def sessions():
    return make_sessions()


@pytest.fixture
def X_y(sessions):
    return sessions.drop(columns="Revenue"), sessions["Revenue"].to_numpy()
