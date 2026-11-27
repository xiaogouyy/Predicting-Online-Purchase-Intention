"""Prediction API for the purchase-intent model (feature/prediction-api).

Serves the registered MLflow model `models:/purchase-intent-stacking@champion`
behind a FastAPI `/predict` endpoint, so the Streamlit app (feature/streamlit-app)
can send sessions and get purchase probabilities back.

Run locally (from the repo root, after `python src/train_mlflow.py`):
    pip install fastapi uvicorn
    uvicorn src.predict_api:app --reload
    # docs at http://127.0.0.1:8000/docs

Interfaces with other branches:
    - Reads the model from the MLflow registry built by feature/mlflow-tracking
      (tracking URI sqlite:///mlflow.db, name purchase-intent-stacking, alias champion).
    - Input columns = the 17 raw UCI columns (same schema checked by feature/data-validation).
    - Output is consumed by feature/streamlit-app.
"""

import os
from pathlib import Path
from typing import List

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MODEL_URI = os.getenv("MODEL_URI", "models:/purchase-intent-stacking@champion")
DEFAULT_THRESHOLD = 0.5  # TODO: replace with the business threshold chosen in the notebooks


class Session(BaseModel):
    """One browsing session: the 17 raw input columns of the UCI dataset."""
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


class PredictRequest(BaseModel):
    sessions: List[Session]
    threshold: float = DEFAULT_THRESHOLD


class Prediction(BaseModel):
    probability: float
    will_purchase: bool


class PredictResponse(BaseModel):
    model_uri: str
    predictions: List[Prediction]


def load_model():
    """Load the champion model from the MLflow registry.

    TODO:
    - mlflow.set_tracking_uri(TRACKING_URI)
    - return mlflow.sklearn.load_model(MODEL_URI)  (keeps predict_proba)
    - fail with a clear error if the registry / alias does not exist
    """
    raise NotImplementedError


def predict_sessions(model, request: PredictRequest) -> PredictResponse:
    """Turn sessions into a DataFrame, score them, apply the threshold.

    TODO:
    - pd.DataFrame([s.model_dump() for s in request.sessions])
    - p = model.predict_proba(df)[:, 1]
    - build Prediction(probability=p_i, will_purchase=p_i >= request.threshold)
    """
    raise NotImplementedError


# TODO (next commits):
# - app = FastAPI(title="Purchase intent API"); load model once at startup
# - GET  /health   -> {"status": "ok", "model_uri": MODEL_URI}
# - POST /predict  -> PredictResponse
# - tests/test_api.py with FastAPI TestClient and a small dummy model (no real mlflow.db in CI)
# - add fastapi + uvicorn to requirements.txt
