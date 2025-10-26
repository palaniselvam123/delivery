import os
import sys
from typing import List, Optional

# Ensure module path is stable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import custom transformer module so pickle can resolve it later
import fe_utils  # noqa: F401
from fe_utils import FrequencyEncoder  # noqa: F401

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import joblib

import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "delay_days_best_model.joblib")


app = FastAPI(
    title="Delay Days Predictor",
    description="Predict shipment leg delay days; Swagger at /docs.",
    version="1.0.0",
)

# Redirect root to Swagger UI
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

# Load the model at startup in the serving process
app.state.pipeline = None

@app.on_event("startup")
def load_model_once():
    app.state.pipeline = joblib.load(MODEL_PATH)

# ---------- Schemas ----------
class Record(BaseModel):
    Route_No: Optional[str] = Field(None, alias="Route No")
    Shipment_Number: Optional[str] = Field(None, alias="Shipment Number")

    Vessel: Optional[str] = None
    Voyage: Optional[str] = None
    Carrier: Optional[str] = None
    Load_Port: Optional[str] = Field(None, alias="Load Port")
    Discharge_Port: Optional[str] = Field(None, alias="Discharge Port")
    Transport_Mode: Optional[str] = Field(None, alias="Transport Mode")
    Early_Late: Optional[str] = Field(None, alias="Early/Late")
    WasRolled: Optional[str] = None
    RolloverReason: Optional[str] = None
    PaymentStatus: Optional[str] = None
    DocumentSubmitted: Optional[str] = None
    WeatherSeverity: Optional[str] = None
    GeoRiskFlag: Optional[str] = None
    CarrierNotification: Optional[str] = None
    ExternalNewsImpact: Optional[str] = None

    LegNumber: Optional[int] = None
    PortCongestion: Optional[int] = None
    Estimated_Transit_Days: Optional[float] = Field(None, alias="Estimated Transit Days")

    ETD: Optional[str] = None
    ETA: Optional[str] = None
    ATD: Optional[str] = None
    ATA: Optional[str] = None

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Route No": "RTE123",
                "Shipment Number": "SHP100001",
                "Vessel": "Vessel-A",
                "Voyage": "VY-001",
                "Carrier": "MAEU",
                "Load Port": "Chennai",
                "Discharge Port": "Los Angeles",
                "Transport Mode": "Sea",
                "Early/Late": "OnTime",
                "WasRolled": "False",
                "RolloverReason": "None",
                "PaymentStatus": "Paid",
                "DocumentSubmitted": "True",
                "WeatherSeverity": "Moderate",
                "GeoRiskFlag": "None",
                "CarrierNotification": "None",
                "ExternalNewsImpact": "False",
                "LegNumber": 1,
                "PortCongestion": 6,
                "Estimated Transit Days": 21,
                "ETD": "2025-02-01T10:00:00Z",
                "ETA": "2025-02-23T12:00:00Z",
                "ATD": "2025-02-01T12:00:00Z",
                "ATA": "2025-02-23T18:00:00Z"
            }
        }

class BatchRequest(BaseModel):
    items: List[Record]

class Prediction(BaseModel):
    DelayDays_Pred: float

class BatchResponse(BaseModel):
    predictions: List[Prediction]

# ---------- Feature engineering (match training) ----------
DATE_COLS = ["ETD","ETA","ATD","ATA"]
NUM_COLS = [
    "Estimated Transit Days","LegNumber","PortCongestion",
    "sched_transit_days_calc","actual_transit_days_calc","late_vs_sched_days",
    "ETD_year","ETD_month","ETD_dow","ETD_day","ETD_is_wknd",
    "ETA_year","ETA_month","ETA_dow","ETA_day","ETA_is_wknd",
    "ATD_year","ATD_month","ATD_dow","ATD_day","ATD_is_wknd",
    "ATA_year","ATA_month","ATA_dow","ATA_day","ATA_is_wknd"
]
CAT_COLS = [
    "Vessel","Voyage","Carrier","Load Port","Discharge Port","Transport Mode",
    "Early/Late","WasRolled","RolloverReason","PaymentStatus","DocumentSubmitted",
    "WeatherSeverity","GeoRiskFlag","CarrierNotification","ExternalNewsImpact"
]
DROP_COLS = ["Route No","Shipment Number","ETD","ETA","ATD","ATA","Delay Days","Transit Days"]

def to_df(records: List[Record]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump(by_alias=True) for r in records])
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    def date_feats(s: pd.Series, name: str) -> pd.DataFrame:
        return pd.DataFrame({
            f"{name}_year": s.dt.year,
            f"{name}_month": s.dt.month,
            f"{name}_dow": s.dt.dayofweek,
            f"{name}_day": s.dt.day,
            f"{name}_is_wknd": s.dt.dayofweek.isin([5,6]).astype("float32"),
        })

    for name in ["ETD","ETA","ATD","ATA"]:
        if name in df.columns:
            feats = date_feats(df[name], name)
            df = pd.concat([df, feats], axis=1)
        else:
            for col in [f"{name}_year", f"{name}_month", f"{name}_dow", f"{name}_day", f"{name}_is_wknd"]:
                df[col] = np.nan

    df["sched_transit_days_calc"]  = (df["ETA"] - df["ETD"]).dt.total_seconds() / 86400.0
    df["actual_transit_days_calc"] = (df["ATA"] - df["ATD"]).dt.total_seconds() / 86400.0
    df["late_vs_sched_days"]       = df["actual_transit_days_calc"] - df["sched_transit_days_calc"]

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    for col in NUM_COLS + CAT_COLS:
        if col not in X.columns:
            X[col] = np.nan

    ordered = [c for c in NUM_COLS if c in X.columns] + [c for c in CAT_COLS if c in X.columns]
    X = X[ordered + [c for c in X.columns if c not in ordered]]
    return X

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.pipeline is not None}

@app.post("/predict", response_model=Prediction, summary="Predict Delay Days for one record")
def predict_one(payload: Record):
    X = to_df([payload])
    y_pred = app.state.pipeline.predict(X)[0]
    return {"DelayDays_Pred": float(y_pred)}

@app.post("/predict-batch", response_model=BatchResponse, summary="Predict Delay Days for a batch")
def predict_batch(payload: BatchRequest):
    X = to_df(payload.items)
    y_pred = app.state.pipeline.predict(X)
    return {"predictions": [{"DelayDays_Pred": float(v)} for v in y_pred]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
