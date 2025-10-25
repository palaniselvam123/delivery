import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Ensure import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import fe_utils
from fe_utils import FrequencyEncoder  # noqa: F401

# --- Path to your pipeline and CSV (update if needed) ---
MODEL_PATH = r"C:\Users\Dell\Documents\ML Projects\Delivery Time predictor\data\delay_days_best_model.joblib"
DATA_PATH = r"C:\Users\Dell\Documents\ML Projects\Delivery Time predictor\data\routes_20000 (3).csv"

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# --- Feature Engineering Section (matches your API/train!) ---
DATE_COLS = ["ETD","ETA","ATD","ATA"]
NUM_COLS = [
    'Estimated Transit Days','LegNumber','PortCongestion',
    'sched_transit_days_calc','actual_transit_days_calc','late_vs_sched_days',
    'ETD_year','ETD_month','ETD_dow','ETD_day','ETD_is_wknd',
    'ETA_year','ETA_month','ETA_dow','ETA_day','ETA_is_wknd',
    'ATD_year','ATD_month','ATD_dow','ATD_day','ATD_is_wknd',
    'ATA_year','ATA_month','ATA_dow','ATA_day','ATA_is_wknd'
]
CAT_COLS = [
    'Vessel','Voyage','Carrier','Load Port','Discharge Port','Transport Mode',
    'Early/Late','WasRolled','RolloverReason','PaymentStatus','DocumentSubmitted',
    'WeatherSeverity','GeoRiskFlag','CarrierNotification','ExternalNewsImpact'
]
DROP_COLS = ["Route No","Shipment Number","ETD","ETA","ATD","ATA","Delay Days","Transit Days"]

df2 = df.copy()
for c in DATE_COLS:
    if c in df2.columns:
        df2[c] = pd.to_datetime(df2[c], errors="coerce")

def date_feats(s, name):
    return pd.DataFrame({
        f"{name}_year": s.dt.year,
        f"{name}_month": s.dt.month,
        f"{name}_dow": s.dt.dayofweek,
        f"{name}_day": s.dt.day,
        f"{name}_is_wknd": s.dt.dayofweek.isin([5,6]).astype("float32"),
    })

for name in DATE_COLS:
    if name in df2.columns:
        feats = date_feats(df2[name], name)
        df2 = pd.concat([df2, feats], axis=1)
    else:
        for col in [f"{name}_year",f"{name}_month",f"{name}_dow",f"{name}_day",f"{name}_is_wknd"]:
            df2[col] = np.nan

df2["sched_transit_days_calc"]  = (df2["ETA"] - df2["ETD"]).dt.total_seconds() / 86400.0
df2["actual_transit_days_calc"] = (df2["ATA"] - df2["ATD"]).dt.total_seconds() / 86400.0
df2["late_vs_sched_days"]       = (df2["actual_transit_days_calc"] - df2["sched_transit_days_calc"])

# Now drop as in your API/train, keep same column order
X = df2.drop(columns=[c for c in DROP_COLS if c in df2.columns], errors="ignore")
y = df2["Delay Days"] if "Delay Days" in df2.columns else None

# Permutation Importance
model = pipeline[-1]
X_for_pred = pipeline.named_steps['prep'].transform(X)
result = permutation_importance(model, X_for_pred, y, n_repeats=5, random_state=42, n_jobs=-1)
importances = result.importances_mean

feature_names = NUM_COLS + CAT_COLS
# Automatically synchronize feature_names and importances if count mismatch
if len(importances) != len(feature_names):
    minlen = min(len(importances), len(feature_names))
    importances = importances[:minlen]
    feature_names = feature_names[:minlen]
fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
plt.figure(figsize=(10, 8))
fi.tail(20).plot(kind='barh')
plt.title('Top 20 Feature Importances (Permutation)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_perm.png')
print("Feature importance plot (permutation) saved as 'feature_importance_perm.png'")
