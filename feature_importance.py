import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make sure fe_utils.py can be imported from current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import fe_utils
from fe_utils import FrequencyEncoder  # noqa: F401

# Path to your trained model file
MODEL_PATH = r"C:\Users\Dell\Documents\ML Projects\Delivery Time predictor\data\delay_days_best_model.joblib"
pipeline = joblib.load(MODEL_PATH)

NUM_COLS = [
    'Transit Days','Estimated Transit Days','LegNumber','PortCongestion',
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
feature_names = NUM_COLS + CAT_COLS

# Use the model part of the pipeline
model = pipeline[-1]  # HistGradientBoostingRegressor

# Check if feature_importances_ exists (scikit-learn >=1.1.0)
if not hasattr(model, "feature_importances_"):
    raise AttributeError(
        "Your scikit-learn version does not support feature_importances_ for HistGradientBoostingRegressor. "
        "Please upgrade by running: pip install -U 'scikit-learn>=1.1.0'"
    )

importances = model.feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
fi.tail(20).plot(kind='barh')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("DONE! Feature importance plot saved as 'feature_importance.png' in this folder.")
