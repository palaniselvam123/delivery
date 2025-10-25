import sys
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import fe_utils
from fe_utils import FrequencyEncoder  # noqa: F401

MODEL_PATH = r"C:\Users\Dell\Documents\ML Projects\Delivery Time predictor\data\delay_days_best_model.joblib"
pipeline = joblib.load(MODEL_PATH)

print("Pipeline steps:", pipeline)
print("Last step type:", type(pipeline[-1]))
print("Last step object:", pipeline[-1])
print("Has .feature_importances_?", hasattr(pipeline[-1], "feature_importances_"))
if hasattr(pipeline[-1], "feature_importances_"):
    print("Feature importances shape:", pipeline[-1].feature_importances_.shape)
