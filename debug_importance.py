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

model = pipeline[-1]
print("Model:", type(model))
print("Attributes:", dir(model))
print("n_features_in_:", getattr(model, 'n_features_in_', 'NA'))
print("categories_: ", hasattr(model, 'categories_'))
print("Has feature_importances_:", hasattr(model, 'feature_importances_'))
print("Feature importance attr (if any):", getattr(model, 'feature_importances_', 'NA'))
print("Categorical features:", getattr(model, 'categorical_features', 'NA'))
print("Is model fitted?:", getattr(model, '_is_fitted', 'NA'), getattr(model, 'n_iter_', 'NA'))
