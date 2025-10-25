# shipment_delay_model.py
# Predict Delay Days with leakage-safe CV and compact categorical handling.

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import randint, uniform
import warnings

# ==================== CONFIG ====================
DATA_PATH = r"C:\Users\Dell\Documents\ML Projects\Delivery Time predictor\data\routes_20000 (3).csv"
TARGET = "Delay Days"
GROUP_COL = "Shipment Number"
RANDOM_STATE = 42

TEMPORAL_HOLDOUT = True     # last 20% by ATD for forward-looking evaluation
TEST_SIZE = 0.20
LOW_CARD_THRESHOLD = 170    # native categorical only if categories <= 170

# ==================== LOAD ====================
df = pd.read_csv(DATA_PATH)

# ==================== REQUIRED COLUMNS CHECK ====================
required = [
    "Route No","Shipment Number","Vessel","Voyage","Carrier","Load Port","Discharge Port",
    "ETD","ETA","ATD","ATA","Estimated Transit Days","Transport Mode",
    "Delay Days","Early/Late","LegNumber","WasRolled","RolloverReason","PaymentStatus",
    "DocumentSubmitted","WeatherSeverity","PortCongestion","GeoRiskFlag",
    "CarrierNotification","ExternalNewsImpact"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ==================== DATETIME & FEATURES ====================
for c in ["ETD","ETA","ATD","ATA"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")

def date_feats(s: pd.Series, name: str):
    return pd.DataFrame({
        f"{name}_year": s.dt.year,
        f"{name}_month": s.dt.month,
        f"{name}_dow": s.dt.dayofweek,
        f"{name}_day": s.dt.day,
        f"{name}_is_wknd": s.dt.dayofweek.isin([5,6]).astype("float32"),
    })

etdF = date_feats(df["ETD"], "ETD")
etaF = date_feats(df["ETA"], "ETA")
atdF = date_feats(df["ATD"], "ATD")
ataF = date_feats(df["ATA"], "ATA")

df["sched_transit_days_calc"]  = (df["ETA"] - df["ETD"]).dt.total_seconds() / 86400.0
df["actual_transit_days_calc"] = (df["ATA"] - df["ATD"]).dt.total_seconds() / 86400.0
df["late_vs_sched_days"]       = df["actual_transit_days_calc"] - df["sched_transit_days_calc"]

df = pd.concat([df, etdF, etaF, atdF, ataF], axis=1)

# ==================== TARGET ====================
y = pd.to_numeric(df[TARGET], errors="coerce").astype(float)

# ==================== FEATURE SET ====================
num_cols = [
    "Estimated Transit Days","LegNumber","PortCongestion",
    "sched_transit_days_calc","actual_transit_days_calc","late_vs_sched_days",
    "ETD_year","ETD_month","ETD_dow","ETD_day","ETD_is_wknd",
    "ETA_year","ETA_month","ETA_dow","ETA_day","ETA_is_wknd",
    "ATD_year","ATD_month","ATD_dow","ATD_day","ATD_is_wknd",
    "ATA_year","ATA_month","ATA_dow","ATA_day","ATA_is_wknd"
]

cat_cols_all = [
    "Vessel","Voyage","Carrier","Load Port","Discharge Port","Transport Mode",
    "Early/Late","WasRolled","RolloverReason","PaymentStatus","DocumentSubmitted",
    "WeatherSeverity","GeoRiskFlag","CarrierNotification","ExternalNewsImpact"
]

drop_cols = ["Route No","Shipment Number","ETD","ETA","ATD","ATA","Transit Days",TARGET]
X = df.drop(columns=drop_cols, errors="ignore")

num_cols = [c for c in num_cols if c in X.columns]
cat_cols_all = [c for c in cat_cols_all if c in X.columns]

# Split categoricals by cardinality to satisfy HGBR native category limits
card = {c: X[c].astype("object").nunique(dropna=True) for c in cat_cols_all}
low_card_cols  = [c for c, k in card.items() if k <= LOW_CARD_THRESHOLD]
high_card_cols = [c for c, k in card.items() if k >  LOW_CARD_THRESHOLD]
print("Low-cardinality categoricals (native categorical):", low_card_cols)
print("High-cardinality categoricals (frequency encoded):", high_card_cols)

# ==================== FREQUENCY ENCODER ====================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.maps_ = None
        self.columns_ = None
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.maps_ = {}
        for col in self.columns_:
            vc = X[col].astype("object").value_counts(normalize=True)
            self.maps_[col] = vc
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            out[col] = X[col].astype("object").map(self.maps_[col]).fillna(0.0).astype("float32")
        return out.to_numpy()

# ==================== PREPROCESSING ====================
ord_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

numeric_tf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
])

cat_low_tf = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("enc", ord_enc),
])

cat_high_tf = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("freq", FrequencyEncoder()),
])

# Keep numeric first, then low-card, then high-card for a clean categorical mask
preprocess = ColumnTransformer(
    transformers=[
        ("num",      numeric_tf,   num_cols),
        ("cat_low",  cat_low_tf,   low_card_cols),
        ("cat_high", cat_high_tf,  high_card_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0
)

# Categorical mask for HGBR: only low-card columns are native categorical
cat_mask = np.array(
    [False]*len(num_cols) +
    [True]*len(low_card_cols) +
    [False]*len(high_card_cols)
)

model = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
    categorical_features=cat_mask,
    max_bins=255
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model),
])

# ==================== SPLIT (TIME-AWARE) ====================
if TEMPORAL_HOLDOUT and "ATD" in df.columns:
    df_sorted = df.sort_values("ATD")
    cutoff = int((1 - TEST_SIZE) * len(df_sorted))
    test_idx = df_sorted.index[cutoff:]
    train_idx = df_sorted.index[:cutoff]
else:
    rs = np.random.RandomState(RANDOM_STATE)
    test_idx = df.sample(frac=TEST_SIZE, random_state=RANDOM_STATE).index
    train_idx = df.index.difference(test_idx)

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

# ==================== CV (LEAKAGE-SAFE) ====================
groups = df.loc[train_idx, GROUP_COL].astype(str).fillna("NA_GROUP")
gkf = GroupKFold(n_splits=5)

# ==================== HYPERPARAM SEARCH ====================
param_dist = {
    "model__learning_rate": uniform(0.03, 0.17),
    "model__max_depth": randint(3, 9),
    "model__max_leaf_nodes": randint(15, 63),
    "model__min_samples_leaf": randint(10, 60),
    "model__l2_regularization": uniform(0.0, 0.2),
    "model__max_bins": randint(128, 255),
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=40,
    cv=gkf.split(X_train, y_train, groups=groups),
    scoring="neg_mean_absolute_error",
    n_jobs=1,         # use 1 to cap peak RAM; increase if you have headroom
    verbose=1,
    random_state=RANDOM_STATE,
    refit=True,
    error_score="raise"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    search.fit(X_train, y_train)

best = search.best_estimator_

# ==================== EVALUATE ====================
pred_test = best.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
mse = mean_squared_error(y_test, pred_test)     # compatible with older sklearn
rmse = float(np.sqrt(mse))                      # manual RMSE for version safety
r2 = r2_score(y_test, pred_test)

print("Best params:", search.best_params_)
print(f"Test MAE:  {mae:0.3f}")
print(f"Test RMSE: {rmse:0.3f}")
print(f"Test R2:   {r2:0.3f}")

# ==================== EXPORT ====================
outdir = Path(DATA_PATH).parent
pred_path = outdir / "delay_days_predictions_test.csv"
model_path = outdir / "delay_days_best_model.joblib"

out = df.loc[test_idx, ["Route No","Shipment Number","LegNumber"]].copy()
out["y_true"] = y_test.values
out["y_pred"] = pred_test
out.to_csv(pred_path, index=False)
print(f"Saved predictions -> {pred_path}")

try:
    import joblib
    joblib.dump(best, model_path)
    print(f"Saved model -> {model_path}")
except Exception as e:
    print(f"Model save skipped: {e}")
