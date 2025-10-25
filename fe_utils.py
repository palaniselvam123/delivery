import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
            out[col] = (
                X[col].astype("object").map(self.maps_[col]).fillna(0.0).astype("float32")
            )
        return out.to_numpy()
