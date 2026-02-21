"""
Custom transformers for hand landmark preprocessing.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HandCentering(BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self):
        return True
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.copy(X)
        X = X.reshape((X.shape[0], -1, 3))
        reference_point = X[:, 0, :]
        X = X - reference_point[:, np.newaxis, :]
        X = X.reshape((X.shape[0], -1))
        return X


class HandNormalization(BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self):
        return True
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.copy(X)
        X = X.reshape((X.shape[0], -1, 3))
        max_val = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        X = X / (max_val + 1e-8)
        X = X.reshape((X.shape[0], -1))
        return X
