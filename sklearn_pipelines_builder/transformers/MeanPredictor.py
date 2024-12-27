from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
import pandas as pd


class MeanPredictor(BaseEstimator, RegressorMixin):
    """A predictor that always predicts the mean value calculated during fit."""

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)

