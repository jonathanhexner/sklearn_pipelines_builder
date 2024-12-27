from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
import pandas as pd


class MostFrequentPredictor(BaseEstimator, ClassifierMixin):
    """A predictor that always predicts the most frequent value calculated during fit."""

    def fit(self, X, y):
        self.most_frequent_ = pd.Series(y).mode()[0]
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.most_frequent_)
