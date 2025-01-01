import copy
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np



class MeanPredictor(BaseEstimator, RegressorMixin):
    """A predictor that always predicts the mean value calculated during fit."""
    def __init__(self, config={}):
        self.mean_ = None
        self.config = copy.deepcopy(config)

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.mean_)

    def transform(self, X):
        return self.predict(X)
