import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class MostFrequentPredictor(BaseEstimator, ClassifierMixin):
    """
    A predictor that always predicts the most frequent value of the target variable calculated during fit.

    This predictor is compatible with Scikit-learn's BaseEstimator and ClassifierMixin,
    making it suitable for use in Scikit-learn pipelines.
    """
    def __init__(self, config=None):
        """
        Initialize the MostFrequentPredictor.

        Parameters:
        - config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.most_frequent_ = None
        self.config = copy.deepcopy(config) if config is not None else {}

    def fit(self, X, y):  # pylint: disable=unused-argument
        """
        Fit the predictor by finding the most frequent value of the target variable.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Features (ignored in this implementation).
        - y (pd.Series or np.ndarray): Target variable.

        Returns:
        - self: The fitted instance.
        """
        self.most_frequent_ = pd.Series(y).mode()[0]
        return self

    def predict(self, X):
        """
        Predict the target variable using the most frequent value.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Features (used to determine the number of predictions).

        Returns:
        - np.ndarray: An array of predictions, all equal to the most frequent value.
        """
        return np.full(shape=(len(X),), fill_value=self.most_frequent_)

    def transform(self, X):
        """
        Transform the input by returning predictions.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Features.

        Returns:
        - np.ndarray: An array of predictions, all equal to the most frequent value.
        """
        return self.predict(X)
