import copy
import numpy as np
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer

class MeanPredictor(BaseConfigurableTransformer):
    """
    A predictor that always predicts the mean value of the target variable calculated during fit.

    This predictor is compatible with Scikit-learn's BaseEstimator and RegressorMixin,
    making it suitable for use in Scikit-learn pipelines.
    """
    def __init__(self, config=None):
        """
        Initialize the MeanPredictor.

        Parameters:
        - config (dict, optional): Configuration dictionary. Defaults to None.
        """
        super().__init__(config)
        self.mean_ = None

    def fit(self, X, y):
        """
        Fit the predictor by calculating the mean of the target variable.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Features (ignored in this implementation).
        - y (pd.Series or np.ndarray): Target variable.

        Returns:
        - self: The fitted instance.
        """
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        """
        Predict the target variable using the mean value.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Features (used to determine the number of predictions).

        Returns:
        - np.ndarray: An array of predictions, all equal to the mean value.
        """
        return np.full(shape=(len(X),), fill_value=self.mean_)

    def transform(self, X):
        """
        Transform the input by returning predictions.

        Parameters:
        - X (pd.DataFrame or np.ndarray): Features.

        Returns:
        - np.ndarray: An array of predictions, all equal to the mean value.
        """
        return self.predict(X)
