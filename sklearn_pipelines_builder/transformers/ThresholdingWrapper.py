from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdingWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        """
        Wraps a model to apply thresholding to its outputs.

        Parameters:
        - model: The trained model that outputs probabilities.
        - threshold (float): The threshold for converting probabilities to binary labels.
        """
        self.model = model
        self.threshold = threshold

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        probabilities = self.model.predict(X)
        return (probabilities >= self.threshold).astype(int)  # Apply thresholding

    def predict_proba(self, X):
        # Return probabilities for compatibility
        return self.model.predict(X)

    @property
    def classes_(self):
        # Delegate to the wrapped estimator
        return self.model.classes_
