import numpy as np
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory


class ModelImputer(BaseConfigurableTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = ElementFactory().create(config.get("model_config", {}))

    def fit(self, X, y=None):
        # Separate rows with missing values
        not_nan_mask = ~np.isnan(X).any(axis=1)
        self.model.fit(X[not_nan_mask], y[not_nan_mask])
        return self

    def transform(self, X):
        # Predict missing values
        X_filled = X.copy()
        nan_mask = np.isnan(X)
        X_filled[nan_mask] = self.model.predict(X_filled[~nan_mask])
        return X_filled
