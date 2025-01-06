from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd

class NumericStandardScaler(BaseConfigurableTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.numeric_columns = None

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        # Select only numeric columns
        self.numeric_columns = X.select_dtypes(include='number').columns
        # Fit the scaler only on numeric columns
        self.scaler.fit(X[self.numeric_columns])
        return self

    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        X_copy = X.copy()
        X_copy[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])
        return X_copy
