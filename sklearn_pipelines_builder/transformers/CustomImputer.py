from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, columns=None):
        """
        Initialize with an estimator (imputer or model) and the columns to apply it to.
        - `estimator`: Can be an imputer (e.g., SimpleImputer) or a model (e.g., RandomForestRegressor).
        - `columns`: List of columns to apply the imputation to.
        """
        self.estimator = estimator
        self.columns = columns  # Columns to apply imputation to

    def fit(self, X, y=None):
        # Check if columns are specified, otherwise use all columns
        if self.columns is None:
            self.columns = X.columns

        # Select only the columns specified for imputation
        X_selected = X[self.columns]

        # Apply different fitting logic based on estimator type
        if hasattr(self.estimator, 'transform'):
            # If it has a transform method, treat it as an imputer
            self.estimator.fit(X_selected)
        elif hasattr(self.estimator, 'predict'):
            # If it has a predict method, treat it as a model
            mask = ~np.isnan(X_selected)
            self.estimator.fit(X_selected[mask], X_selected[mask].ravel())
        return self

    def predict(self, X):
        # Ensure estimator is fitted
        check_is_fitted(self, 'estimator')

        # Apply imputation only to specified columns
        X_transformed = X.copy()
        if hasattr(self.estimator, 'transform'):
            X_transformed[self.columns] = self.estimator.transform(X[self.columns])
        elif hasattr(self.estimator, 'predict'):
            mask = np.isnan(X[self.columns])
            X_transformed.loc[mask, self.columns] = self.estimator.predict(X[mask])

        return X_transformed
