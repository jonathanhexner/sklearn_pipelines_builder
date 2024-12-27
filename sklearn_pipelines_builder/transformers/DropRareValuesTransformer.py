import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropRareValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config={}):
        """
        Initialize with the columns to check and the rarity threshold.

        Parameters:
        - columns: List of columns to check for rare values.
        - threshold: Minimum frequency (relative) below which rows are dropped.
        """
        self.config = config
        self.columns = config.get('columns')
        self.threshold = config.get('threshold', 0.001)
        self.value_counts_ = {}

    def fit(self, X, y=None):
        """
        Compute the frequency of values in the specified columns.

        Parameters:
        - X: Input DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")

        self.value_counts_ = {
            col: X[col].value_counts(normalize=True) for col in self.columns
        }
        return self

    def transform(self, X):
        """
        Drop rows with rare values in the specified columns.

        Parameters:
        - X: Input DataFrame.

        Returns:
        - Transformed DataFrame with rows containing rare values dropped.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")

        X_transformed = X.copy()
        for col in self.columns:
            rare_values = self.value_counts_[col][self.value_counts_[col] < self.threshold].index
            X_transformed = X_transformed[~X_transformed[col].isin(rare_values)]
        return X_transformed.reset_index(drop=True)
