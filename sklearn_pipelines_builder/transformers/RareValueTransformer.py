import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.utils.logger import logger

class RareValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        """
        Initialize the transformer with a configuration dictionary.

        Parameters:
        - config: A dictionary containing configuration parameters:
            - "threshold": Minimum percentage frequency for common values (default: 0.01).
            - "placeholder": The value to replace rare or unseen values with (default: "Rare").
        """
        self.threshold = config.get("threshold", 0.01)
        self.placeholder = config.get("placeholder", "Rare")
        self.common_values_ = {}
        self.config = config

    def fit(self, X, y=None):
        """
        Identify common (non-rare) values for each string column based on the threshold.

        Parameters:
        - X: The input DataFrame.

        Returns:
        - self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")

        # Identify string columns
        self.string_columns_ = X.select_dtypes(include=['object', 'string']).columns

        # Identify common values for each string column
        self.common_values_ = {}
        for col in self.string_columns_:
            value_counts = X[col].value_counts(normalize=True)
            common_values = value_counts[value_counts >= self.threshold].index.tolist()
            self.common_values_[col] = common_values

        return self

    def transform(self, X):
        """
        Replace rare or unseen values in each string column with the placeholder.

        Parameters:
        - X: The input DataFrame.

        Returns:
        - Transformed DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")

        X = X.copy()

        # Replace rare or unseen values with the placeholder
        for col in self.string_columns_:
            common_values = self.common_values_.get(col, [])
            X[col] = X[col].apply(lambda x: x if x in common_values else self.placeholder)
            logger.info('Done rare value transforming for column %s', col)
        return X
