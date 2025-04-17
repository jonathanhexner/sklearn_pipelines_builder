import pandas as pd
from sklearn.model_selection import BaseCrossValidator

class DateBasedCV(BaseCrossValidator):
    def __init__(self, config):
        """
        Custom CV based on date column for time series data.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - date_column (str): Column containing date values.
            - threshold_dates (list): List of threshold dates for splits.
        """
        self.date_column = config.get("date_column")
        self.threshold_dates = config.get("threshold_dates", [])
        self.n_splits = len(self.threshold_dates)

    def split(self, X, y=None, groups=None):
        """
        Generate train-test splits based on threshold dates.

        Parameters:
        - X: DataFrame with the date column.
        - y: Ignored for this implementation.
        - groups: Ignored for this implementation.

        Yields:
        - train_indices, test_indices: Indices for train and test sets.
        """
        if X[self.date_column].dtype != 'datetime64[ns]':
            X[self.date_column] = pd.to_datetime(X[self.date_column])
        train_test_tuples = []
        for threshold_date in self.threshold_dates:
            train_indices = X[X[self.date_column] < pd.to_datetime(threshold_date)].index
            test_indices = X[X[self.date_column] >= pd.to_datetime(threshold_date)].index
            train_test_tuples.append((train_indices, test_indices))
        return train_test_tuples

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits
