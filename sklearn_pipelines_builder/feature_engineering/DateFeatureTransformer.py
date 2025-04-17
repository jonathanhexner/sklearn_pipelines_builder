import copy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create useful features from a date column.

    Attributes:
    - date_column: str, the name of the date column to transform.
    """

    def __init__(self, config=None):
        self.config = copy.deepcopy(config) if config is not None else {}
        self.date_column = config.get('date_column', 'date')

    def fit(self, X, y=None):
        """
        No fitting necessary for this transformer.
        """
        return self

    def transform(self, X):
        """
        Transforms the date column into multiple useful date-related features.

        Parameters:
        - X: pd.DataFrame, input DataFrame with a date column.

        Returns:
        - pd.DataFrame: DataFrame with new date-related features.
        """
        X = X.copy()

        # Ensure the date column is in datetime format
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')

        if X[self.date_column].isna().any():
            raise ValueError(f"Invalid dates found in column '{self.date_column}'.")

        # Extract date-related features
        X[f"{self.date_column}_year"] = X[self.date_column].dt.year
        X[f"{self.date_column}_month"] = X[self.date_column].dt.month
        X[f"{self.date_column}_week"] = X[self.date_column].dt.isocalendar().week
        X[f"{self.date_column}_day"] = X[self.date_column].dt.day
        X[f"{self.date_column}_weekday"] = X[self.date_column].dt.weekday  # Monday=0, Sunday=6
        X[f"{self.date_column}_is_weekend"] = X[f"{self.date_column}_weekday"].isin([5, 6]).astype(int)
        X[f"{self.date_column}_quarter"] = X[self.date_column].dt.quarter
        X[f"{self.date_column}_is_month_start"] = X[self.date_column].dt.is_month_start.astype(int)
        X[f"{self.date_column}_is_month_end"] = X[self.date_column].dt.is_month_end.astype(int)
        X[f"{self.date_column}_day_of_year"] = X[self.date_column].dt.day_of_year
        X[f"{self.date_column}_days_in_month"] = X[self.date_column].dt.days_in_month
        # ✅ Extract Year-Month
        X["year_month"] = X[self.date_column].dt.strftime("%Y-%m")


        return X
