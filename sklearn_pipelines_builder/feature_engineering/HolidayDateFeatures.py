import copy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HolidayDateFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer to create robust date-specific features for each city using a deduplicated subset.

    Attributes:
    - date_column: str, the name of the date column.
    - city_column: str, the name of the city column.
    """

    def __init__(self, config=None):
        self.config = copy.deepcopy(config) if config is not None else {}
        self.date_column = self.config.get('date_column', 'date')
        self.city_column = self.config.get('city_column', 'city')

    def fit(self, X, y=None):
        """No fitting necessary for this transformer."""
        return self

    def transform(self, X):
        """
        Transform the dataset by creating city-specific date features in a deduplicated subset
        and merging them back to the original dataset.

        Parameters:
        - X: pd.DataFrame, input DataFrame with date, city, and holiday-related columns.

        Returns:
        - pd.DataFrame: Original DataFrame with additional robust features.
        """
        X = X.copy()

        # Ensure the date column is in datetime format
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')

        if X[self.date_column].isna().any():
            raise ValueError(f"Invalid dates found in column '{self.date_column}'.")

        # Create a subset with unique city-date pairs
        subset = X[[self.city_column, self.date_column]].drop_duplicates()

        # Compute basic features on the subset
        subset['is_weekend'] = subset[self.date_column].dt.weekday.isin([5, 6]).astype(int)

        # Previous and next day holiday indicators
        subset['is_next_day_holiday'] = (
            subset.groupby(self.city_column)[self.date_column]
            .shift(-1)
            .isin(X.loc[X['holiday'] == 1, self.date_column])
            .astype(int)
        )
        subset['is_prev_day_holiday'] = (
            subset.groupby(self.city_column)[self.date_column]
            .shift(1)
            .isin(X.loc[X['holiday'] == 1, self.date_column])
            .astype(int)
        )

        # Calculate last and next store open days
        subset['last_store_open'] = None
        subset['next_store_open'] = None

        for city in subset[self.city_column].unique():
            city_subset = subset[subset[self.city_column] == city]
            open_days = city_subset.loc[~city_subset[self.date_column].isin(
                X.loc[X['holiday'] == 1, self.date_column]
            ) & (city_subset['is_weekend'] == 0), self.date_column]

            # Last store open days
            subset.loc[subset[self.city_column] == city, 'last_store_open'] = city_subset[self.date_column].apply(
                lambda d: (d - open_days[open_days < d].max()).days if not open_days[open_days < d].empty else None
            )

            # Next store open days
            subset.loc[subset[self.city_column] == city, 'next_store_open'] = city_subset[self.date_column].apply(
                lambda d: (open_days[open_days > d].min() - d).days if not open_days[open_days > d].empty else None
            )

        # Fill missing values
        subset['last_store_open'] = subset['last_store_open'].fillna(-1).astype(int)
        subset['next_store_open'] = subset['next_store_open'].fillna(-1).astype(int)

        # Merge the subset back to the original DataFrame
        result = X.merge(subset, on=[self.city_column, self.date_column], how='left')

        return result
