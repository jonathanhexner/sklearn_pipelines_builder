import numpy as np
import copy

from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.SingletonContainer import SingleContainer

import os

# Global configuration
global_config = Config()


class SelectImputerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.config = config
        self.column = self.config.get('column')

        self.columns = self.config.get('columns')
        if self.columns is None:
            self.columns = list(set(SingleContainer.null_numeric_features) - set(SingleContainer.columns_to_drop))
        self.pipelines = {}
        self.features_for_imputation = []

    def _setup_features_for_imputation(self, X):
        """
        get the list of available features to predict a null column
        """
        self.string_features = list(X.select_dtypes(include='object').columns)
        self.numeric_features = list(set(X.columns) - set(self.string_features))

        df = copy.deepcopy(X)
        # df[self.column] = y
        self.df_is_null = df.isnull()
        self.df_null_corr = self.df_is_null.corr()
        df_is_null_mean = self.df_is_null.mean()
        columns_to_exclude = list(df_is_null_mean[df_is_null_mean > 0.4].keys())
        self.available_numeric_features = [col for col in self.numeric_features if col not in columns_to_exclude]
        logger.info('Excluding columns {} due to high null ratio'.format(','.join(columns_to_exclude)))

        self.df_null_corr.drop(columns=columns_to_exclude, inplace=True)

    def _get_features_for_imputation(self, X):

        if not self.df_is_null[self.column].any():
            logger.info(f"Skipping column {self.column} because it doesn't contains NaN values.")
            return self

        correlated_null_values = self.df_null_corr[self.column]
        columns_with_zero_nan_corr = list(correlated_null_values[correlated_null_values == 0].keys()) + \
                                     list(correlated_null_values[correlated_null_values.isna()].keys())
        df_ = X[columns_with_zero_nan_corr + [self.column]].dropna(axis=0)

        features = columns_with_zero_nan_corr
        self.string_features = [col for col in self.string_features if col in features]
        self.available_numeric_features = [col for col in self.available_numeric_features if col in features]

        self.df_corr = df_[self.available_numeric_features].corr('spearman').abs()

        corr_matrix = self.df_corr[self.available_numeric_features][self.available_numeric_features]
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
        remaining_features = list(set(features) - set(to_drop))
        return remaining_features

    def fit(self, X, y=None):
        # Determine columns dynamically if not provided
        if self.columns is None:
            from sklearn_pipelines_builder.SingletonContainer import SingleContainer
            self.columns = list(set(SingleContainer.null_numeric_features) - set(SingleContainer.columns_to_drop))

        logger.info("Columns for imputation: %s", self.columns)
        X[self.column] = y
        self._setup_features_for_imputation(X)
        self.features_for_imputation = self._get_features_for_imputation(X)
        return self

    def transform(self, X):
        logger.info("Transforming data using MultiColumnImputer")
        return X[self.features_for_imputation]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
