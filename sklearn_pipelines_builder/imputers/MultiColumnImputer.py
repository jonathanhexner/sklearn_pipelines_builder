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


class MultiColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        # Default configuration
        default_config = {
            'columns': None,  # If None, determine columns dynamically
            'actual_imputer_config': {'element_type': 'mean'}  # Default imputation strategy

        }
        # Override defaults with provided config
        self.config = {**default_config, **(config or {})}
        self.columns = self.config['columns']
        self.actual_imputer_config = self.config['actual_imputer_config']
        if self.columns is None:
            self.columns = list(set(SingleContainer.null_numeric_features) - set(SingleContainer.columns_to_drop))
        self.pipelines = {}

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

    def _get_features_for_imputation(self, X, column):

        if not self.df_is_null[column].any():
            logger.info(f"Skipping column {column} because it doesn't contains NaN values.")
            return self

        correlated_null_values = self.df_null_corr[column]
        columns_with_zero_nan_corr = list(correlated_null_values[correlated_null_values == 0].keys()) + \
                                     list(correlated_null_values[correlated_null_values.isna()].keys())
        df_ = X[columns_with_zero_nan_corr + [column]].dropna(axis=0)

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
            self.columns = sorted(list(set(SingleContainer.null_numeric_features) - set(SingleContainer.columns_to_drop)))

        logger.info("Columns for imputation: %s", self.columns)
        self._setup_features_for_imputation(X)
        # Create and fit pipelines for each column
        for column in self.columns:
            logger.info("Creating pipeline for column: %s", column)
            config = copy.deepcopy(self.actual_imputer_config)
            config.update({'column': column})
            features_for_imputation = self._get_features_for_imputation(X, column)
            pipeline = ElementFactory().create(config)
            pipeline.fit(X[features_for_imputation], X[column])
            self.pipelines[column] = pipeline

        return self

    def transform(self, X):
        logger.info("Transforming data using MultiColumnImputer")
        X_transformed = X.copy()

        for column, pipeline in self.pipelines.items():
            logger.info("Transforming column: %s", column)
            for step_name, step_transformer in pipeline.steps:
                logger.info("Running transformer %s", step_name)
                X_transformed[column] = step_transformer.transform(X_transformed)

        # Save logs or diagnostics if needed
        output_dir = os.path.join(global_config.output_folder, global_config.run_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Transformation complete. Results saved in %s", output_dir)

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
