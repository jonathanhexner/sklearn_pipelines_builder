import copy
import os
import pandas as pd
import dill
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump, load
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.utils.logger import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
global_config = Config()
# from my_utils import calc_measures
single_container = SingleContainer()


class SelectBestImputer(BaseEstimator, TransformerMixin):
    def __init__(self, config={}, output_folder=None):
        self.config = copy.deepcopy(config)
        self.column = config.get('column')
        self.output_folder = output_folder or global_config.output_folder
        self.imputation_methods = config.get('imputation_methods', [])
        self.best_model = None
        self.best_r2 = None
        self.used_features = []
        os.makedirs(self.output_folder, exist_ok=True)

    def _train_test_split(self, df):
        """Split the data into training and test sets."""
        train_size = global_config.get('imputation_train_size', 0.8)
        df_train, df_test = train_test_split(df, train_size=train_size, random_state=42)
        return df_train, df_test

    def fit(self, X, y):
        logger.info(f"Starting imputation for column: {self.column}")
        if not y.isnull().any():
            logger.info(f"Skipping column {self.column}: no missing values.")
            return self
        X[self.column] = y
        all_results = pd.DataFrame()
        # Prepare training and testing data
        df_dropna = X.dropna(subset=[self.column])
        features = list(set(df_dropna.columns) - set([self.column]))
        df_train, df_test = self._train_test_split(df_dropna)
        # Why is this needed? Not clear
        # df_train = pd.concat([df_train, X[X[self.column].isnull()]]).reset_index(drop=True)

        # Iterate over imputation methods
        for config in self.imputation_methods:
            element_alias = config.get('element_alias', config['element_type'])
            logger.info(f"Testing imputation for column %s alias %s, name %s:", self.column, element_alias,
                        config['element_type'])
            config.update({'column': self.column})
            pipe_line_model = ElementFactory().create_pipe_line(config)
            pipe_line_model.fit(df_train[features], df_train[self.column])
            pred = pipe_line_model.transform(df_test[features])
            r2 = r2_score(df_test[self.column], pred)

            # Save results
            if self.best_r2 is None or r2 > self.best_r2:
                self.best_r2 = r2
                self.best_model = pipe_line_model

            logger.info(f"Column {self.column}:  achieved R2: {r2}")

            # Save the results to disk
            df_results = pd.DataFrame({'Feature': self.column, 'Model_Name': config['element_type'], 'R2': r2}, index=[0])
            all_results = pd.concat([all_results, df_results]).reset_index(drop=True)

        # Save results and best model
        all_results.to_csv(os.path.join(self.output_folder, f'{self.column}_imputation_results.csv'), index=False)
        with open(os.path.join(self.output_folder, f'{self.column}_best_model.joblib'), "wb") as f_out:
            dill.dump(self.best_model, f_out)
        return self

    def transform(self, X):
        if self.best_model is None:
            raise ValueError(f"The imputer for column '{self.column}' has not been fitted yet.")
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")

        null_indices = X[self.column].isnull()
        if null_indices.any():
            X.loc[null_indices, self.column] = self.best_model.transform(X.loc[null_indices, :])
        return X[self.column] # TODO: Consider if it's better to reutrn only transformed column or all the data


