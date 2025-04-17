import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
global_config = Config()
single_container = SingleContainer()

class RobustScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.columns = config.get("columns", None)
        self.scaler = RobustScaler()
        self.fitted_columns = None

    def fit(self, X, y=None):
        # Select target columns: specified or all numeric
        if self.columns is not None:
            self.fitted_columns = self.columns
        else:
            self.fitted_columns = [col for col in X.select_dtypes(include='number').columns.tolist()
                                    if col not in single_container.meta_training_columns + ['response_copy']]

        for col in self.fitted_columns:
            X[col].replace([np.inf, -np.inf], X[col].min(), inplace=True)
            X[col].fillna(X[col].mean(), inplace=True)


        self.scaler.fit(X[self.fitted_columns])
        return self

    def transform(self, X):
        for col in self.fitted_columns:
            X[col].replace([np.inf, -np.inf], X[col].min(), inplace=True)
            X[col].fillna(X[col].mean(), inplace=True)

        X[self.fitted_columns] = self.scaler.transform(X[self.fitted_columns])

        return X
