import copy
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer


class ConvertFeatureToString(BaseConfigurableTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.features = config.get('features')

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        return self

    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        X[self.features] = X[self.features].astype(str)
        return X
