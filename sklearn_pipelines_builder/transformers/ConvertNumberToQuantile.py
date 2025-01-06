import copy
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn.preprocessing import QuantileTransformer


class ConvertNumberToQuantile(BaseConfigurableTransformer):
    def __init__(self, config=None):
        self.config = copy.deepcopy(config) if config is not None else {}
        self.feature_name = config.get('feature_name')
        self.transformer = QuantileTransformer(output_distribution='normal')


    def fit(self, X, y=None):
        self.transformer.fit(X[[self.feature_name]])
        return self

    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        X[self.feature_name] = self.transformer.transform(X[[self.feature_name]])
        return X
