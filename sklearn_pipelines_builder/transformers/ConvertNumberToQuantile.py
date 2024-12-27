import copy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import QuantileTransformer


class ConvertNumberToQuantile(BaseEstimator, TransformerMixin):
    def __init__(self, config={}):
        self.config = copy.deepcopy(config)
        self.feature_name = config.get('feature_name')
        self.transformer = QuantileTransformer(output_distribution='normal')


    def fit(self, X, y=None):
        self.transformer.fit(X[[self.feature_name]])
        return self

    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        X[self.feature_name] = self.transformer.transform(X[[self.feature_name]])
        return X
