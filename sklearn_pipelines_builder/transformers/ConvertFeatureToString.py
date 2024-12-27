import copy
from sklearn.base import TransformerMixin, BaseEstimator


class ConvertFeatureToString(BaseEstimator, TransformerMixin):
    def __init__(self, config={}):
        self.config = copy.deepcopy(config)
        self.features = config.get('features')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        X[self.features] = X[self.features].astype(str)
        return X