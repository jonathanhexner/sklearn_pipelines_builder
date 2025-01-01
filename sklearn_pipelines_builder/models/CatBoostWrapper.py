import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier, Pool


class CatBoostWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_params = dict(iterations=500, verbose=False)
        self.model_params.update(config.get("model_params", {}))
        self.string_features = []
        self.classes_ = []
        self.feature_importances_ = None

        self.importance_type = config.get('importance_type', 'LossFunctionChange')

    def fit(self, X, y):
        # Update cat_features in model_params based on dynamically identified features
        self.string_features = list(X.select_dtypes(include='object').columns)
        self.model_params['cat_features'] = self.string_features
        self.model = CatBoostClassifier(**self.model_params)
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        train_pool = Pool(X, label=y, feature_names=list(X.columns))

        self.feature_importances_ = self.model.get_feature_importance(type=self.importance_type, data=train_pool)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def transform(self, X):
        return self.model.transform(X)
