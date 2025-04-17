import numpy as np
from catboost import Pool
from catboost.utils import eval_metric

from sklearn_pipelines_builder.utils.basic_utils import get_features
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.models.CatBoostModelFactory import CatBoostModelFactory
from sklearn_pipelines_builder.infrastructure.Config import Config

global_config = Config()

scorer_dict = {
    "mse": "RMSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "weighted_mae": "MAE",
    'r2': "R2"
}

class CatBoostWrapper(BaseConfigurableTransformer):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_config = dict(iterations=500, verbose=False, eval_metric=scorer_dict[global_config.scoring])
        self.model_config.update(config.get("model_config", {}))
        self._model_type = self.config.get('element_type', 'catboost_regressor')
        self.weight_column = global_config.get('weight_column', None)
        self.string_features = []
        self.classes_ = []
        self.feature_importances_ = None
        self.weights = None

        self.importance_type = config.get('importance_type', 'LossFunctionChange')

    def fit(self, X, y):
        # Update cat_features in model_config based on dynamically identified features
        feature_names = get_features(X)
        self.string_features = list(X[feature_names].select_dtypes(include='object').columns)
        self.model_config['cat_features'] = self.string_features
        self.model = CatBoostModelFactory.create(self._model_type, **self.model_config)
        self.model.fit(X[feature_names], y)
        self.classes_ = np.unique(y)
        if self.weight_column is None:
            weights = [1]*len(X)
        else:
            weights = X[self.weight_column]
        train_pool = Pool(X[feature_names], label=y, feature_names=feature_names, cat_features=self.string_features,
                          weight=weights)

        self.feature_importances_ = self.model.get_feature_importance(type=self.importance_type, data=train_pool)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        feature_names = get_features(X)
        return self.model.predict(X[feature_names])

    def transform(self, X):
        feature_names = get_features(X)
        return self.model.transform(X[feature_names])

    def get_feature_importance(self):
        return self.feature_importances_

    def __call__(self, X):
        return self.predict(X)
