import numpy as np
import lightgbm as lgb

from sklearn_pipelines_builder.utils.basic_utils import get_features
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.infrastructure.Config import Config


global_config = Config()

scorer_dict = {
    "mse": "l2",
    "rmse": "rmse",
    "mae": "l1",
    "weighted_mae": "l1",
    'r2': "r2"
}


class LightGBMWrapper(BaseConfigurableTransformer):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_config = dict(n_estimators=500, verbose=-1, objective="regression",
                                 metric=scorer_dict.get(global_config.scoring, "rmse"))
        self.model_config.update(config.get("model_config", {}))

        self._model_type = self.config.get('element_type', 'lgbm_regressor')
        self.weight_column = global_config.get('weight_column', None)
        self.feature_importances_ = None
        self.classes_ = []

    def fit(self, X, y):
        feature_names = get_features(X)
        categorical_cols = list(X[feature_names].select_dtypes(include=['object', 'category']).columns)

        # Prepare weights
        if self.weight_column is None:
            weights = np.ones(len(X))
        else:
            weights = X[self.weight_column]
        X[feature_names] = X[feature_names].apply(
            lambda col: col.astype('category') if col.dtype == 'object' else col
        )
        X_ = X[feature_names]

        train_dataset = lgb.Dataset(X_, label=y, weight=weights, categorical_feature=categorical_cols, free_raw_data=False)
        self.model = lgb.train(self.model_config, train_set=train_dataset)

        self.feature_importances_ = self.model.feature_importance(importance_type='gain')
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba is not supported for regression.")

    def predict(self, X):
        feature_names = get_features(X)
        X[feature_names] = X[feature_names].apply(
            lambda col: col.astype('category') if col.dtype == 'object' else col
        )

        return self.model.predict(X[feature_names])

    def transform(self, X):
        feature_names = get_features(X)
        return self.model.predict(X[feature_names])

    def get_feature_importance(self):
        return self.feature_importances_

    def __call__(self, X):
        return self.predict(X)
