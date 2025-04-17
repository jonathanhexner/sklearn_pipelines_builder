from io import UnsupportedOperation
from typing import Dict, Any
import optuna
import numpy as np
import pandas as pd
from mlflow import catboost
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn_pipelines_builder.utils.basic_utils import get_features
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.validation.cross_val_score import cross_val_score
from sklearn_pipelines_builder.optuna_objectives.BaseObjective import BaseObjective
from sklearn_pipelines_builder.models.CatBoostModelFactory import CatBoostModelFactory
from sklearn_pipelines_builder.data_splitters.create_splitter import create_splitter
from sklearn_pipelines_builder.models.CatBoostWrapper import scorer_dict


global_config = Config()



class CatBoostObjective(BaseObjective):
    def __init__(self, config: Dict[str, Any]):

        default_param_distributions = {
            "iterations": {"method": "suggest_int", "low": 500, "high": 2000},
            "depth": {"method": "suggest_int", "low": 4, "high": 10},
            "learning_rate": {"method": "suggest_float", "low": 1e-3, "high": 1.0, "log": True},
            "l2_leaf_reg": {"method": "suggest_float", "low": 1, "high": 10},
            "border_count": {"method": "suggest_int", "low": 4, "high": 500},
            "bagging_temperature": {"method": "suggest_float", "low": 0, "high": 100},
        }
        self.cv_config = config.get("cv", {})
        self.cv_splitter = create_splitter(self.cv_config)
        self.model_config = config.get('model_config')
        self.weight_column = global_config.get('weight_column', None)
        if self.model_config.get('element_type')=='catboost_classifier':
            default_param_distributions.update({"scale_pos_weight": {"method": "suggest_float", "low": 1, "high": 30}})
        super().__init__(config, default_param_distributions)

    def __call__(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Objective function for Optuna.

        Parameters:
        - trial (optuna.Trial): The Optuna trial object.
        - X: Features as a DataFrame.
        - y: Labels as a Series.

        Returns:
        - float: Cross-validation score.
        """
        params = self.suggest_hyperparameters(trial)
        model = self._setup_model(X, params)

        # Cross-validation
        # n_splits = self.config.get("n_splits", 5)
        # cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Use cross_val_score to evaluate the model
        return np.mean(cross_val_score(model, X, y, cv=self.cv_splitter, scoring=global_config.scoring))

    def _setup_model(self, X, params):
        params.update(dict(verbose=False))
        feature_names = get_features(X)
        string_features = list(X[feature_names].select_dtypes(include='object').columns)
        params.update({'cat_features': string_features})
        params.update(dict(eval_metric = scorer_dict[global_config.scoring]))
        model = CatBoostModelFactory.create(self.model_config.get('element_type'), **params)
        return model

    def train_best_model(self, best_params: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """
        Train the best CatBoost model using the best parameters.

        Parameters:
        - best_params (dict): Best hyperparameters.
        - X: Features as a DataFrame.
        - y: Labels as a Series.

        Returns:
        - CatBoostClassifier: Trained model.
        """
        model = self._setup_model(X, best_params)
        feature_names = get_features(X)
        string_features = list(X[feature_names].select_dtypes(include='object').columns)
        train_pool = cb.Pool(X[feature_names], label=y, cat_features=string_features)
        if self.weight_column is not None:
            train_pool = cb.Pool(X[feature_names], label=y, weight=X[self.weight_column], cat_features=string_features)
        model.fit(train_pool)
        return model

