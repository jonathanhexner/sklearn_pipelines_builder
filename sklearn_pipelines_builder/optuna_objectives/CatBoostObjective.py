from io import UnsupportedOperation
from typing import Dict, Any
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.validation.cross_val_score import cross_val_score
from sklearn_pipelines_builder.optuna_objectives.BaseObjective import BaseObjective
from sklearn_pipelines_builder.models.CatBoostModelFactory import CatBoostModelFactory
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
            "scale_pos_weight": {"method": "suggest_float", "low": 1, "high": 30},
        }
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
        params.update(dict(verbose=False))
        model = CatBoostModelFactory.create(self.config.get('model_name'),**params)

        # Cross-validation
        n_splits = self.config.get("n_splits", 5)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Use cross_val_score to evaluate the model
        return np.mean(cross_val_score(model, X, y, cv=cv, scoring="accuracy"))

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
        best_params.update({'verbose': False})
        model = CatBoostModelFactory.create(self.config.get('model_name'), **best_params)
        model.fit(X, y)
        return model

