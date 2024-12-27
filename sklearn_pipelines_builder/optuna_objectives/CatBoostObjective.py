import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.validation.cross_val_score import cross_val_score

global_config = Config()



class CatBoostObjective:
    def __init__(self, config):
        """
        Objective for optimizing CatBoost with Optuna.

        Parameters:
        - config (dict): Configuration dictionary.
        """
        self.config = config
        self.default_param_distributions = {
            "iterations": {"method": "suggest_int", "low": 500, "high": 2000},
            "depth": {"method": "suggest_int", "low": 4, "high": 10},
            "learning_rate": {"method": "suggest_float", "low": 1e-3, "high": 1.0, "log": True},
            "l2_leaf_reg": {"method": "suggest_float", "low": 1, "high": 10},
            "border_count": {"method": "suggest_int", "low": 4, "high": 500},
            "bagging_temperature": {"method": "suggest_float", "low": 0, "high": 100},
            "scale_pos_weight": {"method": "suggest_float", "low": 1, "high": 30},
        }

    def suggest_hyperparameters(self, trial):
        """
        Dynamically generate hyperparameters for Optuna.

        Parameters:
        - trial (optuna.Trial): The Optuna trial object.

        Returns:
        - params (dict): Suggested hyperparameters.
        """
        param_distributions = self.config.get("param_distributions", copy.deepcopy(self.default_param_distributions))
        params = {'verbose': False}

        for param_name, dist in param_distributions.items():
            method = dist.pop("method")
            params[param_name] = getattr(trial, method)(param_name, **dist)

        return params

    def __call__(self, trial, X, y):
        from catboost import CatBoostClassifier

        # Generate hyperparameters dynamically
        model_params = self.suggest_hyperparameters(trial)

        # Initialize and evaluate the model
        model = CatBoostClassifier(**model_params)
        n_splits = self.config.get("n_splits", 5)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        scoring = global_config.get('scoring')
        return np.mean(cross_val_score( X, y, model, cv=cv, scoring=global_config.scoring))

    def train_best_model(self, best_params, X, y):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**best_params)
        model.fit(X, y)
        return model
