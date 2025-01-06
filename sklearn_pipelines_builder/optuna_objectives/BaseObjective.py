import optuna
from typing import Dict, Any
import pandas as pd


class BaseObjective:
    def __init__(self, config: Dict[str, Any], default_param_distributions: Dict[str, Any]):
        """
        Base class for Optuna objectives.

        Parameters:
        - config (dict): Configuration dictionary, can include 'params_override'.
        - default_param_distributions (dict): Default hyperparameter distributions.
        """
        self.config = config
        self.default_param_distributions = default_param_distributions.copy()

    def apply_overrides(self):
        """
        Apply parameter overrides from the config to the default distributions.
        """
        overrides = self.config.get("params_override", {})
        updated_distributions = self.default_param_distributions.copy()

        for param_name, override in overrides.items():
            if param_name in updated_distributions:
                updated_distributions[param_name].update(override)

        return updated_distributions

    def suggest_hyperparameters(self, trial: optuna.Trial):
        """
        Generate hyperparameters dynamically using Optuna.

        Parameters:
        - trial (optuna.Trial): The Optuna trial object.

        Returns:
        - params (dict): Suggested hyperparameters.
        """
        param_distributions = self.apply_overrides()
        params = {}

        for param_name, dist in param_distributions.items():
            dist_copy = dist.copy()
            method = dist_copy.pop("method")
            params[param_name] = getattr(trial, method)(param_name, **dist_copy)

        return params

    def train_best_model(self, best_params: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """
        Train the best model using the best parameters.
        Override this method in derived classes for specific behavior.

        Parameters:
        - best_params (dict): Best hyperparameters.
        - X: Features as a DataFrame.
        - y: Labels as a Series.

        Returns:
        - Trained model.
        """
        raise NotImplementedError("train_best_model must be implemented in the derived class.")
