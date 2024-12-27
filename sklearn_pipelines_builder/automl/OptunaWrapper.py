import copy

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.optuna_objectives.ObjectiveFactory import ObjectiveFactory
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
import mlflow


class OptunaWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        """
        Wrapper for Optuna hyperparameter optimization compatible with Scikit-learn API.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - model_name (str): Name of the model to optimize.
            - param_distributions (dict): Hyperparameter search space.
            - scoring (str): Metric for evaluation.
            - cv (int): Number of cross-validation folds.
            - n_trials (int): Number of optimization trials.
            - random_state (int): Seed for reproducibility.
            - pruner (str): Type of Optuna pruner to use.
            - pruner_config (dict): Parameters for the pruner.
        """
        self.config = config
        self.model_name = config.get("model_name")
        self.param_distributions = config.get("param_distributions", {})
        self.cv = config.get("cv", 3)
        self.n_trials = config.get("n_trials", 2)
        self.random_state = config.get("random_state", None)
        self.scoring = config.get("scoring", Config().scoring)  # Default to global scoring
        self.pruner = self._get_pruner(config.get("pruner", "median"), config.get("pruner_config", {}))
        self.best_params_ = None
        self.best_model_ = None

    def _get_pruner(self, pruner_name, pruner_config):
        """
        Create an Optuna pruner based on the name and configuration.

        Parameters:
        - pruner_name (str): Name of the pruner.
        - pruner_config (dict): Pruner-specific configuration.

        Returns:
        - Optuna pruner object.
        """
        if pruner_name == "median":
            return MedianPruner(**pruner_config)
        elif pruner_name == "hyperband":
            return HyperbandPruner(**pruner_config)
        elif pruner_name == "none":
            return NopPruner()
        else:
            raise ValueError(f"Unsupported pruner: {pruner_name}")

    def fit(self, X, y):
        """
        Fit the model using Optuna for hyperparameter optimization.

        Parameters:
        - X: Features.
        - y: Target.
        """
        logger.info(f"Starting Optuna optimization for {self.model_name}")
        mlflow.log_param("model_name", self.model_name)

        # Create the objective using the factory
        objective = ObjectiveFactory.create_objective(self.config)

        # Create and optimize the study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=self.pruner,
        )
        study.optimize(lambda trial: objective(trial, X, y), n_trials=self.n_trials)

        # Save the best parameters and model
        self.best_params_ = study.best_params
        mlflow.log_params(self.best_params_)
        logger.info(f"Best parameters found: {self.best_params_}")
        params = {'verbose': False}
        params.update(copy.deepcopy(self.best_params_))
        self.best_model_ = objective.train_best_model(params, X, y)
        mlflow.log_metric("best_cv_score", study.best_value)
        logger.info(f"Best CV score: {study.best_value}")
        return self

    def transform(self, X):
        """
        Transform input data using the best model.

        Parameters:
        - X: Features.

        Returns:
        - Predictions: Predictions from the best model.
        """
        if self.best_model_ is None:
            raise ValueError("Model is not fitted yet. Call `fit` first.")
        return self.best_model_.predict(X)

    def predict(self, X):
        """
        Transform input data using the best model.

        Parameters:
        - X: Features.

        Returns:
        - Predictions: Predictions from the best model.
        """
        if self.best_model_ is None:
            raise ValueError("Model is not fitted yet. Call `fit` first.")
        return self.best_model_.predict(X)
