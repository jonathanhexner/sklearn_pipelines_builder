
from typing import Dict, Any
import optuna
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold

from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.optuna_objectives.BaseObjective import BaseObjective

global_config = Config()

class NNObjective(BaseObjective):
    def __init__(self, config: Dict[str, Any]):
        default_param_distributions = {
            "num_layers": {"method": "suggest_int", "low": 1, "high": 3},
            "units": {"method": "suggest_int", "low": 16, "high": 128, "step": 16},
            "activation": {"method": "suggest_categorical", "choices": ["relu", "tanh", "sigmoid"]},
            "last_activation": {"method": "suggest_categorical", "choices": ["sigmoid"]},
            "dropout_rate": {"method": "suggest_float", "low": 0.0, "high": 0.5},
            "learning_rate": {"method": "suggest_float", "low": 1e-4, "high": 1e-2, "log": True},
        }
        self._loss = config.get("loss", "binary_crossentropy")
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
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Build the neural network model
        model = Sequential()
        model.add(Dense(params["units"], activation=params["activation"], input_shape=(X.shape[1],)))
        for _ in range(params["num_layers"] - 1):
            model.add(Dense(params["units"], activation=params["activation"]))
            model.add(Dropout(params["dropout_rate"]))
        model.add(Dense(1, activation=params["last_activation"]))

        optimizer = Adam(learning_rate=params["learning_rate"])
        model.compile(optimizer=optimizer, loss=self._loss, metrics=[global_config.scoring])

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config.get("n_splits", 5), shuffle=True, random_state=42)
        val_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Adjust epochs as needed
            val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
            val_scores.append(val_accuracy)

        return np.mean(val_scores)

    def train_best_model(self, best_params: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """
        Train the best model using the best parameters.

        Parameters:
        - best_params (dict): Best hyperparameters.
        - X: Features as a DataFrame.
        - y: Labels as a Series.

        Returns:
        - model: Trained model.
        """
        model = Sequential()
        model.add(Dense(best_params["units"], activation=best_params["activation"], input_shape=(X.shape[1],)))
        for _ in range(best_params["num_layers"] - 1):
            model.add(Dense(best_params["units"], activation=best_params["activation"]))
            model.add(Dropout(best_params["dropout_rate"]))
        model.add(Dense(1, activation=best_params["last_activation"]))

        optimizer = Adam(learning_rate=best_params["learning_rate"])
        model.compile(optimizer=optimizer, loss=self._loss, metrics=[global_config.scoring])

        # Train the model
        model.fit(X, y, epochs=50, batch_size=32, verbose=1)  # Adjust epochs and batch size as needed
        return model
