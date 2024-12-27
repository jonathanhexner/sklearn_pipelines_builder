import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from borutashap import BorutaShap
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config

global_config = Config()

class RedirectHandler(logging.Handler):
    """Custom handler to redirect logs from one logger to another."""
    def emit(self, record):
        logger.handle(record)  # Redirect to your custom logger

class BorutaShapWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        """
        BorutaShap wrapper for Scikit-learn compatibility.

        Parameters:
        - config (dict): Configuration dictionary for BorutaShap.
        """
        self.config = config or {}
        self.model = self.config.get("model", None)  # Pass the model for feature importance
        self.target = self.config.get("target", SingleContainer.response)
        self.train_samples = self.config.get("train_samples", 0.8)  # Default 80% of samples for training
        self.verbose = self.config.get("verbose", True)
        self.random_state = self.config.get("random_state", 42)
        self.selected_features = []

    def fit(self, X, y=None):
        """
        Fit the BorutaShap model to select features.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - self
        """
        if y is None:
            raise ValueError("Target variable `y` is required for BorutaShap.")

        # Initialize BorutaShap with the given model and settings
        boruta_shap = BorutaShap(
            model=self.model,
            importance_measure='shap',
            classification=True if global_config.get('prediction_type') == 'classification' else False
        )

        logger.info("Starting BorutaShap feature selection.")
        boruta_shap.fit(X=X, y=y, n_trials=100, sample=self.train_samples, verbose=self.verbose, random_state=self.random_state)

        # Store the selected features
        self.selected_features = boruta_shap.support_  # Boolean mask of selected features
        self.selected_feature_names = list(X.columns[self.selected_features])

        logger.info(f"Selected features: {self.selected_feature_names}")

        # Log the results to MLflow
        if SingleContainer.mlflow_run_id:
            import mlflow
            mlflow.log_params({
                "borutashap_train_samples": self.train_samples,
                "borutashap_selected_features_count": len(self.selected_feature_names),
            })
            mlflow.log_text(",".join(self.selected_feature_names), "borutashap_selected_features.txt")

        return self

    def transform(self, X):
        """
        Transform the dataset to include only selected features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - pd.DataFrame: Transformed dataset with selected features.
        """
        if not self.selected_features:
            raise ValueError("The model is not fitted yet or no features were selected. Call `fit` first.")

        logger.info("Transforming dataset using selected features.")
        return X[self.selected_feature_names]

    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the dataset.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - pd.DataFrame: Transformed dataset with selected features.
        """
        self.fit(X, y)
        return self.transform(X)
