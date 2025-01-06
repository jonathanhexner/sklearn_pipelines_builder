import os
import logging
import pandas as pd
from tpot import TPOTClassifier
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.factories.MultiColumnElementFactory import create_simple_transformer

# Global configuration
global_config = Config()

class TPOTWrapper(BaseConfigurableTransformer):
    def __init__(self, config=None):
        """
        TPOT wrapper for Scikit-learn compatibility.

        Parameters:
        - config (dict): Configuration dictionary for TPOT.
        """
        self.config = config or {}
        self.generations = self.config.get("generations", 5)
        self.population_size = self.config.get("population_size", 20)
        self.cv = self.config.get("cv", 5)
        self.random_state = self.config.get("random_state", 42)
        self.verbosity = self.config.get("verbosity", 2)
        self.pipeline_config = self.config.get("pipeline", [])  # Custom pipeline
        self.tpot = None
        self.best_pipeline = None

    def fit(self, X, y=None):
        """
        Fit the TPOT model and perform genetic optimization.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - self
        """
        logger.info("Initializing TPOT optimization.")

        # Build a custom pipeline if defined
        if self.pipeline_config:
            logger.info("Building custom pipeline.")
            pipeline = [create_simple_transformer(step) for step in self.pipeline_config]
            logger.info(f"Custom pipeline steps: {[str(step) for step in pipeline]}.")

        # Initialize TPOT
        self.tpot = TPOTClassifier(
            generations=self.generations,
            population_size=self.population_size,
            cv=self.cv,
            random_state=self.random_state,
            verbosity=self.verbosity
        )

        # Fit TPOT
        self.tpot.fit(X, y)

        # Extract the best pipeline
        self.best_pipeline = self.tpot.fitted_pipeline_
        logger.info(f"Best pipeline: {self.best_pipeline}")

        # Log to MLflow
        if SingleContainer.mlflow_run_id:
            import mlflow
            mlflow.log_params({
                "tpot_generations": self.generations,
                "tpot_population_size": self.population_size,
                "tpot_cv": self.cv
            })
            mlflow.log_text(str(self.best_pipeline), "tpot_best_pipeline.txt")

        # Save pipeline as Python script
        output_path = os.path.join(global_config.output_folder, global_config.run_name)
        os.makedirs(output_path, exist_ok=True)
        pipeline_script = os.path.join(output_path, "tpot_pipeline.py")
        self.tpot.export(pipeline_script)
        logger.info(f"Best pipeline script saved to: {pipeline_script}")

        return self

    def transform(self, X):
        """
        Transform the dataset using the best pipeline.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - pd.DataFrame: Transformed dataset.
        """
        if self.best_pipeline is None:
            raise ValueError("The TPOT model is not fitted yet. Call `fit` before `transform`.")

        logger.info("Transforming dataset using the best pipeline.")
        return self.best_pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit the TPOT model and transform the dataset.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - pd.DataFrame: Transformed dataset.
        """
        self.fit(X, y)
        return self.transform(X)
