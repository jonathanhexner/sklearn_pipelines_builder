import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.selection import RecursiveFeatureElimination
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory

global_config = Config()

class FeatureEngineRfe(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        """
        Initialize the transformer with the given configuration.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - "model": The underlying model (e.g., GradientBoostingClassifier).
            - "threshold": Maximum performance drop allowed to remove a feature.
            - "cv": Number of cross-validation folds.
            - "output_folder": Folder to save the plots.
        """
        self.config = config
        self.model = ElementFactory().create(config.get("model_config", {}))
        self.threshold = config.get("threshold", 0.0005)
        self.cv = config.get("cv", 2)
        self.output_folder = global_config.output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Use scoring from the global Config
        self.scorer = global_config.scoring

    def fit(self, X, y):
        """
        Fit the transformer by performing Recursive Feature Elimination.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - self
        """
        # Initialize RFE
        self.rfe_selector = RecursiveFeatureElimination(
            variables=None,
            estimator=self.model,
            scoring=global_config.scoring,  # Pass the scoring function
            threshold=self.threshold,
            cv=self.cv
        )

        # Fit the RFE selector
        self.rfe_selector.fit(X, y)

        # Log initial model performance to MLflow
        # with mlflow.start_run(run_id=SingleContainer.mlflow_run_id):
        mlflow.log_metric("initial_rfe_model_performance", self.rfe_selector.initial_model_performance_)

        # Plot and save feature importances
        self._plot_feature_importances()

        return self

    def transform(self, X):
        """
        Transform the input dataset by removing unimportant features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - Transformed DataFrame with selected features.
        """
        if not hasattr(self, "rfe_selector"):
            raise ValueError("The transformer has not been fitted yet.")
        return self.rfe_selector.transform(X)

    def evaluate_final_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the final model using the selected features.

        Parameters:
        - X_train, X_test: Train and test features.
        - y_train, y_test: Train and test targets.

        Logs final model performance and returns the final score.
        """
        # Fit the model on selected features
        self.model.fit(X_train, y_train)

        # Make predictions
        if global_config.get('prediction_type') in ['class', 'regression', 'binary']:
            y_pred_test = self.model.predict(X_test)
        else:
            y_pred_test = self.model.predict_proba(X_test)[:, 1]
        # if hasattr(self.model, "predict_proba"):  # For classification
        #     y_pred_test = self.model.predict_proba(X_test)[:, 1]
        # else:  # For regression or binary classification
        #     y_pred_test = self.model.predict(X_test)

        # Calculate and log the final score
        final_score = self.scorer._score_func(y_test, y_pred_test)
        # with mlflow.start_run(run_id=SingleContainer.mlflow_run_id):
        mlflow.log_metric("final_rfe_model_performance", final_score)

        return final_score

    def _plot_feature_importances(self):
        """
        Plot and save feature importances from the RFE selector.
        """
        feature_importances = pd.Series(self.rfe_selector.feature_importances_, index=self.rfe_selector.variables_).sort_values(ascending=False)

        plt.figure(figsize=(20, 6))
        feature_importances.plot.bar()
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importances from Recursive Feature Elimination")

        plot_path = os.path.join(self.output_folder, "feature_importances.png")
        plt.savefig(plot_path)
        plt.close()

        # Log the plot to MLflow
        mlflow.log_artifact(plot_path)
