import logging
import pandas as pd
from featurewiz import FeatureWiz  # Ensure featurewiz is installed
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer


class FeatureWizWrapper(BaseConfigurableTransformer):
    def __init__(self, config=None):
        """
        Featurewiz wrapper for Scikit-learn compatibility.

        Parameters:
        - config (dict): Configuration dictionary for Featurewiz.
        """
        super().__init__(config)
        self.target = self.config.get("target", SingleContainer.response)
        self.corr_limit = self.config.get("corr_limit", 0.70)
        self.feature_engg = self.config.get("feature_engg", ['interactions'])
        self.verbose = self.config.get("verbose", 1)
        self.imbalanced = self.config.get('imbalanced', False)
        self.category_encoders = config.get('category_encoders', 'auto')
        self.selected_features = []
        self.feature_wiz = None


    def fit(self, X, y=None):
        """
        Fit the Featurewiz model to select features.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable. Optional if target is in X.

        Returns:
        - self
        """
        if y is not None:
            X = pd.concat([X, pd.Series(y, name=self.target)], axis=1)
        elif self.target not in X.columns:
            raise ValueError(f"The target column '{self.target}' must be included in the dataset or passed as `y`.")

        logger.info("Starting Featurewiz fit with target: %s", self.target)
        logger.info("Using corr_limit=%.2f and feature_engg=%s", self.corr_limit, self.feature_engg)

        # Propagate logs from featurewiz to our logger
        featurewiz_logger = logging.getLogger("featurewiz")
        featurewiz_logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            featurewiz_logger.addHandler(handler)

        self.feature_wiz = FeatureWiz(feature_engg=self.feature_engg, nrows=None,
                                      category_encoders=self.category_encoders, add_missing=False,
                                      verbose=self.verbose, corr_limit=self.corr_limit,
                                      )
        self.feature_wiz = self.feature_wiz.fit(X, y)
        logger.info("Featurewiz selected features: %s", self.feature_wiz.features)

        # Log the results to MLflow
        if SingleContainer.mlflow_run_id:
            import mlflow
            mlflow.log_params({
                "featurewiz_corr_limit": self.corr_limit,
                "featurewiz_feature_engg": self.feature_engg,
                "featurewiz_selected_features_count": len(self.selected_features)
            })
            mlflow.log_text("\n".join(self.selected_features), "featurewiz_selected_features.txt")

        return self

    def transform(self, X):
        """
        Transform the dataset to include only selected features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - pd.DataFrame: Transformed dataset with selected features.
        """
        logger.info("Transforming dataset using selected features.")
        return self.feature_wiz.transform(X)[0]




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
