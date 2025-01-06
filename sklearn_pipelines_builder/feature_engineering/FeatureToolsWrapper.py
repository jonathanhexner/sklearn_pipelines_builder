import os
from featuretools import dfs, EntitySet, calculate_feature_matrix
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer


global_config = Config()

# Featuretools Wrapper
class FeatureToolsWrapper(BaseConfigurableTransformer):
    def __init__(self, config=None):
        """
        FeatureTools wrapper for Scikit-learn compatibility.

        Parameters:
        - config (dict): Configuration dictionary for FeatureTools.
        """
        self.default_config = {"max_depth": 2}
        self.default_config.update(config.get('config', {}))
        self.feature_defs = None
        self.selected_features = None

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """
        Fit the FeatureTools feature engineering process.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable (optional).

        Returns:
        - self
        """
        logger.info("Starting FeatureTools feature engineering.")

        es = EntitySet(id="dataset")
        es.add_dataframe(dataframe_name="data", dataframe=X, index=X.index.name or 'index')

        # feature_matrix, feature_defs = dfs(
        #     entityset=es, target_dataframe_name="data", max_depth=self.max_depth,
        #     trans_primitives=["add_numeric", "multiply_numeric"],
        # )
        feature_matrix, self.feature_defs = dfs(
            entityset=es, target_dataframe_name="data", **self.default_config
        )

        self.selected_features = feature_matrix.columns.tolist()

        # Log key results
        logger.info("Generated features: %s", self.selected_features)

        # Save results to MLflow
        if SingleContainer.mlflow_run_id:
            import mlflow
            mlflow.log_text(",".join(self.selected_features), "featuretools_generated_features.txt")

        # Save key plots
        output_path = os.path.join(global_config.output_folder, global_config.run_name)
        os.makedirs(output_path, exist_ok=True)
        feature_matrix.to_csv(os.path.join(output_path, "feature_matrix.csv"), index=False)

        return self

    def transform(self, X):
        """
        Transform the dataset using generated features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - pd.DataFrame: Transformed dataset with new features.
        """
        logger.info("Transforming dataset with FeatureTools features.")
        es_test = EntitySet(id="example_test")
        es_test.add_dataframe(dataframe_name="data", dataframe=X, index="id")

        feature_matrix_test = calculate_feature_matrix(features=self.feature_defs, entityset=es_test)

        return feature_matrix_test
