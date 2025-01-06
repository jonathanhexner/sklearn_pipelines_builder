import logging
import os
import pandas as pd
from autogluon.tabular import TabularPredictor

from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer

global_config = Config()

autogluon_logger = logging.getLogger("autogluon.tabular.predictor.predictor")

class RedirectHandler(logging.Handler):
    """
    Custom handler to redirect logs from one logger to another.
    """

    def emit(self, record):
        logger.handle(record)  # Redirect to your custom logger

class AutoGluonWrapper(BaseConfigurableTransformer):
    """
    Wrapper for AutoGluon TabularPredictor to make it compatible with Scikit-learn pipelines.

    This wrapper allows AutoGluon to be used as part of a Scikit-learn pipeline for fitting and transforming data.
    """

    def __init__(self, config=None):
        """
        Initialize the AutoGluonWrapper.

        Parameters:
        - config (dict): Configuration dictionary for AutoGluon.
        """
        super().__init__(config)
        self.time_limit = self.config.get("time_limit", 60)  # Default to 60 seconds
        self.presets = self.config.get("presets", "medium_quality_faster_train")
        self._auto_gluon_config = {
            'label': self.config.get("label", SingleContainer.response),
            'problem_type': self.config.get("problem_type", global_config.get('prediction_type')),
            'eval_metric': self.config.get("eval_metric", global_config.scoring),
            'log_to_file': True,
            'log_file_path': os.path.join(
                global_config.output_folder,
                global_config.run_name,
                'AutoGluon.log'
            ),
        }
        self.label_column = self._auto_gluon_config['label']
        self.predictor = None

    def fit(self, X, y=None):
        """
        Fit the AutoGluon model.

        Parameters:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Target variable.

        Returns:
        - self: The fitted instance.
        """
        if y is not None:
            X = pd.concat([X, pd.Series(y, name=self.label_column)], axis=1)
        elif self.label_column not in X.columns:
            raise ValueError(
                f"The label column '{self.label_column}' must be included in the dataset or passed as `y`."
            )

        autogluon_logger.addHandler(RedirectHandler())
        autogluon_logger.setLevel(logging.INFO)
        autogluon_logger.info("This log is redirected to your custom logger.")

        self.predictor = TabularPredictor(**self._auto_gluon_config).fit(
            X, time_limit=self.time_limit, presets=self.presets
        )

        leaderboard = self.predictor.leaderboard(silent=True)
        leaderboard_path = os.path.join(
            global_config.output_folder, global_config.run_name, 'AutoGluonModel.csv'
        )
        leaderboard.to_csv(leaderboard_path, index=False)

        return self

    def transform(self, X):
        """
        Predict using the AutoGluon model.

        Parameters:
        - X (pd.DataFrame): Features.

        Returns:
        - pd.DataFrame: Predictions as a DataFrame.
        """
        if self.predictor is None:
            raise ValueError("The model is not fitted yet. Call `fit` before `transform`.")

        predictions = self.predictor.predict(X)
        return pd.DataFrame(predictions, columns=["prediction"])

    def fit_transform(self, X, y=None): # pylint: disable=arguments-differ
        """
        Fit the model and return predictions for the training data.

        Parameters:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Target variable.

        Returns:
        - pd.DataFrame: Predictions as a DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        Predict using the AutoGluon model.

        Parameters:
        - X (pd.DataFrame): Features.

        Returns:
        - pd.Series: Model predictions.
        """
        if self.predictor is None:
            raise ValueError("The model is not fitted yet. Call `fit` before `predict`.")

        return self.predictor.predict(X)
