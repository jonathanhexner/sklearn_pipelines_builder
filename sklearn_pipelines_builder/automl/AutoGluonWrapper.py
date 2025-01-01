import logging
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from autogluon.tabular import TabularPredictor

from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
global_config = Config()

autogluon_logger = logging.getLogger("autogluon.tabular.predictor.predictor")



class RedirectHandler(logging.Handler):
    """Custom handler to redirect logs from one logger to another."""
    def emit(self, record):
        logger.handle(record)  # Redirect to your custom logger




class AutoGluonWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        """
        AutoGluon wrapper for Scikit-learn compatibility.

        Parameters:
        - config (dict): Configuration dictionary for AutoGluon.
        """
        self.config = config or {}
        self.time_limit= self.config.get("time_limit", 60)  # Default to 60 seconds
        self.presets= self.config.get("presets", "medium_quality_faster_train")
        self._auto_gluon_config = {'label': self.config.get("label", SingleContainer.response),
                                   'problem_type': self.config.get("problem_type", global_config.get('prediction_type')),
                                   'eval_metric': self.config.get("eval_metric", global_config.scoring),
                                   'log_to_file': True,
                                   'log_file_path': os.path.join(global_config.output_folder, global_config.run_name,
                                                                 'AutoGluon.log')}
        self.predictor = None

    def fit(self, X, y=None):
        """
        Fit the AutoGluon model.

        Parameters:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Target variable.

        Returns:
        - self
        """
        if y is not None:
            X = pd.concat([X, pd.Series(y, name=self._auto_gluon_config['label'])], axis=1)
        elif self.label_column not in X.columns:
            raise ValueError(
                f"The label column '{self._auto_gluon_config['label']}' must be included in the dataset or passed as `y`.")
        # Attach the custom handler
        autogluon_logger.addHandler(RedirectHandler())
        autogluon_logger.setLevel(logging.INFO)

        # Test logging
        autogluon_logger.info("This log is redirected to your custom logger.")

        # autogluon_logger.propagate = False

        # Clear any existing handlers in AutoGluon's logger
        # autogluon_logger.handlers.clear()

        # Attach your logger's handlers to AutoGluon's logger
        # for handler in logger.handlers:
        #     autogluon_logger.addHandler(handler)
        self.predictor = TabularPredictor(**self._auto_gluon_config).fit(X, time_limit=self.time_limit,
                                                                         presets=self.presets)
        leaderboard = self.predictor.leaderboard(silent=True)
        leaderboard.to_csv(os.path.join(global_config.output_folder, global_config.run_name, 'AutoGluonModel.csv'),
                           index=False)

        return self

    def transform(self, X):
        """
        Predict using the AutoGluon model.

        Parameters:
        - X (pd.DataFrame): Features.

        Returns:
        - predictions (pd.Series or pd.DataFrame): Model predictions.
        """
        if self.predictor is None:
            raise ValueError("The model is not fitted yet. Call `fit` before `transform`.")

        predictions = self.predictor.predict(X)
        return pd.DataFrame(predictions, columns=["prediction"])

    def fit_transform(self, X, y=None):
        """
        Fit the model and return predictions for the training data.

        Parameters:
        - X (pd.DataFrame): Features.
        - y (pd.Series): Target variable.

        Returns:
        - predictions (pd.Series or pd.DataFrame): Model predictions.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return self.predictor.predict(X)
