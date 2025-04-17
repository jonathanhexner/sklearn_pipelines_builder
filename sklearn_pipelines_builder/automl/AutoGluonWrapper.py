import logging
import os
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from sklearn_pipelines_builder.utils.basic_utils import get_features
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.utils.custom_scorer import get_custom_scorer


global_config = Config()

autogluon_logger = logging.getLogger("autogluon.tabular.predictor.predictor")

scorer_dict = {
    "mse": "MSE",         # ✅ AutoGluon uses "mean_squared_error"
    "rmse": "RMSE",   # ✅ Corrected from "RMSE" to "root_mean_squared_error"
    "mae": "MAE",        # ✅ AutoGluon uses "mean_absolute_error"
    "weighted_mae": "MAE",  # ✅ Still maps to MAE (if weighted is handled separately)
    "r2": "r2"                           # ✅ "r2" is correct
}


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
        self.model_type = self.config.get("model_type", "tabular")
        self.date_column = self.config.get('date_column')
        self.freq = self.config.get('freq')
        self.id_column = self.config.get('id_column')
        self.presets = self.config.get("presets", "medium_quality_faster_train")
        self._auto_gluon_config = {
            'label': self.config.get("label", SingleContainer.response),
            'eval_metric': scorer_dict[config.get("scoring", global_config.scoring)],
            'log_to_file': True,
            'log_file_path': os.path.join(
                global_config.output_folder,
                global_config.run_name,
                'AutoGluon.log'
            ),
        }
        if self.model_type == 'time_series':
            self._auto_gluon_config.update({'freq': self.freq})

        else:
            self._auto_gluon_config.update({'problem_type':
                                                self.config.get("problem_type", global_config.get('prediction_type'))})

        self.weight_column = global_config.get('weight_column', None)
        self.label_column = self._auto_gluon_config['label']
        self.model = None
        self.classes_ = None

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
        features = get_features(X)

        if self.date_column is not None:
            if X[self.date_column].dtype == 'object':
                X[self.date_column] = pd.to_datetime(X[self.date_column])

        if self.weight_column is not None:
            sample_weights_train = X[self.weight_column]
        else:
            sample_weights_train = [1]*len(X)

        if self.model_type == 'tabular':
            self.model = TabularPredictor(**self._auto_gluon_config).fit(
                X[features], time_limit=self.time_limit, presets=self.presets, sample_weight=sample_weights_train
            )
        elif self.model_type == 'time_series':
            X_ts = self.create_time_series_dataframe(X.copy(True), features)
            self.model = TimeSeriesPredictor(**self._auto_gluon_config).fit(
                X_ts, time_limit=self.time_limit, presets=self.presets)

        leaderboard = self.model.leaderboard(silent=True)
        leaderboard_path = os.path.join(
            global_config.output_folder, 'AutoGluonModel.csv'
        )
        leaderboard.to_csv(leaderboard_path, index=False)
        # Extract and set the class labels

        if self._auto_gluon_config.get("problem_type") in ["binary", "multiclass"]:
            self.classes_ = list(self.model.class_labels)
        else:
            self.classes_ = None  # For regression tasks, there are no class labels
        return self

    def create_time_series_dataframe(self, X, features):
        keep_columns = list(set(features + [self.id_column, self.date_column]))
        X_ts = TimeSeriesDataFrame.from_data_frame(X[keep_columns], id_column=self.id_column,
                                                   timestamp_column=self.date_column)
        return X_ts

    def transform(self, X):
        """
        Predict using the AutoGluon model.

        Parameters:
        - X (pd.DataFrame): Features.

        Returns:
        - pd.DataFrame: Predictions as a DataFrame.
        """
        return X

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
        if self.model is None:
            raise ValueError("The model is not fitted yet. Call `fit` before `predict`.")
        features = get_features(X)
        if self.model_type == 'time_series':
            X_ts = self.create_time_series_dataframe(X, features)
            self.model.predict(X_ts)
        else:
            return self.model.predict(X[features])
