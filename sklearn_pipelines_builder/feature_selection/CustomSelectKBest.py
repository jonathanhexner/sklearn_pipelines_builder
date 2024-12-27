from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.feature_selection.ScoreFunctionFactory import ScoreFunctionFactory
import os

# Global configuration
global_config = Config()


class CustomSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        # Default configuration
        default_config = {
            'score_func_name': 'f_classif',  # Default to SelectKBest's default
            'k': 10  # Default number of features to select
        }

        # Override defaults with provided config
        self.config = {**default_config, **(config or {})}

        # Use the factory class to get the score function
        self.score_func = ScoreFunctionFactory.get(self.config['score_func_name'])
        if self.score_func is None:
            raise ValueError(f"Unsupported score_func_name: {self.config['score_func_name']}")

        self.k = self.config['k']
        self.selector = SelectKBest(score_func=self.score_func, k=self.k)

    def fit(self, X, y=None):
        logger.info("Fitting CustomSelectKBest with config: %s", self.config)
        self.selector.fit(X, y)

        # Log key information to MLflow
        logger.log_to_mlflow({
            'selected_features': self.selector.get_support(indices=True).tolist(),
            'score_func_name': self.config['score_func_name'],
            'k': self.k
        })
        return self

    def transform(self, X):
        logger.info("Transforming data using CustomSelectKBest")
        transformed_data = self.selector.transform(X)

        # Save key plots or diagnostics (dummy example)
        output_dir = os.path.join(global_config.output_folder, global_config.run_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.save_plot(os.path.join(output_dir, 'feature_scores.png'), self.selector.scores_)

        return transformed_data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        return self.selector.get_support(indices=indices)
