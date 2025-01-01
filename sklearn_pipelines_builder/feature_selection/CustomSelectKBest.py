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
        score_func = ScoreFunctionFactory().create(self.config['score_func_name'])
        if score_func is None:
            raise ValueError(f"Unsupported score_func_name: {self.config['score_func_name']}")

        self.k = self.config['k']
        self.selector = SelectKBest(score_func=score_func, k=self.k)
        self.selected_features = None

    def fit(self, X, y=None):
        logger.info("Fitting CustomSelectKBest with config: %s", self.config)
        self.selector.fit(X, y)
        self.selected_features = list(X.columns[self.selector.get_support(indices=True)])
        # Log key information to MLflow
        logger.info({
            'selected_features': self.selected_features,
            'score_func_name': self.config['score_func_name'],
            'k': self.k
        })

        # # Save key plots or diagnostics (dummy example)
        # output_dir = os.path.join(global_config.output_folder, global_config.run_name)
        # os.makedirs(output_dir, exist_ok=True)
        # logger.save_plot(os.path.join(output_dir, 'feature_scores.png'), self.selector.scores_)

        return self

    def transform(self, X):
        logger.info("Transforming data using CustomSelectKBest")
        transformed_data = X[self.selected_features]


        return transformed_data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        return self.selector.get_support(indices=indices)
