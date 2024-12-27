import copy

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.utils.logger import logger

class PipelineAggregation(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        """
        Generic wrapper for assembling a pipeline based on a configuration dictionary.

        Parameters:
        - config (dict): Configuration dictionary with a list of pipeline elements.
          Example:
          {
              "elements": [
                  {"element_type": "select_k_best", "max_features": 5},
                  {"element_type": "linear_regression"}
              ]
          }
        """
        self.config = config
        self.pipeline = None
        # TODO: Reconsider if this is the right way to pass this parameter initially used for imputation
        self.copy_fields = ['column']
        self.create_pipe_line()


    def fit(self, X, y=None):
        """Build and fit the pipeline based on the configuration."""

        # Build the pipeline
        # self.pipeline.fit(X, y)
        Xt  = copy.deepcopy(X)
        for n_step, (step_name, step) in enumerate(self.pipeline.steps):
            Xt = step.fit_transform(Xt, y)


        return self

    def create_pipe_line(self):
        steps = []
        for element_config in self.config.get("elements", []):
            element_type = element_config.get("element_type")
            if not element_type:
                raise ValueError("Each element in the pipeline config must specify an 'element_type'.")
            for field in self.copy_fields:
                element_config.update({field: self.config[field]})
            # Create element using ElementFactory
            element = ElementFactory().create(element_config)
            steps.append((element_type, element))
        self.pipeline = Pipeline(steps)

    def transform(self, X):
        """Apply the transformations defined in the pipeline."""
        if self.pipeline is None:
            raise ValueError("The pipeline has not been fitted yet.")
        return self.pipeline.transform(X)
