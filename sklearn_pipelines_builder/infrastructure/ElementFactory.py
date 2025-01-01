import pandas as pd
import importlib
from sklearn.pipeline import Pipeline
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.GenericModelWrapper import GenericModelWrapper

# Global configuration
global_config = Config()

class ElementFactory:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ElementFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, csv_path=None):
        """
        Initialize the ElementFactory with a CSV file containing element definitions.

        Parameters:
        - csv_path (str): Path to the CSV file with columns `element_type`, `description`, `element_path`, `module_path`, and `default_overrides`.
        """
        if csv_path is None:
            csv_path = global_config.element_factory_csv
        if not hasattr(self, 'elements'):
            self.elements = pd.read_csv(csv_path)

    def create(self, config={}):
        """
        Create a sklearn Pipeline based on the configuration.

        Parameters:
        - config (dict): Configuration dictionary with `element_type` and other parameters.

        Returns:
        - Pipeline: A scikit-learn pipeline instance.
        """
        element_type = config.get("element_type")
        if not element_type:
            raise ValueError("`element_type` must be provided in the config.")

        # Find the element in the CSV file
        element_row = self.elements[self.elements["element_type"] == element_type].iloc[0]
        if element_row.empty:
            raise ValueError(f"Element type '{element_type}' not found in the CSV file.")

        element_path = element_row["element_path"]

        if element_path.split('.')[-1] == "GenericModelWrapper":
            # Use GenericModelWrapper for non-standard models
            module_path = element_row["module_path"]
            default_overrides = element_row["default_overrides"]
            if pd.isnull(default_overrides):
                default_overrides = {}

            # Update config with module_path and default_overrides
            config["module_path"] = module_path
            config["default_overrides"] = default_overrides

            # Return the GenericModelWrapper
            return GenericModelWrapper(config)
        else:
            # Extract the element path and dynamically import the class
            module_name, class_name = element_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

            # Instantiate the class with the config
            element_instance = cls(config=config)

            # Return a pipeline
            return element_instance

    def create_pipe_line(self, config):
        element_type = config.get("element_type")
        return Pipeline([(element_type, self.create(config))])

# Example usage
# factory = ElementFactory()
# pipeline = factory.create(config={"element_type": "xgboost_classifier", "model_config": {"n_estimators": 200}})
