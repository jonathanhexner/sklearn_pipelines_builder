import yaml
import os

class Config:
    _instance = None  # Class attribute to hold the singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
        return cls._instance


    def load_config(self, config_path, config_override):
        """Load the YAML config file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")

        with open(config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        if config_override is not None:
            self._config.update(config_override)
        print(self._config['output_folder'])

    def get(self, key, default=None)->{}:
        """Retrieve a configuration value, with optional default."""
        return self._config.get(key, default)

    @property
    def output_folder(self)->str:
        return self._config.get('output_folder')

    @property
    def imputation_config(self)->{}:
        return self._config.get('imputation_config')

    @property
    def scoring(self):
        return self._config.get('scoring')

    @property
    def hyper_parameter_optimization(self)->{}:
        return self._config.get('optimization')

    @property
    def run_name(self):
        return self._config.get('run_name')

    @property
    def element_factory_csv(self):
        current_dir = os.path.dirname(__file__)
        return self._config.get('element_factory_csv', os.path.join(current_dir, 'element_factory_spec.csv'))
