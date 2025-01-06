from sklearn.base import BaseEstimator, TransformerMixin

class BaseConfigurableTransformer(BaseEstimator, TransformerMixin):
    """
    A base transformer class that enforces the use of a configuration dictionary.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.model_config = self.config.get('model_config', {})

    def get_params(self, deep=True):
        """
        Override to include config parameters.
        """
        # Get base parameters and include the reconstructible 'config'
        params = super().get_params(deep=deep)
        params['config'] = self.config
        return params

    def set_params(self, **params):
        """
        Override to update config parameters.
        """
        config_keys = set(self.model_config.keys())
        for key, value in params.items():
            if key in config_keys:
                self.model_config[key] = value
            else:
                super().set_params(**{key: value})
        return self
