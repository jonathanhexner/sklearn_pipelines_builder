import numpy as np
import pandas as pd
import copy
import importlib

class GenericModelWrapper:
    def __init__(self, config):
        self.config = config
        self.column = self.config.get("column")
        self.drop_na = config.get("drop_na", True)
        self.module_path = config.get("module_path")
        self.default_overrides = config.get("default_overrides", {})
        if isinstance(self.default_overrides, str):
            self.default_overrides = eval(self.default_overrides)
        if not self.module_path:
            raise ValueError("module_path must be specified in the config")

        # Dynamically load the model
        module_name, class_name = self.module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self.model_class = getattr(module, class_name)

        # Merge defaults with provided config
        self.model_config = copy.deepcopy({**self.default_overrides, **config.get("model_config", {})})
        self.model = self.model_class(**self.model_config)

    def fit(self, X, y):
        if self.drop_na:
            X = X[~pd.isnull(y)].reset_index(drop=True)
            y = y.dropna().reset_index(drop=True)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        if hasattr(self.model, 'transform'):
            Xt = self.model.transform(X)
            if isinstance(Xt, np.ndarray):
                columns = X.columns[self.model.get_support()]
                Xt = pd.DataFrame(columns=columns, data=Xt)
            return Xt
        return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
