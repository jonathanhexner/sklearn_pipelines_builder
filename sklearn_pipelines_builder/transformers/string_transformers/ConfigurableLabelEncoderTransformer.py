import pandas as pd
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn_pipelines_builder.SingletonContainer import SingleContainer


class ConfigurableLabelEncoderTransformer(BaseConfigurableTransformer):
    def __init__(self, config):
        """
        Initialize the transformer with a configuration dictionary.

        Parameters:
        - config: A dictionary containing configuration parameters, including:
          - "string_columns": A list of columns to apply the label encoder to.
        """
        self.config = config
        self.string_columns = self.config.get("string_columns", None)
        self.label_encoder = OrdinalEncoder()
        self.column_transformer = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fit the label encoder to the string columns. If `string_columns` is not provided in the config,
        it reads them from the class attribute `SingletonContainer`.

        Parameters:
        - X: Input DataFrame
        - y: (Optional) Target variable; not used.

        Returns:
        - self
        """
        # Determine string_columns during fit
        if self.string_columns is None:
            self.string_columns = [col for col in X.columns if col in SingleContainer.string_features]
        self.feature_names_in_ = X.columns.to_list()

        if self.string_columns is None:
            raise ValueError("No string_columns defined in the config or SingletonContainer.")

        # Create the ColumnTransformer
        self.column_transformer = ColumnTransformer(
            [('label_encoder', self.label_encoder, self.string_columns)],
            remainder='passthrough'
        )

        # Fit the ColumnTransformer
        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        """
        Transform the input data by applying the label encoder to the string columns.

        Parameters:
        - X: Input DataFrame

        Returns:
        - Transformed data
        """
        if self.column_transformer is None:
            raise ValueError("The transformer has not been fitted yet.")
        feature_names = self.column_transformer.get_feature_names_out(input_features=self.feature_names_in_)
        feature_names = [col.split("__")[-1] for col in feature_names]
        return pd.DataFrame(data=self.column_transformer.transform(X), columns=feature_names)
