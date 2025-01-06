from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
import pandas as pd
from scipy.sparse import issparse

class ConcatenateEncodedFeatures(BaseConfigurableTransformer):
    def __init__(self, original_columns=None, encoded_columns=None):
        """
        Initialize the transformer with the original and encoded column names.
        """
        self.original_columns = original_columns
        self.encoded_columns = encoded_columns

    def fit(self, X, y=None):
        # Ensure column names are set
        if not self.original_columns or not self.encoded_columns:
            raise ValueError("Both `original_columns` and `encoded_columns` must be provided.")
        return self

    def transform(self, X):
        """
        Concatenate the original and encoded features.
        """
        if issparse(X):
            X = X.toarray()  # Convert sparse matrix to dense array

        # Split original and encoded features
        df_encoded = pd.DataFrame(X[:, :len(self.encoded_columns)], columns=self.encoded_columns)
        df_original = pd.DataFrame(X[:, len(self.encoded_columns):], columns=self.original_columns)

        # Concatenate
        return pd.concat([df_original, df_encoded], axis=1)
