import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


class NormalizeAndPCA(BaseEstimator, TransformerMixin):
    def __init__(self, config={}):
        self.config = copy.deepcopy(config)
        self.features = config.get('features')  # List of features to transform
        self.n_components = config.get('n_components', 3)  # Number of PCA components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        # Normalize and fit PCA on specified features
        X_selected = X[self.features]
        X_normalized = self.scaler.fit_transform(X_selected)
        self.pca.fit(X_normalized)
        return self

    def transform(self, X):
        # Apply normalization and PCA
        X_copy = X.copy()
        X_selected = X_copy[self.features]
        X_normalized = self.scaler.transform(X_selected)
        X_pca = self.pca.transform(X_normalized)

        # Replace the original features with the PCA components
        pca_columns = [f"{self.features[0]}_pca_{i + 1}" for i in range(self.n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

        # Drop original features and add PCA-transformed components
        X_copy = X_copy.drop(columns=self.features)
        X_copy = pd.concat([X_copy, X_pca_df], axis=1)

        return X_copy
