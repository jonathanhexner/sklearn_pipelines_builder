import copy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.utils.basic_utils import remove_from_list


class DropFeature(BaseEstimator, TransformerMixin):
    def __init__(self, config={}):
        self.config = copy.deepcopy(config)
        self.columns_to_drop = config.get('columns')
        if type(self.columns_to_drop) == str:
            self.columns_to_drop = self.columns_to_drop.split(",")
        SingleContainer.columns_to_drop += self.columns_to_drop
        SingleContainer.string_features = remove_from_list(SingleContainer.string_features, self.columns_to_drop)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        logger.info('Dropping columns %s', self.columns_to_drop)
        return X.drop(columns=self.columns_to_drop)
