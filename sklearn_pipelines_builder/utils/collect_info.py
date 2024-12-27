import pandas as pd
import copy
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.utils.basic_utils import remove_from_list


config = Config()


def collect_info(X_all: pd.DataFrame):
    response = config.get('response')
    original_columns = list(X_all.columns)
    all_features = list(set(original_columns)-set([response]))
    string_features = list(X_all.select_dtypes(include='object').columns)
    numeric_features = list(set(X_all.columns) - set(string_features))
    print(string_features)
    # In[144]:
    mean_null_numeric = X_all[numeric_features].isnull().mean()
    null_numeric = list(mean_null_numeric[mean_null_numeric > 0].keys())
    print(null_numeric)
    # In[145]:
    mean_null_string = X_all[string_features].isnull().mean()
    null_string = list(mean_null_string[mean_null_string > 0].keys())
    print(null_string)


    SingleContainer.string_features = remove_from_list(string_features, SingleContainer.columns_to_drop)
    SingleContainer.numeric_features = remove_from_list(numeric_features, SingleContainer.columns_to_drop)
    SingleContainer.mean_null_numeric_features = mean_null_numeric
    SingleContainer.mean_null_string_features = mean_null_string
    SingleContainer.null_string_features = remove_from_list(null_string, SingleContainer.columns_to_drop)
    SingleContainer.null_numeric_features = remove_from_list(null_numeric, SingleContainer.columns_to_drop)
    SingleContainer.original_columns = copy.deepcopy(original_columns)
    SingleContainer.all_features = copy.deepcopy(all_features)
    SingleContainer.response = response
