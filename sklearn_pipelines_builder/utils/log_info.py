import copy
import mlflow
from sklearn_pipelines_builder.SingletonContainer import SingleContainer

def log_info(log_suffix=''):
    # with mlflow.start_run(SingleContainer.mlflow_run_id):
        # with mlflow.start_run(nested=True):
    mlflow.log_param(f'string_features{log_suffix}', copy.deepcopy(SingleContainer.string_features))
    mlflow.log_param(f'numeric_features{log_suffix}', copy.deepcopy(SingleContainer.numeric_features))
    mlflow.log_param(f'null_string_features{log_suffix}', copy.deepcopy(SingleContainer.null_string_features))
    mlflow.log_param(f'null_numeric_features{log_suffix}', copy.deepcopy(SingleContainer.null_numeric_features))
