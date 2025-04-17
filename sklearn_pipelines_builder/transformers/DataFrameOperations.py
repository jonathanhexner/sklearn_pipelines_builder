import copy
import pandas as pd
from typing import List

from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer


class DataFrameOperations(BaseConfigurableTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = copy.deepcopy(config)
        self.operations = config.get('operations', {})
        self.operations_dict = {'cast': DataFrameOperations.cast,
                                'query': DataFrameOperations.query,
                                'drop': DataFrameOperations.drop,
                                'drop_null_rows': DataFrameOperations.drop_null_rows,
                                'column_wise_operation': DataFrameOperations.column_wise_operation}

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        return self


    @staticmethod
    def query(query: str, X: pd.DataFrame):
        return X.query(query)

    @staticmethod
    def drop(columns: List, X: pd.DataFrame):
        return X.drop(columns=columns)

    @staticmethod
    def drop_null_rows(columns: List, X: pd.DataFrame):
        return X.dropna(subset=columns, axis=0)

    @staticmethod
    def cast(cast_config: {}, X: pd.DataFrame):
        return X.astype(cast_config)

    @staticmethod
    def column_wise_operation(operation_config, X: pd.DataFrame):
        columns = operation_config.get('columns', [])
        operation = operation_config.get('operation', '')
        output_column = operation_config.get('output_column', '')
        X[output_column] = X[columns].apply(operation, axis=1)
        return X

    def run_operation(self, operation, operation_config, X):
        return self.operations_dict[operation](operation_config, X)


    def transform(self, X):
        # Apply scaler only to numeric columns, leave others unchanged
        for operation in self.operations:
            operation_name = operation.get('operation_name')
            operation_config = operation.get('operation_config')
            print(operation_name, operation_config)

            X = self.run_operation(operation_name, operation_config, X)
        return X
