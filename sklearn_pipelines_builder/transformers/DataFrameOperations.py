import copy
import pandas as pd
from typing import List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.utils.logger import logger

from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer


class DataFrameOperations(BaseConfigurableTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = copy.deepcopy(config) or {}
        self.operations = self.config.get("operations", [])
        self.logger = logger
        self.operations_dict = {
            "cast": DataFrameOperations.cast,
            "query": DataFrameOperations.query,
            "drop": DataFrameOperations.drop,
            "drop_null_rows": DataFrameOperations.drop_null_rows,
            "column_wise_operation": DataFrameOperations.column_wise_operation,
            "expression": DataFrameOperations.expression_operation,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for op in self.operations:
            operation_name = op.get("operation_name")
            operation_config = op.get("operation_config")

            self.logger.info(f"Running operation: {operation_name} with config: {operation_config}")
            if operation_name not in self.operations_dict:
                raise ValueError(f"Unsupported operation: {operation_name}")

            X = self.operations_dict[operation_name](operation_config, X)
        return X

    # ========== Operation Methods ==========

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
    def cast(cast_config: Dict, X: pd.DataFrame):
        return X.astype(cast_config)

    @staticmethod
    def column_wise_operation(operation_config: Dict, X: pd.DataFrame):
        X = X.copy()
        columns = operation_config.get("columns", [])
        operation = operation_config.get("operation")
        output_column = operation_config.get("output_column")

        if not (columns and operation and output_column):
            raise ValueError("column_wise_operation requires 'columns', 'operation', and 'output_column'")

        X[output_column] = X[columns].apply(operation, axis=1)
        return X

    @staticmethod
    def expression_operation(operation_config: Dict, X: pd.DataFrame):
        X = X.copy()
        output_column = operation_config.get("output_column")
        expression = operation_config.get("expression")

        if not output_column or not expression:
            raise ValueError("Both 'output_column' and 'expression' must be provided.")

        try:
            X[output_column] = eval(
                expression,
                {"__builtins__": {}, "X": X},
                {}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate expression '{expression}': {e}")

        return X

