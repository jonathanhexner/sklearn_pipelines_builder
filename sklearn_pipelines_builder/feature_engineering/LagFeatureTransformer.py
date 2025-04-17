import os
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config

global_config = Config()
from sklearn_pipelines_builder.utils.logger import logger


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        """
        Transformer to perform multiple groupby aggregations, store results, and add lag features.

        Parameters:
        - config (dict): Configuration with the following keys:
            - aggregations: List of aggregation configurations. Each configuration is a dictionary with:
                - groupby_cols: List of columns to group by (e.g., ['date', 'warehouse']).
                - target_col: Column to aggregate (e.g., 'sales').
                - agg_func: Aggregation function to apply (e.g., 'sum', 'mean').
                - output_col: Name of the output column for the aggregated value.
            - storage_path: Path to store/load the Parquet file.
            - delay: Delay periods (e.g., {'days': 30}, {'years': 1}) for lagged features.
            - merge_on: List of columns to use for merging (e.g., ['date', 'warehouse']).
        """
        self.config = config or {}
        self.aggregations = self.config.get("aggregations", [])
        self.date_column = self.config.get("date_column", 'date')
        self.sales_regex = 'sales'
        self.price_regex = 'price'
        self.storage_path = os.path.join(global_config.output_folder, "aggregated_features.parquet")
        self.delays = self.config.get("delays", {"days": 30})  # Default delay of 30 days
        self.merge_on = self.config.get("merge_on", ["date", "warehouse"])
        self.aggregated_features_df = None

    def fit(self, X, y=None):
        """
        Perform aggregations and store results.

        Parameters:
        - X: pd.DataFrame, input DataFrame.

        Returns:
        - self: Fitted transformer.
        """
        X = X.copy()
        X[SingleContainer.response] = y


        # Perform each aggregation
        aggregated_results = []
        for agg in self.aggregations:
            groupby_cols = agg.get("groupby_cols", [])
            target_col = agg.get("target_col", "")
            agg_func = agg.get("agg_func", "sum")
            output_col = agg.get("output_col", f"{'_'.join(groupby_cols)}_{target_col}_{agg_func}")

            # Perform groupby and aggregation
            agg_df = (
                X.groupby(groupby_cols, as_index=False)
                .agg({target_col: agg_func})
                .rename(columns={target_col: output_col})
            )
            aggregated_results.append(agg_df)

        # Merge all aggregated results into one DataFrame
        self.aggregated_features_df = aggregated_results[0]
        for agg_df in aggregated_results[1:]:
            groupby_cols = agg.get("groupby_cols")
            merge_on = list(set(self.aggregated_features_df.columns) & set(agg_df.columns))
            self.aggregated_features_df = self.aggregated_features_df.merge(agg_df, on=merge_on, how="left")

        # Store aggregated features to a Parquet file
        self.aggregated_features_df.to_parquet(self.storage_path, index=False)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Add lagged features using stored aggregated results.

        Parameters:
        - X: pd.DataFrame, input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with lag features added.
        """
        # X = X.copy()

        # Load stored aggregated features
        aggregated_features = pd.read_parquet(self.storage_path)
        latest_date_column = self.date_column
        for delay_tup in self.delays:
            delay_date_column = self.date_column+f'_offset_{list(delay_tup.keys())[0]}_{list(delay_tup.values())[0]}'
            # Apply delay to the date column
            X[delay_date_column] = pd.to_datetime(X[self.date_column]) + pd.DateOffset(**delay_tup)
            aggregated_features = aggregated_features.rename(columns={latest_date_column: delay_date_column})
            latest_date_column = delay_date_column
            aggregated_features[delay_date_column] = pd.to_datetime(aggregated_features[delay_date_column])
            # aggregated_features[self.date_column] = pd.to_datetime(aggregated_features[self.date_column]) + \
            #                                         pd.DateOffset(**delay_tup)

            # Merge delayed aggregated features back to the input DataFrame
            common_cols = list(set(aggregated_features.columns) & set(X.columns))
            new_feature_cols = list(set(aggregated_features.columns)-set(common_cols))
            new_feature_cols_rename = list(map(lambda x: x+f'_{list(delay_tup.keys())[0]}_{list(delay_tup.values())[0]}',
                                               new_feature_cols))
            aggregated_features = aggregated_features.groupby(common_cols,  as_index=False).max()
            X = X.merge(
                aggregated_features.rename(columns=dict(zip(new_feature_cols, new_feature_cols_rename))),
                on=common_cols,
                how="left",
            )
            # Find the minimum available date in aggregated_features
            min_aggregated_date = aggregated_features[delay_date_column].min()
            max_delay = pd.DateOffset(**delay_tup)
            # Remove rows where X's date is before the first valid delayed date
            # logger.info("Drop date < %s rows %s", str(min_aggregated_date - max_delay),
            #             str(len(X[X[delay_date_column] < (min_aggregated_date - max_delay)])))

            # X = X[X[delay_date_column] >= (min_aggregated_date - max_delay)]

            sales_cols = [col for col in X.columns if re.search(self.sales_regex, col)]
            price_cols = [col for col in X.columns if re.search(self.price_regex, col)]
            X[sales_cols] = X[sales_cols].fillna(0)

            common_cols_wo_date = list(set(common_cols)-set([delay_date_column]))
            for price_col in price_cols:
                X[price_col] = X[price_col].fillna(0)
                X[price_col] = X[price_col].fillna(X.groupby(common_cols_wo_date)['sell_price_main'].transform('mean'))

        return X
