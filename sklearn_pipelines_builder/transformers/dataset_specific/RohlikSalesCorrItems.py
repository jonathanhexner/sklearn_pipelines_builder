import gc
import re
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.infrastructure.Config import Config

global_config = Config()

class RohlikSalesCorrItems(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        """
        Transformer to process sales data with missing holidays.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - calendar_file: Path to the calendar CSV file.
            - inventory_file: Path to the inventory CSV file.
        """

        self.corr_storage_path = os.path.join(global_config.output_folder, "corr_items.parquet")
        self.lag_storage_path = os.path.join(global_config.output_folder, "lag_items.parquet")


    @staticmethod
    def calc_corr(X):

        df_train_pivot_month_delta = pd.pivot_table(X,
                                                    values=['sales_sum_by_year_month_warehouse_name',
                                                            'sales_sum_by_year_month_warehouse_name_months_-1'],
                                                    index='year_month',
                                                    columns=['name', 'warehouse'], aggfunc="sum", fill_value=0)
        df_train_pivot_month_delta = df_train_pivot_month_delta.reset_index()
        df_train_pivot_month_delta.columns = ['__'.join(col) if col[1] != '' else col[0] for col in
                                              df_train_pivot_month_delta.columns]
        cols = list(set(df_train_pivot_month_delta.columns) - set(['year_month']))
        df_corr_delta1 = df_train_pivot_month_delta[cols].corr('spearman')
        return df_corr_delta1


    def fit(self, X_in: pd.DataFrame, y=None):
        X = X_in.copy(True)
        X['sales'] = y
        X_month_sales = X.groupby(['year_month', 'name', 'warehouse',
                                      'sales_sum_by_year_month_warehouse_name_months_-1'])['sales'].sum().reset_index(
            name='sales_sum_by_year_month_warehouse_name')

        df_corr = RohlikSalesCorrItems.calc_corr(X_month_sales)
        del X, X_month_sales
        gc.collect()

        lagged_cols = [col for col in df_corr.columns if re.search('months\_\-1', col)]
        non_lagged_cols = [col for col in df_corr.columns if col not in lagged_cols]
        corr_sub_matrix = df_corr.loc[non_lagged_cols, lagged_cols]
        # Find the best correlated lagged column for each non-lagged column
        best_lagged_match = corr_sub_matrix.idxmax(axis=1)
        best_lagged_corr = corr_sub_matrix.max(axis=1)
        extract_name_warehouse = lambda x: x.split("__")[1:]
        df_current = pd.DataFrame(columns=['name', 'warehouse'],
                                  data=list(map(extract_name_warehouse, list(best_lagged_match.keys()))))
        df_lag = pd.DataFrame(columns=['name-1_best_corr', 'warehouse-1_best_corr'],
                              data=list(map(extract_name_warehouse, list(best_lagged_match.values))))
        df_map = pd.concat([df_current, df_lag], axis=1)
        df_map['Correlation'] = best_lagged_corr.values

        # Get unique ('name', 'warehouse') pairs
        unique_pairs = X_in[['name', 'warehouse']].drop_duplicates()

        # Get unique year_month values
        unique_year_months = X_in[['year_month']].drop_duplicates()

        # Perform a cross join using merge (cartesian product)
        crossed_df = unique_pairs.merge(unique_year_months, how='cross')
        df_map = df_map.merge(crossed_df, on=['name', 'warehouse'], how='left')

        X_sum_sales = X_in.groupby(['name', 'warehouse', 'year_month'])['response_copy'].sum().reset_index(name='sum_sales')
        X_sum_sales['year_month'] = pd.to_datetime(X_sum_sales['year_month'], format='%Y-%m').dt.to_period('M')
        X_sum_sales['year_month'] = X_sum_sales['year_month'].astype(str)
        # X_sum_sales['year_month_lag-1'] = (X_sum_sales['year_month'] - 1).astype(str)
        crossed_df = crossed_df.merge(X_sum_sales, on=['year_month', 'name', 'warehouse'],
                                      how='left')
        crossed_df['sum_sales'] = crossed_df['sum_sales'].fillna(0)

         # pd.DateOffset(months=1)
        df_map['year_month'] = pd.to_datetime(df_map['year_month']).dt.to_period('M')
        df_map['year_month_lag-1'] = (df_map['year_month'] - 1).astype(str)
        crossed_df = crossed_df.rename(columns={'sum_sales': 'sales_correlated_item_lag-1',
                                   'name': 'name-1_best_corr',
                                   'warehouse': 'warehouse-1_best_corr',
                                   'year_month': 'year_month_lag-1'})
        X_lag = df_map.merge(crossed_df,
                              on=['name-1_best_corr', 'warehouse-1_best_corr', 'year_month_lag-1'], how='left')



        X_lag[['name', 'warehouse', 'year_month',
               'sales_correlated_item_lag-1', 'name-1_best_corr', 'warehouse-1_best_corr', 'Correlation']].to_parquet(self.lag_storage_path, index=False)
        return self


    def transform(self, X):
        """
        Transform the DataFrame by adding missing holiday information and processing name_base.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with `date` and `warehouse` columns.

        Returns:
        - pd.DataFrame: Transformed DataFrame with additional features.
        """
        X_lag = pd.read_parquet(self.lag_storage_path)
        # Merge general information
        merge_on = ['name', 'warehouse']
        columns_to_merge = [ 'name-1_best_corr', 'warehouse-1_best_corr', 'Correlation']
        X = X.merge(X_lag[columns_to_merge+merge_on].drop_duplicates(),
                    on=merge_on, how='left')
        X[['name-1_best_corr', 'warehouse-1_best_corr']] = \
            X[['name-1_best_corr', 'warehouse-1_best_corr']].fillna('MISSING')
        # Merge specific lags
        merge_on = ['name', 'warehouse', 'year_month']
        columns_to_merge = [ 'sales_correlated_item_lag-1']
        X['year_month'] = pd.to_datetime(X['year_month']).dt.to_period('M')

        X = X.merge(X_lag[columns_to_merge+merge_on].drop_duplicates().dropna(axis=0),
                    on=merge_on, how='left')
        X['sales_correlated_item_lag-1'] = X['sales_correlated_item_lag-1'].fillna(0)
        return X
