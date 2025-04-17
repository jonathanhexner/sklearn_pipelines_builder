import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.infrastructure.Config import Config
global_config = Config()

class EWMA_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.alpha = config.get("alpha", 0.005)  # Default alpha = 0.1
        self.date_col = config.get("date_column", "date")
        self.ewma_col = config.get("ewma_column", "ewma_sales_ema005")
        self.target_column = config.get("target_column", "response_copy")
        self.ewma_file = os.path.join(global_config.output_folder, "train_data_ewma.parquet")
        self.stats_file = os.path.join(global_config.output_folder, "stats.parquet")
        self.keep_columns = config.get("keep_columns", ['unique_id', 'date', 'warehouse', 'name_base', 'response_copy'])
        self.keep_columns_transform = [col for col in self.keep_columns if col != 'response_copy' ]
        self.train_data_ = None
        self.last_train_date_ = None
        self.fitted_ = False
        self.inventory_file = config.get('inventory_file')
        self.sales_stats = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        X_df[self.date_col] = pd.to_datetime(X_df[self.date_col])
        X_df.sort_values(by=self.date_col, inplace=True)

        X_df[self.keep_columns].to_parquet(self.ewma_file, index=False)

        self.last_train_date_ = X_df[self.date_col].max()
        self.fitted_ = True
        df_sales_stats = X.groupby(['name_base', 'warehouse'])['response_copy'].agg([('sales_mean', 'mean'),
                                                                                     ('sales_std', 'std')]).reset_index()
        df_sales_stats.to_parquet(self.stats_file, index=False)
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Transformer has not been fitted yet.")

        X_df = pd.DataFrame(X).copy()
        X_df[self.date_col] = pd.to_datetime(X_df[self.date_col])
        # df_inventory = pd.read_csv(self.inventory_file)

        df_sales_stats = pd.read_parquet(self.stats_file)
        new_data = X_df.loc[X_df[self.date_col] > self.last_train_date_, self.keep_columns_transform]

        if new_data.empty:
            all_data = X_df.copy()
        else:
            train_data = pd.read_parquet(self.ewma_file)
            combined = pd.concat([train_data, new_data]).sort_values(by=self.date_col)
            all_data = combined.loc[new_data.index]

        all_data.sort_values(by=self.date_col, inplace=True)
        train_cp = all_data.groupby('unique_id')['date'].apply(
            lambda s: pd.date_range(all_data['date'].min(), all_data['date'].max())).explode().reset_index()

        train_cp = train_cp.merge(all_data[['unique_id', 'date', 'warehouse', 'name_base', self.target_column]],
                                  on=['date', 'unique_id'], how='left')

        train_cp['last_sales_ema005'] = train_cp[self.target_column].shift(1).ewm(alpha=self.alpha).mean().fillna(0)
        train_cp['CN_sales_sum'] = train_cp.groupby(['name_base', 'warehouse', 'date'])[
            'last_sales_ema005'].transform('sum')

        X = X.merge(train_cp[['last_sales_ema005','CN_sales_sum', 'unique_id', 'date']], on=['unique_id', 'date'],
                    how='left')
        X['last_sales_ema005'] = X['last_sales_ema005'].fillna(0)

        X = X.merge(df_sales_stats, on=['name_base', 'warehouse'], how='left')
        X['sales_mean'] = X['sales_mean'].fillna(0)
        X['sales_std'] = X['sales_std'].fillna(0)

        X['last_sales_zs'] = (X['last_sales_ema005'] - X['sales_mean'])/X['sales_std']
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
