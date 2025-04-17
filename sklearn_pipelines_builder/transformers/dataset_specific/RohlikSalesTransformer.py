import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.infrastructure.Config import Config

global_config = Config()

class RohlikSalesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        """
        Transformer to process sales data with missing holidays.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - calendar_file: Path to the calendar CSV file.
            - inventory_file: Path to the inventory CSV file.
        """

        self.lemmatizer = WordNetLemmatizer()
        self.calendar_file = config.get('calendar_file')
        self.inventory_file = config.get('inventory_file')
        self.weights_file = config.get('weights_file')
        self.merge_weights = config.get("merge_weights", True)
        self.date_coefficient = config.get('date_coefficient', None)

        self.df_calendar = pd.read_csv(self.calendar_file)
        self.df_calendar['date'] = pd.to_datetime(self.df_calendar['date'])
        self.df_weights = pd.read_csv(self.weights_file)
        self.df_inventory = pd.read_csv(self.inventory_file)
        self.missing_holidays = self.create_missing_holidays()
        self.min_date = config.get('min_date')
        self.storage_path = os.path.join(global_config.output_folder, "rohlik_sales.parquet")
        self.stats_path = os.path.join(global_config.output_folder, "rohlik_stats.parquet")

        self._min_date = None


    def create_missing_holidays(self):
        """
        Create a DataFrame of all missing holidays for all warehouses.

        Returns:
        - pd.DataFrame: DataFrame with `warehouse`, `date`, and `missing_holiday_name`.
        """
        holiday_definitions = {
            'Prague': [
                (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
                (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], 'Mother Day'),
            ],
            'Brno': [
                (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
                (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], 'Mother Day'),
            ],
            'Budapest': [],
            'Munich': [
                (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
                (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
            ],
            'Frankfurt': [
                (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
                (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
            ],
        }

        # Generate a DataFrame for all holidays
        holiday_data = []
        for city, holidays in holiday_definitions.items():
            for dates, holiday_name in holidays:
                for date in dates:
                    holiday_data.append((f"{city}", pd.to_datetime(date, format='%m/%d/%Y'), holiday_name))

        return pd.DataFrame(holiday_data, columns=['city', 'date', 'missing_holiday_name'])

    def fit(self, X_in, y=None):
        X = X_in.copy(True)
        X['sales'] = y
        X_first_sale = self.calc_first_sale_df(X,sales_col='sales', col_name='first_sale_date')
        self._min_date = X_in['date'].min()
        self._max_date = X_in['date'].max()
        self._days_from_max_min = ((X_in['date'] - self._max_date).dt.days).min()

        X_first_sale.drop_duplicates().to_parquet(self.storage_path, index=False)
        # x_stats = self.calc_stats(X_in)
        # x_stats.drop_duplicates().to_parquet(self.stats_path, index=False)

        """No fitting is necessary for this transformer."""
        return self

    def calc_stats(self, df):
        df_cp = df.groupby('unique_id')['date'].apply(
            lambda s: pd.date_range(self._min_date, df['date'].max())).explode().reset_index()
        df_cp = df_cp.merge(
            df[['unique_id', 'date', 'sales', 'warehouse', ]],
            on=['unique_id', 'date'], how='left')

        df_cp.sort_values('date', inplace=True)
        df_cp['last_sales_ema005'] = df_cp.groupby(['unique_id'])['sales'].transform(
            lambda x: x.shift(1).ewm(alpha=.005).mean()).fillna(0)
        df_cp['CN_sales_sum'] = df_cp.groupby(['common_name', 'warehouse', 'date'])['last_sales_ema005'].transform(
            'sum')
        return df_cp
        # all_data = all_data.merge(df_cp.set_index(['unique_id', 'date'])[[
        #     'last_sales_ema005', 'CN_sales_sum'
        # ]], left_on=['unique_id', 'date'], right_index=True, how='left')

    def calc_first_sale_df(self, X, sales_col='sales', col_name='first_sale_date'):
        X_first_sale = X[X[sales_col] > 0].groupby(["unique_id", "warehouse"])["date"].min().reset_index()
        X_first_sale.rename(columns={'date': col_name}, inplace=True)
        return X_first_sale

    def calc_discounts(self, df):
        df["total_discount"] = df['type_0_discount'] + df['type_0_discount'] + df['type_1_discount'] + df[
            'type_2_discount'] + df['type_3_discount'] + df['type_4_discount'] + df['type_5_discount'] + df[
                              'type_6_discount']
        df['total_orders_'] = df['total_orders'] / df['sell_price_main']
        df['total_orders_dic'] = df['total_orders_'] / df["total_discount"]
        df['total_orders_sell_price_main'] = df['sell_price_main'] / df["total_discount"]
        for i in range(7):
            df[f'total_orders{i}'] = df[f'type_{i}_discount'] / df["total_orders"]
            df[f'total_orders_sell_price_main_{i}'] = df[f'type_{i}_discount'] / df["total_orders_sell_price_main"]
            df[f'sell_price_main{i}'] = df[f'type_{i}_discount'] / df["sell_price_main"]
            df[f'sell_price_main_x_{i}'] = df[f'type_{i}_discount'] / (df["sell_price_main"] * df["total_orders"])
            df[f'total_orders_dic{i}'] = df[f'type_{i}_discount'] / df["total_orders_dic"]

            df[f'_total_orders{i}'] = df[f'type_{i}_discount'] * df["total_orders"]
            df[f'_total_orders_sell_price_main_{i}'] = df[f'type_{i}_discount'] * df["total_orders_sell_price_main"]
            df[f'_sell_price_main{i}'] = df[f'type_{i}_discount'] * df["sell_price_main"]
            df[f'_total_orders_dic{i}'] = df[f'type_{i}_discount'] * df["total_orders_dic"]
        return df

    def transform(self, X_in):
        """
        Transform the DataFrame by adding missing holiday information and processing name_base.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with `date` and `warehouse` columns.

        Returns:
        - pd.DataFrame: Transformed DataFrame with additional features.
        """
        df_first_sales = pd.read_parquet(self.storage_path)
        # df_stats = pd.read_parquet(self.stats_path)

        X = X_in.copy()
        X = X.merge(df_first_sales, on=['unique_id', 'warehouse'], how='left')
        X['first_sale_date'] = X['first_sale_date'].fillna(X['date'].max())
        if self.min_date is not None:
            logger.info("Previous min date %s, setting new threshold %s", str(X['date'].min()), str(self.min_date))
            num_drop_rows = len(X[pd.to_datetime(X['date'])<=pd.to_datetime(self.min_date)].reset_index(drop=True))
            logger.info("Dropping %s rows", str(num_drop_rows))
            X = X[pd.to_datetime(X['date'])>pd.to_datetime(self.min_date)].reset_index(drop=True)

        if 'availability' in X.columns:
            X.drop(columns=['availability'], inplace=True)

        # Ensure the date column is in datetime format
        X['date'] = pd.to_datetime(X['date'], errors='coerce')

        X['year'] = X['date'].dt.year
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day


        # Merge missing holidays onto the dataset
        X['city'] = X['warehouse'].apply(lambda x: x.split('_')[0])
        X = X.merge(self.missing_holidays, on=['city', 'date'], how='left')
        X['missing_holiday'] = X['missing_holiday_name'].notna().astype(int)


        # Merge additional calendar and inventory data
        X = X.merge(self.df_calendar, on=['date', 'warehouse'], how='left')
        X = X.merge(self.df_inventory, on=['unique_id', 'warehouse'], how='left')
        if self.merge_weights:
            X = X.merge(self.df_weights, on='unique_id', how='left')
            # Find the max date
            # Compute days since max date (negative for past dates)
        if self.date_coefficient is not None:
            X['days_from_max'] = (X['date'] - self._max_date).dt.days
            X['weight_days'] = np.exp(self.date_coefficient * X['days_from_max'])
            X['weight'] = X['weight'] * X['weight_days']


        X.loc[X['missing_holiday'] == 1, ['holiday', 'holiday_name']] = X.loc[
            X['missing_holiday'] == 1, ['missing_holiday', 'missing_holiday_name']]
        X.drop(columns=['missing_holiday', 'missing_holiday_name'], inplace=True)
        X['holiday_name'] = X['holiday_name'].fillna('None')
        # Process `name_base`
        X['name_base'] = X['name'].apply(lambda x: x.split('_')[0]).str.lower()
        X['name_base'] = X['name_base'].apply(lambda x: self.lemmatizer.lemmatize(x))
        X = self.calc_discounts(X)
        return X
