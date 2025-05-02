import os
from calendar import calendar

import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.infrastructure.Config import Config

global_config = Config()

class RohlikSales20_75(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        """
        Transformer to process sales data with missing holidays.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - calendar_file: Path to the self.df_calendar CSV file.
            - inventory_file: Path to the inventory CSV file.
        """

        self.lemmatizer = WordNetLemmatizer()
        self.folder = config.get('folder')
        self.calendar_file = config.get('calendar_file')
        self.inventory_file = config.get('inventory_file')
        self.weights_file = config.get('weights_file')
        self.merge_weights = config.get("merge_weights", True)
        if self.folder:
            self.calendar_file = os.path.join(self.folder, self.calendar_file)
            self.inventory_file = os.path.join(self.folder, self.inventory_file)
            self.weights_file = os.path.join(self.folder, self.weights_file)
        self.df_calendar = pd.read_csv(self.calendar_file)
        self.df_calendar['date'] = pd.to_datetime(self.df_calendar['date'])
        self.process_holidays()
        self.df_weights = pd.read_csv(self.weights_file)
        self.df_inventory = pd.read_csv(self.inventory_file)
        # self.missing_holidays = self.create_missing_holidays()
        self.min_date = config.get('min_date')
        self.storage_path = os.path.join(global_config.output_folder, "rohlik_sales.parquet")
        self.stats_path = os.path.join(global_config.output_folder, "rohlik_stats.parquet")

        self._min_date = None

    def process_holidays(self):
        czech_holiday = [
            (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),  # loss
            (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),  # loss
        ]
        brno_holiday = [
            (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),  # loss
            (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),  # loss
        ]

        budapest_holidays = []
        munich_holidays = [
            (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),  # loss
            (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),  # loss
        ]

        frank_holidays = [
            (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),  # loss
            (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),  # loss
        ]

        def fill_loss_holidays(df_fill, warehouses, holidays):
            df = df_fill.copy()
            for item in holidays:
                dates, holiday_name = item
                generated_dates = [datetime.datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
                for generated_date in generated_dates:
                    df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
                    df.loc[(df['warehouse'].isin(warehouses)) & (
                                df['date'] == generated_date), 'holiday_name'] = holiday_name
            return df

        self.df_calendar = fill_loss_holidays(df_fill=self.df_calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'],
                                      holidays=czech_holiday)
        self.df_calendar = fill_loss_holidays(df_fill=self.df_calendar, warehouses=['Brno_1'], holidays=brno_holiday)
        self.df_calendar = fill_loss_holidays(df_fill=self.df_calendar, warehouses=['Munich_1'], holidays=munich_holidays)
        self.df_calendar = fill_loss_holidays(df_fill=self.df_calendar, warehouses=['Frankfurt_1'], holidays=frank_holidays)
        self.df_calendar = fill_loss_holidays(df_fill=self.df_calendar, warehouses=['Budapest_1'], holidays=budapest_holidays)


    def process_calendars(self):
        calendar_dfs = dict(
            Frankfurt_1 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Frankfurt_1"'),
            Prague_2 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_2"'),
            Brno_1 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Brno_1"'),
            Munich_1 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Munich_1"'),
            Prague_3 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_3"'),
            Prague_1 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_1"'),
            Budapest_1 = self.df_calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Budapest_1"'))

        def process_calendar(df):
            df = df.sort_values('date').reset_index(drop=True)
            df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
            df['next_holiday_date'] = df['next_holiday_date'].bfill()
            df['days_to_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
            df.drop(columns=['next_holiday_date'], inplace=True)
            df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
            df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
            df['days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
            df.drop(columns=['next_shops_closed_date'], inplace=True)
            df['day_after_closing'] = (
                    (df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)
            ).astype(int)

            df['long_weekend'] = (
                    (df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)
            ).astype(int)

            df['weekday'] = df['date'].dt.weekday
            return df

        processed_dfs = [process_calendar(df) for df in calendar_dfs.values()]
        self.df_calendar = pd.concat(processed_dfs).sort_values('date').reset_index(drop=True)


    @staticmethod
    def fe_date(df):
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['dayofyear'] = df['date'].dt.dayofyear
        df['is_month_start'] = df['date'].dt.is_month_start
        df['is_month_end'] = df['date'].dt.is_month_end
        df['quarter'] = df['date'].dt.quarter
        return df

    @staticmethod
    def calc_discounts(df):
        df["total_dic"] = df['type_0_discount'] + df['type_0_discount'] + df['type_1_discount'] + df[
            'type_2_discount'] + df['type_3_discount'] + df['type_4_discount'] + df['type_5_discount'] + df[
                              'type_6_discount']
        df['total_orders_'] = df['total_orders'] / df['sell_price_main']
        df['total_orders_dic'] = df['total_orders_'] / df["total_dic"]
        df['total_orders_sell_price_main'] = df['sell_price_main'] / df["total_dic"]
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

    def fit(self, X_in, y=None):
        X = X_in.copy(True)
        X['sales'] = y
        # X_first_sale = self.calc_first_sale_df(X,sales_col='sales', col_name='first_sale_date')
        self._min_date = X_in['date'].min()
        # X_first_sale.drop_duplicates().to_parquet(self.storage_path, index=False)
        # x_stats = self.calc_stats(X_in)
        # x_stats.drop_duplicates().to_parquet(self.stats_path, index=False)

        """No fitting is necessary for this transformer."""
        return self



    def transform(self, X_in):
        """
        Transform the DataFrame by adding missing holiday information and processing name_base.

        Parameters:
        - X (pd.DataFrame): Input DataFrame with `date` and `warehouse` columns.

        Returns:
        - pd.DataFrame: Transformed DataFrame with additional features.
        """


        # df_first_sales = pd.read_parquet(self.storage_path)
        # df_stats = pd.read_parquet(self.stats_path)

        X = X_in.copy()
        self.process_holidays()
        self.process_calendars()

        # Merge additional self.df_calendar and inventory data
        X['date'] = pd.to_datetime(X['date'])
        X = X.merge(self.df_calendar, on=['date', 'warehouse'], how='left')
        X = X.merge(self.df_inventory, on=['unique_id', 'warehouse'], how='left')
        X = RohlikSales20_75.fe_date(X)
        X = RohlikSales20_75.calc_discounts(X)
        cols = X.columns
        obj_cols = list(cols[X.dtypes == 'object'])
        for obj_col in obj_cols:
            X[obj_col] = X[obj_col].fillna('0')
        other_cols = list(set(X.columns)-set(obj_cols))
        for other_col in other_cols:
            X[other_col] = X[other_col].fillna(0)

        if self.merge_weights:
            X = X.merge(self.df_weights, on='unique_id', how='left')

        # Process `name_base`
        # X['name_base'] = X['name'].apply(lambda x: x.split('_')[0]).str.lower()
        # X['name_base'] = X['name_base'].apply(lambda x: self.lemmatizer.lemmatize(x))
        X['date'] = X['date'].astype('int64')

        categorical_columns = ['unique_id'] + list(X_in.select_dtypes("object").columns)

        for col in categorical_columns:
            X[col] = X[col].astype('category')
        return X
