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
            - calendar_file: Path to the self.df_calendar CSV file.
            - inventory_file: Path to the inventory CSV file.
        """

        self.lemmatizer = WordNetLemmatizer()
        self.calendar_file = config.get('calendar_file')
        self.inventory_file = config.get('inventory_file')
        self.weights_file = config.get('weights_file')
        self.merge_weights = config.get("merge_weights", True)
        self.df_calendar = pd.read_csv(self.calendar_file)
        self.df_calendar['date'] = pd.to_datetime(self.df_calendar['date'])
        self.process_calendar()
        self.df_weights = pd.read_csv(self.weights_file)
        self.df_inventory = pd.read_csv(self.inventory_file)
        self.missing_holidays = self.create_missing_holidays()
        self.min_date = config.get('min_date')
        self.storage_path = os.path.join(global_config.output_folder, "rohlik_sales.parquet")
        self.stats_path = os.path.join(global_config.output_folder, "rohlik_stats.parquet")

        self._min_date = None

    @staticmethod
    def fe_date(df):
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['days_since_2020'] = (df['date'] - pd.to_datetime('2020-01-01')).dt.days.astype('int')
        df['day_of_year'] = df['date'].dt.dayofyear
        df['cos_day'] = np.cos(df['day_of_year'] * 2 * np.pi / 365)
        df['sin_day'] = np.sin(df['day_of_year'] * 2 * np.pi / 365)
        return df

    @staticmethod
    def fe_other(df):
        discount_cols = ['type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 'type_4_discount',
                         'type_5_discount', 'type_6_discount']
        df[discount_cols] = df[discount_cols].clip(0)
        df['max_discount'] = df[
            ['type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 'type_4_discount',
             'type_5_discount']].max(axis=1)

        # Given that we're using XGBoost, which is in theory invariant to monotonic transformations of features, this transformation in isolation doesn't really do anything. I mainly did it because it made the shap plot look more linear. However, I think it did make further feature engineering that used price more effective.
        df['sell_price_main'] = np.log(df['sell_price_main'])

        df['common_name'] = df['name'].apply(lambda x: x[:x.find('_')])
        df['CN_total_products'] = df.groupby(['date', 'warehouse', 'common_name'])['unique_id'].transform('nunique')
        df['CN_discount_avg'] = df.groupby(['date', 'warehouse', 'common_name'])['max_discount'].transform('mean')
        df['CN_WH'] = df['common_name'] + '_' + df['warehouse']
        df['name_num_warehouses'] = df.groupby(['date', 'name'])['unique_id'].transform('nunique')
        return df

    @staticmethod
    def fe_combined(df):
        df['num_sales_days_28D'] = pd.MultiIndex.from_frame(df[['unique_id', 'date']]).map(
            df.sort_values('date').groupby('unique_id').rolling(
                window='28D', on='date', closed='left')['date'].count().fillna(0))

        # This 'price_detrended' feature was one I found pretty late into the game, but I think it helped out a lot. I was trying to make a feature that captured whether an item was cheap or expensive relative to its usual price, which is what 'price_scaled' represents. What I found was that the prices of things generally increase over time. So I removed that time-based trend to construct price_detrended, and that proved very effective.
        mean_prices = df.groupby(df['unique_id'])['sell_price_main'].mean()
        std_prices = df.groupby(df['unique_id'])['sell_price_main'].std()
        df['price_scaled'] = np.where(df['unique_id'].map(std_prices) == 0, 0,
                                      (df['sell_price_main'] - df['unique_id'].map(mean_prices)) / df['unique_id'].map(
                                          std_prices))
        df['price_detrended'] = df['price_scaled'] - df.groupby(['days_since_2020', 'warehouse'])[
            'price_scaled'].transform('mean')
        df.drop('price_scaled', axis=1, inplace=True)

        warehouse_stats = df.groupby(['date', 'warehouse'])['total_orders'].median().rename(
            'med_total_orders').reset_index().sort_values('date')
        warehouse_stats['ewmean_orders_56'] = warehouse_stats.groupby('warehouse')['med_total_orders'].transform(
            lambda x: x.ewm(alpha=1 / 56).mean())
        df['mean_orders_14d'] = pd.MultiIndex.from_frame(df[['warehouse', 'date']]).map(
            warehouse_stats.groupby('warehouse').rolling(on='date', window='14D')['med_total_orders'].mean())
        df['ewmean_orders_56'] = pd.MultiIndex.from_frame(df[['warehouse', 'date']]).map(
            warehouse_stats.set_index(['warehouse', 'date'])['ewmean_orders_56'])

    def calc_train_cp(self, train):
        train_cp = train.groupby('unique_id')['date'].apply(
            lambda s: pd.date_range(s.min(), test.date.max())).explode().reset_index()

        train_cp = train_cp.merge(
            pd.concat([train[['unique_id', 'date', 'sales', 'warehouse', ]],
                       test[['unique_id', 'date', 'warehouse']]]),
            on=['unique_id', 'date'], how='left')
        train_cp = train_cp.merge(self.df_inventory, left_on='unique_id', right_index=True)
        train_cp['common_name'] = train_cp['name'].apply(lambda x: x[:x.find('_')])
        train_cp.sort_values('date', inplace=True)
        train_cp['last_sales_ema005'] = train_cp.groupby(['unique_id'])['sales'].transform(
            lambda x: x.shift(1).ewm(alpha=.005).mean()).fillna(0)
        train_cp['CN_sales_sum'] = train_cp.groupby(['common_name', 'warehouse', 'date'])[
            'last_sales_ema005'].transform('sum')

    def process_calendar(self):
        self.df_calendar.loc[self.df_calendar['holiday_name'].isna(), 'holiday'] = 0  # V3
        self.df_calendar['last_holiday_date'] = self.df_calendar['date']
        self.df_calendar['next_holiday_date'] = self.df_calendar['date']
        self.df_calendar.loc[self.df_calendar['holiday'] == 0, ['last_holiday_date', 'next_holiday_date']] = np.nan
        self.df_calendar['last_holiday_date'] = self.df_calendar.sort_values('date').groupby('warehouse')['last_holiday_date'].ffill()
        self.df_calendar['next_holiday_date'] = self.df_calendar.sort_values('date').groupby('warehouse')['next_holiday_date'].bfill()
        self.df_calendar['days_since_last_holiday'] = ((self.df_calendar['date'] - self.df_calendar['last_holiday_date']).dt.days)
        self.df_calendar['days_to_next_holiday'] = ((self.df_calendar['next_holiday_date'] - self.df_calendar['date']).dt.days)
        self.df_calendar['day_before_holiday'] = self.df_calendar['days_to_next_holiday'] == 1
        self.df_calendar['day_after_holiday'] = self.df_calendar['days_since_last_holiday'] == 1
        self.df_calendar.drop(['last_holiday_date', 'next_holiday_date'], axis=1, inplace=True)
        self.df_calendar.drop(['days_since_last_holiday', 'days_to_next_holiday'], axis=1, inplace=True)
        self.df_calendar.drop(['shops_closed', 'winter_school_holidays', 'school_holidays', 'holiday_name'], axis=1,
                      inplace=True)

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
        X = RohlikSalesTransformer.fe_date(X)
        X = RohlikSalesTransformer.fe_combined(X)
        X = RohlikSalesTransformer.fe_other(X)

        # Merge additional self.df_calendar and inventory data
        X = X.merge(self.df_calendar, on=['date', 'warehouse'], how='left')
        X = X.merge(self.df_inventory, on=['unique_id', 'warehouse'], how='left')
        if self.merge_weights:
            X = X.merge(self.df_weights, on='unique_id', how='left')


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




        X.loc[X['missing_holiday'] == 1, ['holiday', 'holiday_name']] = X.loc[
            X['missing_holiday'] == 1, ['missing_holiday', 'missing_holiday_name']]
        X.drop(columns=['missing_holiday', 'missing_holiday_name'], inplace=True)
        X['holiday_name'] = X['holiday_name'].fillna('None')
        # Process `name_base`
        X['name_base'] = X['name'].apply(lambda x: x.split('_')[0]).str.lower()
        X['name_base'] = X['name_base'].apply(lambda x: self.lemmatizer.lemmatize(x))
        X = self.calc_discounts(X)
        return X
