"""
This module combines all the df into a single dfframe ready for passing into an ML model
"""
import pandas as pd
import numpy as np
import holidays
from calendar import monthrange
import os
import logging
from typing import Tuple
import re

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class PrepareModel:
    def __init__(self, test_or_train: str):

        self.test_or_train = test_or_train

        # Start off with the df provided on Kaggle
        base_df_path = os.path.join(os.path.dirname(__file__), "kaggle_data/ready_for_use",
                                    f"prepared_{test_or_train}_data.csv").replace("\\", "/")
        frequency_df_path = os.path.join(os.path.dirname(__file__), "system_frequency/ready_for_use",
                                         f"prepared_{test_or_train}_hh_freq.csv").replace("\\", "/")
        gen_data_path = os.path.join(os.path.dirname(__file__), "B1610_Actual_Generation.csv",
                                     "B1610_Actual_Generation.csv").replace("\\", "/")

        self.base_df = pd.read_csv(base_df_path, index_col=0)
        self.freq_df = pd.read_csv(frequency_df_path)
        self.gen_df = pd.read_csv(gen_data_path)

        # Removing special json characters that throw errors as heading names when passing into the ML models
        self.gen_df = self.gen_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        self.base_df["UTC_Settlement_DateTime"] = pd.to_datetime(self.base_df["UTC_Settlement_DateTime"],
                                                                 format="%m/%d/%Y %H:%M")

        try:
            self.freq_df["dtm"] = pd.to_datetime(self.freq_df["dtm"], format="%Y-%m-%d %H:%M:%S")
        except KeyError as e:
            logging.error(f"Check the datetime column heading for the frequency data \n {e}")
        try:
            self.gen_df["local_datetime"] = pd.to_datetime(self.gen_df["local_datetime"], format="%d/%m/%Y %H:%M")
        except KeyError as e:
            logging.error(f"Check the datetime column heading for the Generation data \n {e}")

    def merge_dataframes(self, save_merged_df_as_csv: bool) -> pd.DataFrame:

        # Concat the two dataframes on the date
        merged_df = pd.merge(self.base_df, self.freq_df, left_on="UTC_Settlement_DateTime", right_on="dtm", how="left")

        # Add the generation data to the dataframe
        merged_df = pd.merge(merged_df, self.gen_df, left_on="UTC_Settlement_DateTime", right_on="local_datetime",
                             how="left")

        # Drop the extra dtm datetime cols
        merged_df.drop(["dtm", "local_datetime"], axis=1, inplace=True)

        # Save the final merged df
        if save_merged_df_as_csv:
            merged_df.to_csv(os.path.join(os.path.dirname(__file__),
                                          f"prepared_combined_{self.test_or_train}_df.csv").replace("\\", "/"))

        return merged_df

    @staticmethod
    def add_additional_datetime_features(df: pd.DataFrame, datetime_col: str):
        df['year'] = df[f'{datetime_col}'].dt.year
        df['season'] = df[f'{datetime_col}'].dt.month % 12 // 3 + 1
        df['is_winter'] = [1 if val == 4 in df['season'] else 0 for val in df['season']]
        df['month'] = df[f'{datetime_col}'].dt.month
        df['week_of_year'] = df[f'{datetime_col}'].dt.isocalendar().week.astype('int')
        df['day'] = df[f'{datetime_col}'].dt.day
        df['dayofweek'] = df[f'{datetime_col}'].dt.dayofweek
        df['hour'] = df[f'{datetime_col}'].dt.hour
        df['minute'] = df[f'{datetime_col}'].dt.minute
        df["is_wknd"] = np.where(df[f'{datetime_col}'].dt.weekday >= 5, 1, 0)
        df["is_working_hr"] = np.where(((df['hour'] >= 8) & (df['hour'] <= 19)), 1, 0)
        df["is_lunch_hr"] = np.where(df['hour'] == 13, 1, 0)

        uk_holidays = holidays.country_holidays(country="UK")
        df['Holiday'] = [1 if str(val).split()[0] in uk_holidays else 0 for val in df[f'{datetime_col}']]

        # 2D time conversion
        df['days_in_month'] = monthrange(df['year'].all(), df['month'].all())[1]
        df['hourmin'] = df['hour'] + (df['minute'] / 60)
        df['hour_x'] = np.sin((360 / 24) * df['hourmin'])
        df['hour_y'] = np.cos((360 / 24) * df['hourmin'])
        df['day_x'] = np.sin((360 / df['days_in_month']) * df['day'])
        df['day_y'] = np.cos((360 / df['days_in_month']) * df['day'])
        df['month_x'] = np.sin((360 / 12) * df['month'])
        df['month_y'] = np.cos((360 / 12) * df['month'])

        df.drop(['days_in_month'], axis=1, inplace=True)

        # Drop the rows that now contain na values
        df.dropna(axis=0, inplace=True)

        return df

    @staticmethod
    def add_additional_weather_features(df: pd.DataFrame, weather_cols: list[str]):

        for col in weather_cols:
            # Get the temperature 24 hour into the future as another feature
            df[f'{col}_+24h'] = df[col].shift(periods=-48)
            # Get the temperature 12 hour into the future as another feature
            df[f'{col}_+12h'] = df[col].shift(periods=-24)
            # Get the temperature 1 hour into the future as another feature
            df[f'{col}_+1h'] = df[col].shift(periods=-2)
            # Get the temperature 30 mins into the future as another feature
            df[f'{col}_+30m'] = df[col].shift(periods=-1)

            # Get the average temperature over the past day
            df[f'{col}_daily_avg_-24h'] = df[col].rolling(48, min_periods=48).mean()
            # Get the average temperature over the next day
            df[f'{col}_daily_avg_+24h'] = df[col].shift(periods=-48).rolling(48, 48).mean()
            # Get the max temperature over the next day
            df[f'{col}_daily_max_+24h'] = df[col].shift(periods=-48).rolling(48, 48).max()
            # Get the min temperature over the next day
            df[f'{col}_daily_min_+24h'] = df[col].shift(periods=-48).rolling(48, 48).min()

        # Subtract row i from row i - 1
        # diff = np.diff(df[[f'{weather_cols}']], n=1)
        # df[[f'{weather_cols}_diff1']] = np.append([0], diff)

        # Drop the rows that now contain na values
        df.dropna(axis=0, inplace=True)

        return df


if __name__ == "__main__":
    pass
