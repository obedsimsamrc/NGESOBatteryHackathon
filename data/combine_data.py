"""
This module combines all the data into a single dataframe ready for passing into an ML model
"""
import pandas as pd
import os
import logging
from typing import Tuple

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def combine_data(test_or_train: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Start off with the data provided on Kaggle
    base_data_path = os.path.join(os.path.dirname(__file__), "kaggle_data/ready_for_use",
                                  f"prepared_{test_or_train}_data.csv").replace("\\", "/")
    frequency_data_path = os.path.join(os.path.dirname(__file__), "system_frequency/ready_for_use",
                                       f"prepared_{test_or_train}_hh_freq.csv").replace("\\", "/")

    base_df = pd.read_csv(base_data_path, index_col=0)
    base_df["UTC_Settlement_DateTime"] = pd.to_datetime(base_df["UTC_Settlement_DateTime"],
                                                        format="mixed", dayfirst=True, errors="coerce")

    freq_df = pd.read_csv(frequency_data_path)
    freq_df["dtm"] = pd.to_datetime(freq_df["dtm"], format="%Y-%m-%d %H:%M:%S")

    # Concat the two dataframes on the date
    merged_df = pd.merge(base_df, freq_df, left_on="UTC_Settlement_DateTime", right_on="dtm", how="left")

    # Save the final merged df
    merged_df.to_csv(os.path.join(os.path.dirname(__file__),
                                  f"prepared_combined_{test_or_train}_data.csv").replace("\\", "/"))

    return base_df, merged_df


if __name__ == "__main__":
    base_test_df, merged_test_df = combine_data("test")
    base_train_df, merged_train_df = combine_data("train")


