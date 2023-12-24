import os
import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


def clean_wholesale_data(filename, test_or_train: str) -> pd.DataFrame:

    day_ahead_file_path = os.path.join(os.path.dirname(__file__), filename).replace("\\", "/")
    day_ahead_df = pd.read_csv(day_ahead_file_path)

    day_ahead_df["Date Time"] = pd.to_datetime(day_ahead_df["Date Time"], errors='coerce', format="%d/%m/%Y %H:%M")

    day_ahead_df.set_index("Date Time", drop=True, inplace=True)

    day_ahead_df["Value"] = day_ahead_df["Value"].replace('-', np.nan)
    day_ahead_df["Value"] = day_ahead_df["Value"].astype(float)
    day_ahead_df["Value"] = day_ahead_df["Value"].interpolate()

    # Resample to hh data
    day_ahead_df = day_ahead_df.resample("30T").ffill()

    if test_or_train == "train":
        day_ahead_df = day_ahead_df[
            day_ahead_df.index <= pd.Timestamp(day=10, month=11, year=2022, hour=23, minute=30)]
    else:
        day_ahead_df = day_ahead_df[
            (day_ahead_df.index > pd.Timestamp(day=10, month=11, year=2022, hour=23, minute=30)) &
            (day_ahead_df.index <= pd.Timestamp(day=5, month=6, year=2023, hour=23, minute=30))]

    day_ahead_df.to_csv(os.path.join(os.path.dirname(__file__), "ready_for_use",
                                     f"prepared_{test_or_train}_day_ahead_data.csv").replace("\\", "/"))

    return day_ahead_df


if __name__ == "__main__":

    train_df = clean_wholesale_data(filename="nord_pool_day_ahead_prices.csv", test_or_train="train")
    test_df = clean_wholesale_data(filename="nord_pool_day_ahead_prices.csv", test_or_train="test")

