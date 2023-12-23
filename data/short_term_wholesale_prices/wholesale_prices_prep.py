import os
import pandas as pd
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


def clean_wholesale_data() -> pd.DataFrame:

    # Get a list of all files in the directory
    all_files = os.listdir(os.path.dirname(__file__))
    # Use list comprehension to filter only the CSV files
    csv_files = [file for file in all_files if file.endswith('.csv')]

    dfs = []
    for filename in csv_files:
        dfs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), filename).replace("\\", "/")))

    for i in range(len(dfs)):
        # First split the first datetime column on delimiter
        dfs[i][["datetime_start", "datetime_end"]] = dfs[i]["MTU (CET/CEST)"].str.split(' - ', expand=True)
        dfs[i] = dfs[i].drop(["MTU (CET/CEST)", "datetime_end", "BZN|IE(SEM)", "Currency"], axis=1)
        dfs[i]["datetime_start"] = pd.to_datetime(dfs[i]["datetime_start"], format="%d.%m.%Y %H:%M", errors="coerce")
        # dfs[i].set_index("datetime_start", drop=True, inplace=True)

    # Concatenate the DataFrames in the list vertically
    result_df = pd.concat(dfs, ignore_index=False)

    # Resample and interpolate the missing rows
    result_df = result_df.drop_duplicates(subset="datetime_start")
    result_df.set_index("datetime_start", drop=True, inplace=True)
    result_df["Day-ahead Price [EUR/MWh]"] = result_df["Day-ahead Price [EUR/MWh]"].resample("1H").interpolate()

    # Slice to remove any dates after the end of the test period
    result_df = result_df.loc[(result_df.index >= pd.Timestamp(day=9, month=8, year=2020)) &
                              (result_df.index <= pd.Timestamp(day=6, month=6, year=2023))]

    # Resample to 30 min intervals to match with other data
    result_df = result_df.resample("30T").ffill()

    result_df.to_csv(os.path.join(os.path.dirname(__file__), "ready_for_use",
                                  "prepared_day_ahead_data.csv").replace("\\", "/"))

    return result_df


if __name__ == "__main__":

    df = clean_wholesale_data()

