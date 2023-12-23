import os
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


def clean_dx_data(filename: str) -> pd.DataFrame:

    dx_file_path = os.path.join(os.path.dirname(__file__), filename).replace("\\", "/")
    dx_df = pd.read_csv(dx_file_path)

    # Remove unnecessary cols
    cols_to_remove = ["service_type", "auction_id", "efa_date", "delivery_end"]
    dx_df.drop(columns=cols_to_remove, inplace=True)

    # Convert the datetime cols to the correct type
    dx_df["delivery_start"] = pd.to_datetime(dx_df["delivery_start"], errors='coerce', format="%Y-%m-%dT%H:%M:%S.%f")
    dx_df.set_index("delivery_start", drop=True, inplace=True)

    dx_df.index = dx_df.index.tz_localize(None)  # Remove timezone information

    # Create a dataframe for each service
    dcl_df = dx_df[dx_df["service"] == "DCL"].resample('1H').ffill()
    dch_df = dx_df[dx_df["service"] == "DCH"].resample('1H').ffill()
    dml_df = dx_df[dx_df["service"] == "DML"].resample('1H').ffill()
    dmh_df = dx_df[dx_df["service"] == "DMH"].resample('1H').ffill()
    drl_df = dx_df[dx_df["service"] == "DRL"].resample('1H').ffill()
    drh_df = dx_df[dx_df["service"] == "DRH"].resample('1H').ffill()

    # Merge into a single dataframe
    dx_df = (
        pd.merge(dcl_df, dch_df, left_index=True, right_index=True, suffixes=("_dcl", "_dch"), how="left")
        .merge(dml_df.add_suffix("_dml"), left_index=True, right_index=True, how="left")
        .merge(dmh_df.add_suffix("_dmh"), left_index=True, right_index=True, how="left")
        .merge(drl_df.add_suffix("_drl"), left_index=True, right_index=True, how="left")
        .merge(drh_df.add_suffix("_drh"), left_index=True, right_index=True, how="left")
    )

    dx_df.drop(dx_df.columns[dx_df.columns.str.contains("service")], axis=1, inplace=True)
    dx_df.drop(dx_df.columns[dx_df.columns.str.contains("efa")], axis=1, inplace=True)

    # Fill the na volumes and clearing prices with zero
    dx_df.fillna(0, inplace=True)

    dx_df.to_csv(os.path.join(os.path.dirname(__file__), "ready_for_use", "prepared_dx_data.csv").replace("\\", "/"))

    return dx_df


if __name__ == "__main__":
    dx_df = clean_dx_data("dx_results.csv")

