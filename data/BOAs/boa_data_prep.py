import os
import pandas as pd
from pprint import pprint

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


def clean_boa_data(filename: str) -> pd.DataFrame:

    boa_file_path = os.path.join(os.path.dirname(__file__), filename).replace("\\", "/")
    boa_df = pd.read_csv(boa_file_path)

    return boa_df


if __name__ == "__main__":
    boa_df = clean_boa_data("BOAs.csv")

