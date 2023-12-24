from ElexonDataPortal import api
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import os

client = api.Client('jijs8vduwlhp8me')

dfs = []

for x in range(1, 208 - 21 - 7):
    print(f"\n Running day {x} of {208 - 21 - 7}")
    initial_date = '2022-12-11'
    start_date = datetime.strptime(initial_date, '%Y-%m-%d')
    start_date_1 = start_date + timedelta(days=1 * x)
    end_date_1 = start_date_1 + timedelta(days=1)
    df_B1610: pd.DataFrame = client.get_B1610(start_date_1, end_date_1)
    dfs.append(df_B1610)

df_all = pd.concat(dfs, ignore_index=True)

df_all["quantity"] = df_all["quantity"].astype("float")

df_all = df_all[['local_datetime', 'nGCBMUnitID', 'quantity']]

df_all["local_datetime"] = pd.to_datetime(df_all["local_datetime"], format="%Y-%m-%dT%H:%M:%S.%f")

p = df_all.pivot_table(index='local_datetime', columns='nGCBMUnitID', values='quantity', aggfunc=np.sum)

p.fillna(0, inplace=True)

# p.to_csv(os.path.join(os.path.dirname(__file__), "next_7_days_gen_api.csv").replace("\\", "/"))
