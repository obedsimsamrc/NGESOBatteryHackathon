import requests
import pandas as pd
import time
import numpy as np
from datetime import timedelta
import concurrent.futures
from urllib import parse
import os

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


# file_path = os.path.join(os.path.dirname(__file__), "train_data_second_freq.csv").replace("\\", "/")
# train_data = pd.read_csv(file_path)


def system_freq_api():
    # start time
    tic = time.time()

    months = {
        'January': int(timedelta(days=31).total_seconds()),
        'February': int(timedelta(days=28).total_seconds()),
        'March': int(timedelta(days=31).total_seconds()),
        'April': int(timedelta(days=30).total_seconds()),
        'May': int(timedelta(days=31).total_seconds()),
        'June': int(timedelta(days=30).total_seconds()),
        'July': int(timedelta(days=31).total_seconds()),
        'August': int(timedelta(days=31).total_seconds()),
        'September': int(timedelta(days=30).total_seconds()),
        'October': int(timedelta(days=31).total_seconds()),
        'November': int(timedelta(days=30).total_seconds()),
        'December': int(timedelta(days=31).total_seconds())
    }

    # function to fetch data from API
    def fetch_data(url):
        response = requests.get(url)
        print(response)
        return response.json()["result"]

    base_url = 'https://data.nationalgrideso.com/api/3/action/datastore_search?resource_id='

    train_ext_urls = [
        f'3617f1af-4d3c-45d3-a1f6-0d51a9c2167b&limit={months["October"]}',
        f'bcfa7999-1b14-444b-8f8f-6402092a0e9d&limit={months["November"]}',
        f'706ec5fe-777a-4a46-90de-b9089f93853c&limit={months["December"]}',
        f'fe2502b8-7fef-4027-8399-550a0c84f415&limit={months["January"]}',
        f'53e2be31-c68c-40b3-bb83-7ae12240107c&limit={months["February"]}',
        f'ac4e4031-0f86-48ff-8ecd-147d51ffbb81&limit={months["March"]}',
        f'bd6e1b3f-1abb-452c-813c-397a41f7af8d&limit={months["April"]}',
        f'625597eb-7364-4ba8-a306-a7f59dcdc2a7&limit={months["May"]}',
        f'bd754692-bce7-43b7-ba25-813921476391&limit={months["June"]}',
        f'94d95251-ccbb-4187-ae54-1b90ebf12b9f&limit={months["July"]}',
        f'ae40912b-de8f-43b1-8eeb-c58e469f2365&limit={months["August"]}',
        f'5cea7516-cbc3-416e-8dbf-162327c16b17&limit={months["September"]}',
        f'12291f14-95c3-4847-b2ef-3190eaa1193c&limit={months["October"]}',
        f'9bc4746e-3152-4c6f-886e-58377ab88e0e&limit={months["November"]}',
        f'afe9895c-5937-4e78-8949-f0f026643666&limit={months["December"]}',
        f'43000c20-1208-4ca7-a419-712c7a1d375c&limit={months["January"]}',
        f'181e8958-bf78-4bd6-b6ca-b8376da8e1aa&limit={months["February"]}',
        f'182081de-9773-4b5b-9c87-17adbd1576e6&limit={months["March"]}',
        f'b3bbd2e7-50b1-49cb-8326-bc8f09003cde&limit={months["April"]}',
        f'0a32dbfc-89e2-447d-a867-b10d5dbf3192&limit={months["May"]}',
        f'15c16754-6ed4-491e-949f-a42c6145ff24&limit={months["June"]}',
        f'3c62eb13-0531-4b20-b1a3-b9ff5095e5ef&limit={months["July"]}',
        f'232121b1-d3ab-4d5a-9e57-4cfc23d62eba&limit={months["August"]}',
        f'a1ccd82c-e522-4b5d-a4da-57dafab9d6de&limit={months["September"]}',
        f'74a4fae6-11e6-4e3b-9c48-7263f581f5c2&limit={months["October"]}',
        f'ff852ca5-1462-4ed6-ab77-886617719276&limit={months["November"]}',
    ]

    test_ext_urls = [
        f'ff852ca5-1462-4ed6-ab77-886617719276&limit={months["November"]}',
        f'2dbcb246-5f7a-45c8-9d04-19fd3f1b1595&limit={months["December"]}',
        f'5da35567-ca7f-4127-861c-cc07d488fd2d&limit={months["January"]}',
        f'7a9b6d47-b2fd-4efe-bb41-f6e227b0786a&limit={months["February"]}',
        f'69e906af-04f6-49b9-95e9-b02a64a46582&limit={months["March"]}',
        f'83fc86bb-50d6-4965-91fb-ddbdd02a5190&limit={months["April"]}',
        f'66c7be67-9689-4578-bc8d-684a2dea8248&limit={months["May"]}',
        f'2b881e0a-44b4-4957-9cc3-89bf0df6247a&limit={months["June"]}',
    ]

    # use concurrent.futures to fetch data concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = [executor.submit(fetch_data, base_url + url) for url in test_ext_urls]
    results = [future.result() for future in data]

    # This will iterate through the results list and add the dictionaries
    # that contains the key "records" to the records list
    records = [result["records"] for result in results]

    # Flatten the nested list
    flattened_records = [record for sublist in records for record in sublist]

    # extract 'dtm' and 'f' values
    extract_data = [(d['dtm'], d['f']) for d in flattened_records]

    # create dataframe
    df = pd.DataFrame(extract_data, columns=['dtm', 'f'])

    # Memory usage before conversion
    print("Original memory usage:", df.memory_usage(deep=True))

    # Convert the frequency column from and object to a float
    df["f"] = df["f"].astype(np.float32)

    # Convert the datetime column from an object to a datetime column
    df["dtm"] = pd.to_datetime(df["dtm"], format="%Y-%m-%dT%H:%M:%S")
    # convert the date column to datetime64[us]
    df['dtm'] = df['dtm'].astype('datetime64[us]')

    # Memory usage after conversion
    print("Converted memory usage:", df.memory_usage(deep=True))

    # file_name = "test_data_second_freq.csv"
    # file_path = os.path.join(os.path.dirname(__file__), file_name).replace("\\", "/")
    # df.to_csv(file_path)

    # end time
    toc = time.time()
    print(f"Total time of {round(toc - tic, 2):>,} seconds required to import frequency data")

    return df


# df = system_freq_api()


# API for the dynamic frequency market prices
def freq_market_prices(start_date, end_date):
    df = None

    # API Call to SQL db
    sql_query = '''SELECT * FROM  "888e5029-f786-41d2-bc15-cbfd1d285e96" ORDER BY "_id" ASC'''
    params = {'sql': sql_query}

    try:
        response = requests.get('https://api.nationalgrideso.com/api/3/action/datastore_search_sql',
                                params=parse.urlencode(params),
                                # verify=False,
                                )
        # print("Status Code:", response.status_code)
        print('\n')
        data = response.json()["result"]
        df = pd.DataFrame(data["records"])
    except requests.exceptions.RequestException as e:
        print(e.response.text)

    # Set the delivery date columns to datetime format
    df["Delivery Start"] = pd.to_datetime(df["Delivery Start"], format="%Y-%m-%dT%H:%M:%S")
    df["Delivery End"] = pd.to_datetime(df["Delivery End"], format="%Y-%m-%dT%H:%M:%S")
    df["EFA Date"] = pd.to_datetime(df["EFA Date"], format="%Y-%m-%d")

    # Drop useless columns
    df.drop(labels=["_full_text", "_id"], axis=1, inplace=True)

    # Set correct types and convert to £/kW/h
    df["Clearing Price"] = df["Clearing Price"].astype(np.float16)

    df.set_index('Delivery Start', inplace=True)

    # Set date range
    df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    # Create a new column with a unique number for each date
    df['date_number'] = df.groupby('EFA Date').ngroup()

    # Filter based on the service e.g. DCL & DCH
    services = ["DCL", "DCH", "DRL", "DRH", "DML", "DMH"]
    df_dict = {}

    for service in services:
        df_service = df.loc[df["Service"] == service]
        df_dict[service] = df_service

    # Group the data by day as the DCL and DCH are procured on an EFA basis (EFA 1-6)
    # Group the DataFrame by 4h slots, starting at 11pm
    # for df in df_dict.values():
    #     df = df.groupby(pd.Grouper(freq='4H', offset='-1H')).mean()

    return df_dict


# freq_market_prices_df = freq_market_prices(start_date='2021-12-31T23:00:00', end_date='2022-12-31T19:00:00')

test_or_train = "test"

freq_file_path = os.path.join(os.path.dirname(__file__), f"{test_or_train}_data_second_freq.csv").replace("\\", "/")
df = pd.read_csv(freq_file_path, index_col=0)
# df = pd.read_csv("/battery_model/Data/one_sec_frequency_data.csv")
df = pd.DataFrame(df)

df["f"] = df["f"].astype("float16")
df["dtm"] = pd.to_datetime(df["dtm"], format="%Y-%m-%d %H:%M:%S")

# Create a column with the difference between the frequency and the nominal frequency of 50Hz
df["delta_freq"] = df['f'] - 50

# Create a column indicating the dispatch requirements for dynamic containment low
df["disp_dcl_percent"] = np.where(df["delta_freq"] <= -0.5, 100,
                                  np.where(df["delta_freq"] <= -0.2, -316.7 * df["delta_freq"] - 58.3,
                                           np.where(df["delta_freq"] <= -0.015, -27 * df["delta_freq"] - 0.4,
                                                    0)))

# Create a column indicating the dispatch requirements for dynamic containment high
df["disp_dch_percent"] = np.where(df["delta_freq"] <= 0.015, 0,
                                  np.where(df["delta_freq"] < 0.2, -27 * df["delta_freq"] + 0.4,
                                           np.where(df["delta_freq"] < 0.5, -316.7 * df["delta_freq"] + 58.3,
                                                    100)))

# Create a column indicating the dispatch requirements for dynamic moderation low
df["disp_dml_percent"] = np.where(df["delta_freq"] <= -0.2, 100,
                                  np.where(df["delta_freq"] <= -0.1, -950 * df["delta_freq"] - 90,
                                           np.where(df["delta_freq"] <= -0.015,
                                                    -58.82 * df["delta_freq"] - 0.882, 0)))

# Create a column indicating the dispatch requirements for dynamic moderation high
df["disp_dmh_percent"] = np.where(df["delta_freq"] <= 0.015, 0,
                                  np.where(df["delta_freq"] < 0.2, -58.82 * df["delta_freq"] + 0.882,
                                           np.where(df["delta_freq"] < 0.5, -950 * df["delta_freq"] + 90,
                                                    -100)))

# Create a column indicating the dispatch requirements for dynamic regulation low
df["disp_drl_percent"] = np.where(df["delta_freq"] <= -0.2, 100,
                                  np.where(df["delta_freq"] <= -0.015, -540.541 * df["delta_freq"] - 8.1081, 0))

# Create a column indicating the dispatch requirements for dynamic regulation high
df["disp_drh_percent"] = np.where(df["delta_freq"] >= 0.015, -540.541 * df["delta_freq"] + 8.1081,
                                  np.where(df["delta_freq"] >= 0.2, -100, 0))

# Convert the % to fraction and divide by 60*30 to convert to half hours from seconds
cols = ["disp_dcl_percent", "disp_dch_percent", "disp_dml_percent", "disp_dmh_percent",
        "disp_drl_percent", "disp_drh_percent"]
for col in cols:
    df[col + "_kWh_per_kW"] = df[col] / (60 * 30)

# The resulting percentage power capacity values for each second of the year were transformed into
# energy dispatched in each time step
df_grouped = df.groupby(pd.Grouper(key='dtm', freq='30T')).sum()

# Create a column that indicates the EFA block, day and time within block
df_grouped["EFA Block Count"] = np.arange(len(df_grouped)) // 8
df_grouped["EFA Block Count"] = df_grouped["EFA Block Count"] % 12
# Shift the EFA block by 1 hour (2hh periods) as the first EFA block starts at 1am
df_grouped["EFA Block Count"] = df_grouped["EFA Block Count"].shift(periods=2, fill_value=5)

df_grouped["EFA HH Count"] = [i % 8 for i in range(len(df_grouped))]
df_grouped["EFA HH Count"] = df_grouped["EFA HH Count"].shift(periods=2)
df_grouped["EFA HH Count"].iloc[0], df_grouped["EFA HH Count"].iloc[1] = 6, 7
df_grouped["EFA HH Count"] = df_grouped["EFA HH Count"].astype("int")

cols_to_drop = ["f",
                "disp_dml_percent_kWh_per_kW", "disp_dmh_percent_kWh_per_kW",
                "disp_drl_percent_kWh_per_kW", "disp_drh_percent_kWh_per_kW", "disp_dml_percent", "disp_dmh_percent",
                "disp_drl_percent", "disp_drh_percent"]

df_grouped.drop(cols_to_drop, axis=1, inplace=True)

freq_half_hourly_file_path = os.path.join(os.path.dirname(__file__),
                                          f"prepared_{test_or_train}_hh_freq.csv").replace("\\", "/")
df_grouped.to_csv(freq_half_hourly_file_path)
