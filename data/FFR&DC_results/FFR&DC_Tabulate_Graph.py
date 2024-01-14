from datetime import datetime, timedelta, date
import pandas as pd
import os
import xlrd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl
from dateutil.parser import parse


def create_combined_all_ffr_file():
    folder_path = r'C:\Users\nathanael.sims\PycharmProjects\NGESOBatteryHackathon\data\FFR&DC_results\ffr_post_tender_results'
    root_folder_path = r'C:\Users\nathanael.sims\PycharmProjects\NGESOBatteryHackathon\data\FFR&DC_results'
    all_files = os.listdir(folder_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    df_list = []

    for csv in csv_files:
        file_path = os.path.join(folder_path, csv)
        try:
            df = pd.read_csv(file_path)
            df['filename'] = str(csv)
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                df_list.append(df)
            except Exception as e:
                print(f"Could not read file {csv} because of error: {e}")
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)

    combined_df['service'] = 'FFR'

    combined_df = combined_df[['Tender Ref', 'Status', 'Company Name', 'Tendered Unit \n(BMU/Unit ID)', 'Start Date', 'End Date', 'filename', 'service', 'Primary Response (max.) @ 0.8Hz (MW)', 'Primary Response (max.) @ 0.5Hz (MW)',
               'Primary Response (max.) @ 0.2Hz (MW)', 'Secondary Response (max.) @ 0.5/0.5Hz (MW)',
               'Secondary Response (max.) @ 0.2/0.2Hz (MW)', 'High Frequency Response (max.) @ 0.5Hz (MW)',
               'High Frequency Response (max.) @ 0.2Hz (MW)']]

    combined_df.rename(columns = {'Tendered Unit \n(BMU/Unit ID)': 'BMU Unit ID'}, inplace=True)

    combined_df.dropna(subset=['Start Date'], inplace=True)

    # for index, row in combined_df.iterrows():
    #     try:
    #         combined_df.at[index, 'Start Date'] = pd.to_datetime(row['Start Date'], format="%d/%m/%Y")
    #     except:
    #         combined_df.at[index, 'Start Date'] = datetime.strptime('01/01/1900', "%d/%m/%Y") + timedelta(days=float(row['Start Date']))
    #
    # for index, row in combined_df.iterrows():
    #     try:
    #         combined_df.at[index, 'End Date'] = pd.to_datetime(row['End Date'], format="%d/%m/%Y")
    #     except:
    #         combined_df.at[index, 'End Date'] = datetime.strptime('01/01/1900', "%d/%m/%Y") + timedelta(days=float(row['End Date']))

    combined_df.to_csv(os.path.join(root_folder_path, '0_Combined_All_FFR_file.csv'))
    print('Complete')

create_combined_all_ffr_file()

# def sort_ffr_combined_create_accepted_rejected_POTES():
#     root_folder_path = r'C:\Users\nathanael.sims\PycharmProjects\NGESOBatteryHackathon\data\FFR&DC_results\ffr_post_tender_results'
#     combined_df = pd.read_csv(os.path.join(root_folder_path, 'Combined_All_FFR_file.csv'))
#     combined_df.sort_values(by='Start Date', inplace=True)
#     combined_df = combined_df[combined_df['Tendered Unit \n(BMU/Unit ID)'] == 'POTES-1']
#     combined_df_accepted = combined_df[combined_df['Status'] == 'Accepted']
#     combined_df_rejected = combined_df[combined_df['Status'] == 'Rejected']
#     combined_df_accepted.to_csv(os.path.join(root_folder_path, 'Combined_Accepted_FFR_file.csv'), index=False)
#     combined_df_rejected.to_csv(os.path.join(root_folder_path, 'Combined_Rejected_FFR_file.csv'), index=False)
#
# sort_ffr_combined_create_accepted_rejected_POTES()

def combine_ffr_with_DC():
    root_folder_path = r'C:\Users\nathanael.sims\PycharmProjects\NGESOBatteryHackathon\data\FFR&DC_results'
    df_dc = pd.read_excel(os.path.join(root_folder_path, '0_All_DC_Results_Modo.xlsx'))
    df_dc = df_dc[['company', 'unit_name', 'delivery_s ar', 'delivery_end', 'service', 'cleared_volume']]
    df_dc.rename(columns={'delivery_s ar': 'Start Date', 'delivery_end': 'End Date', 'company': 'Company Name', 'unit_name': 'BMU Unit ID', 'cleared_volume': 'DC_Volume'}, inplace=True)
    df_dc['Status'] = 'Accepted'
    combined_df = pd.read_csv(os.path.join(root_folder_path, '0_Combined_All_FFR_file.csv'))
    combined_dc_ffr = pd.concat([combined_df, df_dc], axis=0)
    # combined_dc_ffr['Start Date'] = pd.to_datetime(combined_dc_ffr['Start Date'], format='%d/%m/%Y').dt.date
    # combined_dc_ffr['End Date'] = pd.to_datetime(combined_dc_ffr['End Date'], format='%d/%m/%Y').dt.date
    # combined_dc_ffr.sort_values(by='Start Date', inplace=True)
    combined_dc_ffr.to_csv(os.path.join(root_folder_path, '1_Combined_All_FFR_DC_file.csv'))
    print('Complete')

combine_ffr_with_DC()

def create_final_table():
    root_folder_path = r'C:\Users\nathanael.sims\PycharmProjects\NGESOBatteryHackathon\data\FFR&DC_results'
    df = pd.read_csv(os.path.join(root_folder_path, '1_Combined_All_FFR_DC_file.csv'))

    def dates_range(row):
        x = df.columns.get_loc("Start Date")
        y = df.columns.get_loc("End Date")
        return pd.date_range(start=row.iat[x], end=row.iat[y], freq='D')

    df['DATE'] = df.apply(dates_range, axis='columns')
    df = df[['DATE', 'service']].explode('DATE')
    df['COUNT'] = 1
    df = df.pivot_table(index='DATE', columns='service', aggfunc='sum')

    df.to_csv(os.path.join(root_folder_path, '2_Combined_All_FFR_DC_table.csv'))

create_final_table()

def graph_tabulate_dc_ffr_overall_quantity():
    root_folder_path = r'C:\Users\nathanael.sims\PycharmProjects\NGESOBatteryHackathon\data\FFR&DC_results'
    df = pd.read_csv(os.path.join(root_folder_path, '1_Combined_All_FFR_DC_file.csv'))

    df.fillna(0, inplace=True)

    df = df[df['Status'] == 'Accepted']

    # earliest_date = min(df['Start Date'])
    earliest_date = '01/07/2021'
    latest_date = '31/12/2023'

    df_series = pd.DataFrame(pd.date_range(start=earliest_date, end=latest_date, freq='D'))

    df_series = df_series.rename(columns={df_series.columns[0]: "Date"})

    df_series[['Primary Response (max.) @ 0.8Hz (MW)', 'Primary Response (max.) @ 0.5Hz (MW)',
               'Primary Response (max.) @ 0.2Hz (MW)', 'Secondary Response (max.) @ 0.5/0.5Hz (MW)',
               'Secondary Response (max.) @ 0.2/0.2Hz (MW)', 'High Frequency Response (max.) @ 0.5Hz (MW)',
               'High Frequency Response (max.) @ 0.2Hz (MW)', 'DC_Volume']] = 0

    for _, row in df.iterrows():
        mask = (df_series['Date'] >= row['Start Date']) & (df_series['Date'] <= row['End Date'])
        columns_to_sum = ['Primary Response (max.) @ 0.8Hz (MW)', 'Primary Response (max.) @ 0.5Hz (MW)',
                          'Primary Response (max.) @ 0.2Hz (MW)', 'Secondary Response (max.) @ 0.5/0.5Hz (MW)',
                          'Secondary Response (max.) @ 0.2/0.2Hz (MW)', 'High Frequency Response (max.) @ 0.5Hz (MW)',
                          'High Frequency Response (max.) @ 0.2Hz (MW)', 'DC_Volume']
        df_series.loc[mask, columns_to_sum] += row[columns_to_sum]

    # df_series[['Primary Response (max.) @ 0.8Hz (MW)', 'Primary Response (max.) @ 0.5Hz (MW)',
    #            'Primary Response (max.) @ 0.2Hz (MW)', 'Secondary Response (max.) @ 0.5/0.5Hz (MW)',
    #            'Secondary Response (max.) @ 0.2/0.2Hz (MW)', 'High Frequency Response (max.) @ 0.5Hz (MW)',
    #            'High Frequency Response (max.) @ 0.2Hz (MW)', 'DC_Volume']] = 0
    #
    # for index1, row1 in df.iterrows():
    #     mask = (df_series['Date'] >= row1['Start Date']) & (df_series['Date'] <= row1['End Date'])
    #     df_series.loc[mask, 'Primary Response (max.) @ 0.8Hz (MW)'] += row1['Primary Response (max.) @ 0.8Hz (MW)']
    #     df_series.loc[mask, 'Primary Response (max.) @ 0.5Hz (MW)'] += row1['Primary Response (max.) @ 0.5Hz (MW)']
    #     df_series.loc[mask, 'Primary Response (max.) @ 0.2Hz (MW)'] += row1['Primary Response (max.) @ 0.2Hz (MW)']
    #
    #     df_series.loc[mask, 'Secondary Response (max.) @ 0.5/0.5Hz (MW)'] += row1[
    #         'Secondary Response (max.) @ 0.5/0.5Hz (MW)']
    #     df_series.loc[mask, 'Secondary Response (max.) @ 0.2/0.2Hz (MW)'] += row1[
    #         'Secondary Response (max.) @ 0.2/0.2Hz (MW)']
    #
    #     df_series.loc[mask, 'High Frequency Response (max.) @ 0.5Hz (MW)'] += row1[
    #         'High Frequency Response (max.) @ 0.5Hz (MW)']
    #     df_series.loc[mask, 'High Frequency Response (max.) @ 0.2Hz (MW)'] += row1[
    #         'High Frequency Response (max.) @ 0.2Hz (MW)']
    #
    #     df_series.loc[mask, 'DC_Volume'] += row1['DC_Volume']

    df_series.to_csv(os.path.join(root_folder_path, '3_FFR_DC_Table2.csv'))

    # # create stacked bar chart
    # df_series.set_index('Date', inplace=True)
    #
    # df_series.plot(kind='bar', stacked=True, figsize=(10, 6))
    #
    # plt.xlabel('Date')
    # plt.ylabel('Response Values')
    # plt.title('Stacked Bar Chart of Response Values over Time')
    #
    # # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    # # plt.gca().xaxis.set_major_locator(mdates.DateFormatter('%b %d'))
    #
    # plt.show()

graph_tabulate_dc_ffr_overall_quantity()