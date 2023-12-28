
import pandas as pd
import os

# replace with your folder's path
folder_path = r'C:\Users\nathanael.sims\PycharmProjects\ESO_Data_Challenge-Elexon\ffr_post_tender_results'

all_files = os.listdir(folder_path)

csv_files = [f for f in all_files if f.endswith('.csv')]

df_list = []

for csv in csv_files:
    file_path = os.path.join(folder_path, csv)
    try:
        df = pd.read_csv(file_path)
        df_list.append(df)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")
    except Exception as e:
        print(f"Could not read file {csv} because of error: {e}")

combined_df = pd.concat(df_list, ignore_index=True)

combined_df = combined_df[combined_df['Tendered Unit \n(BMU/Unit ID)'] == 'POTES-1']

combined_df_accepted = combined_df[combined_df['Status'] == 'Accepted']
combined_df_rejected = combined_df[combined_df['Status'] == 'Rejected']

combined_df_accepted.to_csv(os.path.join(folder_path, 'Combined_Accepted_FFR_file.csv'), index=False)
combined_df_rejected.to_csv(os.path.join(folder_path, 'Combined_Rejected_FFR_file.csv'), index=False)
