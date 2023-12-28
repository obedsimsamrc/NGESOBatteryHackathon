import pandas as pd

# import file with FFR results and DC results included.
df = pd.read_excel('ESO Data Challenge/DC_Results_with_FFR_Added.xlsx', sheet_name='POTES')

def dates_range(row):
    return pd.date_range(start=row.iat[3], end=row.iat[5], freq='H')

df['DATE'] = df.apply(dates_range, axis='columns')
df = df[['DATE', 'service']].explode('DATE')
df['COUNT'] = 1
df = df.pivot_table(index='DATE', columns='service', aggfunc='sum')

#output final file
df.to_csv('DC&FFR_Results_Tabulated.csv')



