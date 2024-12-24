import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')


df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)
df['Avg Bearer TP DL (kbps)'].fillna(df['Avg Bearer TP DL (kbps)'].mean(), inplace=True)
df['Avg Bearer TP UL (kbps)'].fillna(df['Avg Bearer TP UL (kbps)'].mean(), inplace=True)


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].mean(), df[column])


remove_outliers(df, 'TCP DL Retrans. Vol (Bytes)')
remove_outliers(df, 'TCP UL Retrans. Vol (Bytes)')
remove_outliers(df, 'Avg RTT DL (ms)')
remove_outliers(df, 'Avg RTT UL (ms)')
remove_outliers(df, 'Avg Bearer TP DL (kbps)')
remove_outliers(df, 'Avg Bearer TP UL (kbps)')


df['Handset Type'].fillna(df['Handset Type'].mode()[0], inplace=True)


aggregated_data = df.groupby('IMSI').agg({
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Handset Type': 'first',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean'
}).reset_index()


aggregated_data['Average Throughput DL (kbps)'] = (aggregated_data['Avg Bearer TP DL (kbps)'])
aggregated_data['Average Throughput UL (kbps)'] = (aggregated_data['Avg Bearer TP UL (kbps)'])


print(aggregated_data.head())
