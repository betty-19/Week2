import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')


agg_data = data.groupby('MSISDN').agg({
    'Dur. (ms)': 'sum', 
    'Total DL (Bytes)': 'sum',  
    'Total UL (Bytes)': 'sum', 
    'bearer id': 'count'  
}).reset_index()


agg_data.rename(columns={
    'Dur. (ms)': 'Total_Duration',
    'Total DL (Bytes)': 'Total_DL',
    'Total UL (Bytes)': 'Total_UL',
    'bearer id': 'Session_Frequency'
}, inplace=True)


agg_data['Total_Traffic'] = agg_data['Total_DL'] + agg_data['Total_UL']


top_customers_duration = agg_data.nlargest(10, 'Total_Duration')
top_customers_dl = agg_data.nlargest(10, 'Total_DL')
top_customers_ul = agg_data.nlargest(10, 'Total_UL')
top_customers_sessions = agg_data.nlargest(10, 'Session_Frequency')


scaler = MinMaxScaler()
agg_data[['Total_Duration', 'Total_DL', 'Total_UL', 'Session_Frequency']] = scaler.fit_transform(
    agg_data[['Total_Duration', 'Total_DL', 'Total_UL', 'Session_Frequency']]
)


kmeans = KMeans(n_clusters=3, random_state=42)
agg_data['Cluster'] = kmeans.fit_predict(
    agg_data[['Total_Duration', 'Total_DL', 'Total_UL', 'Session_Frequency']]
)


inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(agg_data[['Total_Duration', 'Total_DL', 'Total_UL', 'Session_Frequency']])
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.show()


cluster_summary = agg_data.groupby('Cluster').agg({
    'Total_Duration': ['min', 'max', 'mean', 'sum'],
    'Total_DL': ['min', 'max', 'mean', 'sum'],
    'Total_UL': ['min', 'max', 'mean', 'sum'],
    'Session_Frequency': ['min', 'max', 'mean', 'sum']
})


application_traffic = data.groupby('MSISDN').agg({
    'Social Media DL (Bytes)': 'sum',
    'Social Media UL (Bytes)': 'sum',
    'YouTube DL (Bytes)': 'sum',
    'YouTube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Netflix UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Google UL (Bytes)': 'sum'
}).reset_index()


application_traffic['Social_Media_Total'] = application_traffic['Social Media DL (Bytes)'] + application_traffic['Social Media UL (Bytes)']
application_traffic['YouTube_Total'] = application_traffic['YouTube DL (Bytes)'] + application_traffic['YouTube UL (Bytes)']
application_traffic['Netflix_Total'] = application_traffic['Netflix DL (Bytes)'] + application_traffic['Netflix UL (Bytes)']
application_traffic['Google_Total'] = application_traffic['Google DL (Bytes)'] + application_traffic['Google UL (Bytes)']


top_social_media_users = application_traffic.nlargest(10, 'Social_Media_Total')
top_youtube_users = application_traffic.nlargest(10, 'YouTube_Total')
top_netflix_users = application_traffic.nlargest(10, 'Netflix_Total')


app_usage = {
    'Social Media': application_traffic['Social_Media_Total'].sum(),
    'YouTube': application_traffic['YouTube_Total'].sum(),
    'Netflix': application_traffic['Netflix_Total'].sum()
}

plt.figure(figsize=(8, 6))
plt.bar(app_usage.keys(), app_usage.values(), color=['blue', 'red', 'green'])
plt.xlabel('Application')
plt.ylabel('Total Traffic (Bytes)')
plt.title('Top 3 Most Used Applications')
plt.show()


silhouette_avg = silhouette_score(
    agg_data[['Total_Duration', 'Total_DL', 'Total_UL', 'Session_Frequency']], 
    agg_data['Cluster']
)
print(f"Silhouette Score for Clustering: {silhouette_avg}")


cluster_summary.to_csv('cluster_summary.csv', index=False)


print("Clustering complete. Visualizations and summary exported.")
