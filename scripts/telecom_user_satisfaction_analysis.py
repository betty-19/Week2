import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')




engagement_features = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
experience_features = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[engagement_features + experience_features])


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)




throughput_cluster = df.groupby('Cluster')[engagement_features].mean()
less_engaged_cluster = throughput_cluster.mean(axis=1).idxmin()  


retransmission_cluster = df.groupby('Cluster')[experience_features].mean()
worst_experience_cluster = retransmission_cluster.mean(axis=1).idxmax() 


engagement_centroid = kmeans.cluster_centers_[less_engaged_cluster]
experience_centroid = kmeans.cluster_centers_[worst_experience_cluster]


def euclidean_distance(row, centroid):
    return np.linalg.norm(row - centroid)


df['Engagement Score'] = df[engagement_features].apply(
    lambda row: euclidean_distance(row, engagement_centroid), axis=1)

df['Experience Score'] = df[experience_features].apply(
    lambda row: euclidean_distance(row, experience_centroid), axis=1)


print("\n--- Engagement and Experience Scores ---")
print(df[['User', 'Engagement Score', 'Experience Score']])


df.to_csv('user_scores.csv', index=False)
