from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')

clustering_data = df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
                      'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)


centroids = kmeans.cluster_centers_


print("\n--- Cluster Centers ---")
print(centroids)


print("\n--- Cluster Descriptions ---")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i+1}:")
    print(f"- Avg. Downlink Throughput: {centroid[0]}")
    print(f"- Avg. Uplink Throughput: {centroid[1]}")
    print(f"- Avg. TCP DL Retrans. Vol: {centroid[2]}")
    print(f"- Avg. TCP UL Retrans. Vol: {centroid[3]}")
    print("\n")
