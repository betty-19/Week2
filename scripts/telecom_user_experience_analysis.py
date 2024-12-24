import pandas as pd


df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')


throughput_by_handset = df.groupby('Handset Type')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean()
tcp_retransmission_by_handset = df.groupby('Handset Type')[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean()


print("\n--- Average Throughput per Handset Type ---")
print(throughput_by_handset)

print("\n--- Average TCP Retransmission per Handset Type ---")
print(tcp_retransmission_by_handset)


