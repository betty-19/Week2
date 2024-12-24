import pandas as pd


df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')


tcp_columns = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
rtt_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)']
throughput_columns = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']


def compute_values(df, column_list):
    for col in column_list:
        print(f"\n--- {col} ---")
        
       
        top_10 = df[col].nlargest(10)
        print("Top 10 values:")
        print(top_10)

       
        bottom_10 = df[col].nsmallest(10)
        print("\nBottom 10 values:")
        print(bottom_10)
        
      
        most_frequent = df[col].value_counts().head(10)
        print("\nMost frequent values:")
        print(most_frequent)


compute_values(df, tcp_columns)
compute_values(df, rtt_columns)
compute_values(df, throughput_columns)
