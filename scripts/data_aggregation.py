import pandas as pd
import os


data_folder = os.path.abspath('../Data')


if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Directory {data_folder} does not exist.")


csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {data_folder}.")


required_columns = [
    'MSISDN/Number', 'bearer id', 'Dur. (ms)', 
    'Total DL (Bytes)', 'Total UL (Bytes)', 
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)', 
    'YouTube DL (Bytes)', 'YouTube UL (Bytes)', 
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 
    'Google DL (Bytes)', 'Google UL (Bytes)', 
    'Email DL (Bytes)', 'Email UL (Bytes)', 
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
    'Other DL', 'Other UL'
]


all_data = []
for file in csv_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)

    
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"Missing required columns in {file}")
    
    all_data.append(df)


combined_df = pd.concat(all_data, ignore_index=True)


grouped = combined_df.groupby('MSISDN/Number').agg({
    'bearer id': 'count',                
    'Dur. (ms)': 'sum',                  
    'Total DL (Bytes)': 'sum',           
    'Total UL (Bytes)': 'sum',           
    'Social Media DL (Bytes)': 'sum',
    'Social Media UL (Bytes)': 'sum',
    'YouTube DL (Bytes)': 'sum',
    'YouTube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Netflix UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Google UL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Email UL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Gaming UL (Bytes)': 'sum',
    'Other DL': 'sum',
    'Other UL': 'sum'
})


grouped['Social Media Total'] = grouped['Social Media DL (Bytes)'] + grouped['Social Media UL (Bytes)']
grouped['YouTube Total'] = grouped['YouTube DL (Bytes)'] + grouped['YouTube UL (Bytes)']
grouped['Netflix Total'] = grouped['Netflix DL (Bytes)'] + grouped['Netflix UL (Bytes)']
grouped['Google Total'] = grouped['Google DL (Bytes)'] + grouped['Google UL (Bytes)']
grouped['Email Total'] = grouped['Email DL (Bytes)'] + grouped['Email UL (Bytes)']
grouped['Gaming Total'] = grouped['Gaming DL (Bytes)'] + grouped['Gaming UL (Bytes)']
grouped['Other Total'] = grouped['Other DL'] + grouped['Other UL']


result = grouped.reset_index()


output_folder = os.path.abspath('./results')
os.makedirs(output_folder, exist_ok=True)
result.to_csv(os.path.join(output_folder, 'aggregated_data.csv'), index=False)

print("Data aggregation complete! Results saved to './results/aggregated_data.csv'.")
