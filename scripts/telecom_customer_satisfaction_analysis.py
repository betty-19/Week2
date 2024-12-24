import pandas as pd
import numpy as np





df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')


df['Satisfaction Score'] = (df['Engagement Score'] + df['Experience Score']) / 2


top_10_satisfied_customers = df[['User', 'Satisfaction Score']].sort_values(by='Satisfaction Score', ascending=False).head(10)

print("\n--- Top 10 Satisfied Customers ---")
print(top_10_satisfied_customers)


top_10_satisfied_customers.to_csv('top_10_satisfied_customers.csv', index=False)
