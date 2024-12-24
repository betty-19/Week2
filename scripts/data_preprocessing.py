
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv('../Data/Copy of Week2_challenge_data_source(CSV).csv')


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


df.fillna(df.mean(), inplace=True)


from scipy.stats import zscore

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
z_scores = np.abs(zscore(df[numerical_columns]))


outlier_threshold = 3
outliers = np.where(z_scores > outlier_threshold)
df_cleaned = df[(z_scores < outlier_threshold).all(axis=1)]


print("\nData Types and Non-null Counts:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())


df_cleaned['Total Data'] = df_cleaned['Total DL (Bytes)'] + df_cleaned['Total UL (Bytes)']


df_cleaned['Decile Class'] = pd.qcut(df_cleaned['Dur. (ms)'], 10, labels=False)


decile_summary = df_cleaned.groupby('Decile Class')['Total Data'].sum()
print("\nDecile Summary:\n", decile_summary)


basic_metrics = df_cleaned.describe().T
print("\nBasic Metrics:\n", basic_metrics)


dispersion_params = {
    "Range": df_cleaned.max() - df_cleaned.min(),
    "Variance": df_cleaned.var(),
    "Std Dev": df_cleaned.std(),
    "IQR": df_cleaned.quantile(0.75) - df_cleaned.quantile(0.25)
}
print("\nDispersion Parameters:\n", dispersion_params)


plt.figure(figsize=(10, 6))
df_cleaned['Total Data'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Total Data")
plt.xlabel("Total Data (Bytes)")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['Total Data'], color='skyblue')
plt.title("Boxplot of Total Data")
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Social Media DL (Bytes)', y='Total Data', data=df_cleaned, color='green')
plt.title("Social Media DL vs Total Data")
plt.xlabel("Social Media DL (Bytes)")
plt.ylabel("Total Data (Bytes)")
plt.show()


correlation_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                       'YouTube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

correlation_matrix = df_cleaned[correlation_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print("\nCorrelation Matrix:\n", correlation_matrix)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned[correlation_columns])


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio (PCA):\n", explained_variance)


plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='black')
plt.title("PCA Result")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


print("\nPCA Interpretation:")
print("1. The first principal component explains {:.2f}% of the variance.".format(explained_variance[0] * 100))
print("2. The second principal component explains {:.2f}% of the variance.".format(explained_variance[1] * 100))
print("3. Together, the two components explain {:.2f}% of the variance.".format(sum(explained_variance) * 100))
print("4. PCA helps in reducing redundant information and simplifying the dataset.")

