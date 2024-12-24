# User Engagement Analysis

## Overview
This project focuses on exploring and analyzing a telecommunications dataset to understand user engagement and improve Quality of Service (QoS). The analysis consists of two main tasks:

1. **Exploratory Data Analysis (EDA):** To uncover patterns, treat missing values, and identify outliers.
2. **User Engagement Analysis:** To assess user activity metrics and classify users based on engagement levels.

---

## Task 1: Exploratory Data Analysis (EDA)

### Goals:
1. Handle missing values and outliers.
2. Analyze key metrics (mean, median, etc.) and their importance.
3. Conduct univariate and bivariate analysis.
4. Perform dimensionality reduction using PCA.

### Steps:
1. **Data Cleaning:**
   - Replace missing values with the column mean.
   - Identify and treat outliers using appropriate methods (e.g., IQR).

2. **Variable Transformation:**
   - Segment users into deciles based on total session duration.
   - Compute total data usage (DL+UL) per decile.

3. **Statistical Analysis:**
   - Compute mean, median, variance, and standard deviation for quantitative variables.
   - Analyze the dispersion of data to understand variability.

4. **Visualization:**
   - Histograms, boxplots, and scatterplots for key variables.
   - Correlation heatmap to identify relationships between variables.

5. **Dimensionality Reduction:**
   - Apply Principal Component Analysis (PCA) to reduce dimensions and interpret results.

---

## Task 2: User Engagement Analysis

### Goals:
1. Aggregate user engagement metrics (session frequency, duration, total traffic).
2. Classify users into engagement groups using k-means clustering.
3. Analyze application-specific engagement and identify top users.

### Steps:
1. **Metric Aggregation:**
   - Compute session frequency, total session duration, and total data usage for each user (MSISDN).
   - Report the top 10 users for each metric.

2. **Normalization and Clustering:**
   - Normalize engagement metrics.
   - Use the elbow method to find the optimal number of clusters for k-means.
   - Classify users into three engagement clusters and compute summary statistics for each cluster.

3. **Application Analysis:**
   - Aggregate total traffic per application.
   - Identify the top 10 most engaged users per application.
   - Visualize the top three applications by usage.

---



---

## How to Run the Analysis

1. **Dependencies:**
   - Python 3.x
   - Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

2. **Scripts:**
   - `eda_analysis.py`: Contains code for Task 1.
   - `engagement_analysis.py`: Contains code for Task 2.

3. **Execution:**
   - Run `eda_analysis.py` for exploratory data analysis.
   - Run `user_engagement_analysis.py` for user engagement analysis.



## Visualization Examples

### Task 1:
- Correlation heatmap of application data usage.
- PCA biplot to illustrate dimensionality reduction.

### Task 2:
- Elbow plot for optimal k in k-means.
- Bar chart of top 3 applications by traffic.

---