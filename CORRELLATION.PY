import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset (update path if needed)
file_path = "Titanic-Dataset.csv"  # Change if needed
df = pd.read_csv(file_path)

# Selecting numerical features only
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# 1. Compute correlation coefficients
pearson_corr = numerical_df.corr(method='pearson')  # Pearson correlation
spearman_corr = numerical_df.corr(method='spearman')  # Spearman correlation
kendall_corr = numerical_df.corr(method='kendall')  # Kendall correlation

# Print correlation matrices
print("\nPearson Correlation:\n", pearson_corr)
print("\nSpearman Correlation:\n", spearman_corr)
print("\nKendall Correlation:\n", kendall_corr)

# 2. Visualizing Correlations using Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Spearman Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Kendall Correlation Heatmap")
plt.show()

# 3. Identifying Highly Correlated Variables (Threshold: 0.7)
high_corr = pearson_corr.abs().unstack().sort_values(ascending=False)
high_corr = high_corr[high_corr != 1]  # Remove self-correlation (1.0)
high_corr_pairs = high_corr[high_corr > 0.7]  # Select highly correlated pairs

print("\nHighly Correlated Variables (Pearson, > 0.7):\n", high_corr_pairs)
