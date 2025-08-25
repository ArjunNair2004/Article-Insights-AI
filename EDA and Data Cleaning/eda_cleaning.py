import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Create folder for graphs
os.makedirs("eda_graphs", exist_ok=True)

# 1. Load the dataset
df = pd.read_csv("dataset/dataset.csv", encoding='utf-8')
df.columns = df.columns.str.strip()

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

if 'shares' not in df.columns:
    print("ERROR: 'shares' column missing!")
    exit(1)

# 2. Data types and missing values
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# 3. Drop all-NaN columns and fill numeric missing values
df.dropna(axis=1, how='all', inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 4. Drop non-predictive column 'url'
if 'url' in df.columns:
    df.drop('url', axis=1, inplace=True)

# 5. Summary statistics
print("\nSummary Statistics:\n", df.describe())

# 6. Univariate Analysis – Shares Distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['shares'], bins=50, kde=True)
plt.title("Shares Distribution (Before Log)")
plt.xlabel("Shares")
plt.tight_layout()
plt.savefig("eda_graphs/shares_distribution_before_log.png")
plt.close()

# 7. Boxplot to detect outliers
plt.figure(figsize=(12, 2))
sns.boxplot(x=df['shares'])
plt.title("Shares Boxplot")
plt.tight_layout()
plt.savefig("eda_graphs/shares_boxplot.png")
plt.close()

# 8. Analyze timedelta (univariate)
if 'timedelta' in df.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df['timedelta'], bins=50, kde=True)
    plt.title("Timedelta Distribution")
    plt.xlabel("Days since publication")
    plt.tight_layout()
    plt.savefig("eda_graphs/timedelta_distribution.png")
    plt.close()

# 9. Bivariate Analysis – Timedelta vs Shares
if 'timedelta' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='timedelta', y='shares', data=df)
    plt.title("Shares vs Timedelta")
    plt.xlabel("Days since publication")
    plt.ylabel("Shares")
    plt.tight_layout()
    plt.savefig("eda_graphs/shares_vs_timedelta.png")
    plt.close()

# 10. Log transform 'shares'
df['shares'] = np.log1p(df['shares'])

# 11. Shares distribution after log transform
plt.figure(figsize=(10, 4))
sns.histplot(df['shares'], bins=50, kde=True)
plt.title("Shares Distribution (After Log)")
plt.xlabel("Log(1 + Shares)")
plt.tight_layout()
plt.savefig("eda_graphs/shares_distribution_after_log.png")
plt.close()

# 12. Correlation heatmap (Multivariate)
correlation = df.corr(numeric_only=True)
top_corr = correlation['shares'].sort_values(ascending=False)
print("\nTop Features Correlated with 'shares':\n", top_corr.head(10))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation[['shares']].sort_values(by='shares', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation with Target ('shares')")
plt.tight_layout()
plt.savefig("eda_graphs/correlation_with_shares.png")
plt.close()

# 13. Scatterplots for Top Correlated Features vs Shares
top_features = top_corr.index[1:6]  

for feature in top_features:
    if feature in ['is_weekend', 'kw_avg_avg', 'kw_max_avg', 'kw_min_avg', 'LDA_03']:
        continue
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[feature], y=df['shares'])
    plt.title(f"Log(Shares) vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Log(1 + Shares)")
    plt.tight_layout()
    plt.savefig(f"eda_graphs/shares_vs_{feature}.png")
    plt.close()

# 14. Pairplot of Top Correlated Features + Shares
filtered_top_features = [f for f in top_features if f not in ['is_weekend', 'kw_avg_avg', 'kw_max_avg', 'kw_min_avg', 'LDA_03']]
sns.pairplot(df[filtered_top_features + ['shares']])
plt.savefig("eda_graphs/pairplot_top_features.png")
plt.close()

# Drop 'timedelta' column before saving the cleaned dataset
if 'timedelta' in df.columns:
    df.drop('timedelta', axis=1, inplace=True)

# 15. Save cleaned dataset
df.to_csv("dataset/cleaned_dataset.csv", index=False)
print("\nCleaned data saved as 'dataset/cleaned_dataset.csv'")

