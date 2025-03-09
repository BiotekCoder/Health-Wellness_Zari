import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load datasets
df = pd.read_csv(r'C:\Users\user 1\OneDrive\Documents\LivWell\livwell_lin_interpolated.csv') #Location of the dataset
indicators = pd.read_csv(r'C:\Users\user 1\OneDrive\Documents\LivWell\indicators.csv')       #Location of the feature list of dataset

# Display basic info
print(df.info())
print(df.head())

# Check missing values percentage
missing_percent = df.isnull().sum() * 100 / len(df)
missing_summary = missing_percent[missing_percent > 0].sort_values(ascending=False)
print(missing_summary)

# Drop columns where more than 50% values are missing
threshold = 50
df_reduced = df.drop(columns=missing_summary[missing_summary > threshold].index)

print(f"Original columns: {df.shape[1]}, After dropping high-missing columns: {df_reduced.shape[1]}")


# Fill missing numerical values with the median
num_cols = df_reduced.select_dtypes(include=['float64', 'int64']).columns
df_reduced[num_cols] = df_reduced[num_cols].fillna(df_reduced[num_cols].median())

# Fill missing categorical values with the mode
cat_cols = df_reduced.select_dtypes(include=['object']).columns
df_reduced[cat_cols] = df_reduced[cat_cols].fillna(df_reduced[cat_cols].mode().iloc[0])

# Check if any missing values remain
print(df_reduced.isnull().sum().sum())  # Should be 0

# Drop non-informative columns
drop_cols = ['country_name', 'country_code']
df_cluster = df_reduced.drop(columns=drop_cols, errors='ignore')

# Check remaining columns
print(df_cluster.shape)
print(df_cluster.head())


# Check data types
non_numeric_cols = df_cluster.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols)

df_cluster = df_cluster.drop(columns=non_numeric_cols, errors='ignore')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)
df_scaled = pd.DataFrame(df_scaled, columns=df_cluster.columns)
print(df_scaled.head())


#     DATA STANDARDIZATION DONE 
#     PROCEEDING TO HEIRARCHICAL DATA CLUSTERING
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(df_scaled, method='ward'))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering

# Apply hierarchical clustering
n_clusters = 2  # Value found by analyzing the dendrogram
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df_cluster = df_cluster.copy()
df_cluster['cluster_hc'] = hc.fit_predict(df_scaled)

# Check cluster distribution
print(df_cluster['cluster_hc'].value_counts())

from sklearn.mixture import GaussianMixture

# Apply GMM with 2 clusters
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
df_cluster['cluster_gmm'] = gmm.fit_predict(df_scaled)

# Check cluster distribution
print(df_cluster['cluster_gmm'].value_counts())


#             VISUALIZING THE CLUSTERS
from sklearn.decomposition import PCA

# Reduce dimensions to 2D
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(10, 5))

# Hierarchical Clustering
plt.subplot(1, 2, 1)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_cluster['cluster_hc'], cmap='viridis', alpha=0.5)
plt.title("Hierarchical Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Gaussian Mixture Model
plt.subplot(1, 2, 2)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_cluster['cluster_gmm'], cmap='plasma', alpha=0.5)
plt.title("Gaussian Mixture Model")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.show()

# Compare feature means for each cluster
cluster_summary_hc = df_cluster.groupby('cluster_hc').mean()
cluster_summary_gmm = df_cluster.groupby('cluster_gmm').mean()

print("Hierarchical Clustering Summary:\n", cluster_summary_hc.head())
print("Gaussian Mixture Model Summary:\n", cluster_summary_gmm.head())


#               ANALYZING THE CLUSTERS 
# Find the top 10 features with the biggest differences between clusters
diff_features = (cluster_summary_hc.loc[1] - cluster_summary_hc.loc[0]).abs().sort_values(ascending=False).head(10)
print(diff_features)

print(cluster_summary_hc[['gdp_pc', 'FF_ASFR_30.34', 'FF_ASFR_25.29', 'FF_ASFR_20.24']])
print(cluster_summary_gmm[['gdp_pc', 'FF_ASFR_30.34', 'FF_ASFR_25.29', 'FF_ASFR_20.24']])

# Create a comparison plot
features_to_plot = ['gdp_pc', 'FF_ASFR_30.34', 'FF_ASFR_25.29', 'FF_ASFR_20.24']
cluster_summary_hc[features_to_plot].plot(kind='bar', figsize=(10, 5), colormap='viridis')

plt.title("Comparison of Key Features Across Clusters")
plt.xlabel("Cluster")
plt.ylabel("Value")
plt.legend(title="Features")
plt.show()
