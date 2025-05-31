import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('K means/student_clustering.csv')

# Step 2: Convert to NumPy array
X = df.iloc[:, :].values

# Step 3: Apply the Elbow Method
wcss = []
for k in range(1, 11):  # Try K from 1 to 10
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)  # WCSS

# Step 4: Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()
# Apply KMeans with the chosen K (e.g., 4)
km = KMeans(n_clusters=4, init='k-means++', max_iter=100, random_state=42)
y_means = km.fit_predict(X)

# Plot the clusters
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='red', label='Cluster 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='blue', label='Cluster 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color='green', label='Cluster 3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='yellow', label='Cluster 4')

# Plot centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=200, color='black', marker='X', label='Centroids')

plt.title('K-Means Clustering (K=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()