import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# start seaborn plotting libs
sns.set()

"""
assign random data points

n_samples – If int, it is the total number of points equally divided among clusters. If array-like, each element of the sequence indicates the number of samples per cluster.
n_features – The number of features for each sample.
centers – The number of centers to generate, or the fixed center locations. If n_samples is an int and centers is None, 3 centers are generated. If n_samples is array-like, centers must be either None or an array of length equal to the length of n_samples.
cluster_std – The standard deviation of the clusters.
center_box – The bounding box for each cluster center when centers are generated at random.
shuffle – Shuffle the samples.
random_state – Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls. See Glossary  .
return_centers – If True, then return the centers of each cluster
"""
points, cluster_indexes = make_blobs(n_samples=300, centers=4,
                                     cluster_std=0.8, random_state=0)
x = points[:, 0]
y = points[:, 1]

# elbow method used to determine number of clusters
inertias = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(points)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# plot data points
plt.scatter(x, y, s=50, alpha=0.7)

# build out k means cluster
kmeans = KMeans(n_clusters=4, random_state=0)
# train data model
kmeans.fit(points)
# make prediction
predicted_cluster_indexes = kmeans.predict(points)

# categorized cluster graph
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')

# show centroids for each cluster
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100)
plt.show()


