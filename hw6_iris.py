# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# Homework 6

import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm

# function definitions
# this function is from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
# end function definitions


# Load the datasets w/o assigning labels
print("Loading the Iris dataset features...")
iris = datasets.load_iris()
X = iris.data
# check-expect 150 3
# print(X.shape)

# using elbow method to choose K
distortions = []
for i in range(1, 11):
    km_elbow = KMeans(n_clusters=i, init='k-means++', n_init='auto', max_iter=100, random_state=0)
    km_elbow.fit(X)
    distortions.append(km_elbow.inertia_)

#plot distortions for different K
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()

# model parameters
truncate_level = 5 # dendogram truncate level
clusters = 3
iterations = 100

# Begin kmeans analysis
km = KMeans(n_clusters=clusters, init='k-means++', n_init='auto', max_iter=iterations, tol=1e-04, random_state=0)
km_train_start_time = time.time()
labels_km = km.fit_predict(X)
training_time = time.time() - km_train_start_time

# check-expect 150 data points 
# print(labels_km) 
print(km.cluster_centers_) #centroids
print('SE = %.3f' % km.inertia_)    
print(f' KMeans Training time: {training_time}')
# end kmeans analysis

# Begin Agglomerative clustering
ac1 = AgglomerativeClustering(n_clusters=clusters, metric='euclidean', linkage='complete')
ac1_train_start_time = time.time()
labels_ac = ac1.fit_predict(X)
training_time = time.time() - ac1_train_start_time
print(f'Agglomerative 1 Training time: {training_time}')

# Begin dendogram plot fo Agglomerative clustering
ac2 = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='average', distance_threshold=0)
ac2_train_start_time = time.time()
y_ac = ac2.fit_predict(X)
training_time = time.time() - ac2_train_start_time
print(f'Agglomerative 2 Training time: {training_time}')


fig, ax = plt.subplots(figsize=(8, 6))
plt.title("Agglomerative Clustering Dendrogram")
plt.xlabel("Data points")
plt.tight_layout()
plot_dendrogram(ac2, truncate_mode="level", p=5)
plt.show()
# end dendogram plot Agglomerative clustering

# scipy hierarchical clustering 
# this code is from the scipy documentation 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

linkage_start_time = time.time()
linkage_matrix = linkage(X, method='ward', metric='euclidean')
training_time = time.time() - linkage_start_time
print(f'Linkage Training time: {training_time}')

# plot the dendrogram
fig, ax = plt.subplots(figsize=(8, 6))

plt.title('Hierarchical Clustering Dendrogram (Euclidean)')
plt.xlabel('Data points') # uncomment this for datapoints on x axis -> too many points for readability with truncation
plt.ylabel('Distance')
plt.tight_layout()
dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
plt.show()

# calculate the linkage matrix
Z = linkage(X, method='ward')

# plot the dendrogram
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Datapoints')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='level', p=5)
plt.show()

# assign clusters based on a distance threshold
max_dist = 3
clusters = fcluster(Z, t=max_dist, criterion='distance')
# check-expect cluster labels
print(f'Cluster labels: {clusters}')
# end scipy hierarchical clustering
centers = np.array([X[ac1.labels_ == i].mean(axis=0) for i in range(ac1.n_clusters)])
sse_agg = ((X - centers[ac1.labels_]) ** 2).sum()

# create Linkage model
linkage_model = linkage(X, method='ward')

# compute SSE for Linkage model
sse_linkage = linkage_model[-1, 3]

print("SSE for Agglomerative Clustering model:", sse_agg)
print("SSE for Linkage model:", sse_linkage)

# silhouette plotting modelot

sil_labels = km.labels_
cluster_labels = np.unique(sil_labels)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, labels_ac, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[sil_labels == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()

print("Silhouette Score:", silhouette_score(iris.data, labels_km))