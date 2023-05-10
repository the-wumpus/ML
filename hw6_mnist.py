import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score

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
print("Loading the MNIST dataset features...")
digits = datasets.load_digits()
X = digits.data
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
clusters = 10
iterations = 1000

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

# silhouette plot
# calculate silhouette score
silhouette_avg = silhouette_score(digits.data, labels_km)
print("For n_clusters =", 10, "The average silhouette_score is :", silhouette_avg)

# calculate silhouette for each sample
sample_silhouette_values = silhouette_samples(digits.data, labels_km)

# plot silhouette diagram
fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10
for i in range(10):
    ith_cluster_silhouette_values = sample_silhouette_values[labels_km == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.tab10(i / 10)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_title("Silhouette plot for the MNIST dataset")
ax.set_xlabel("Silhouette coefficient values")
ax.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg, color="red", linestyle="--")

# Add the legend
ax.legend(["Average silhouette score = {:.2f}".format(silhouette_avg)], loc="best")

plt.show()

# end silhouette plot
# Begin dendogram plot fo Agglomerative clustering
ac2 = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='complete', distance_threshold=0)
ac2_train_start_time = time.time()
y_ac = ac2.fit_predict(X)
training_time = time.time() - ac2_train_start_time
print(f'Agglomerative 2 Training time: {training_time}')

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Agglomerative Clustering Dendrogram")
plt.xlabel("Data points")
plt.tight_layout()
plot_dendrogram(ac2, truncate_mode="level", p=truncate_level)
plt.show()
# end dendogram plot Agglomerative clustering


# scipy hierarchical clustering 
# this code is from the scipy documentation 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

linkage_start_time = time.time()
linkage_matrix = linkage(X, method='average', metric='euclidean')
training_time = time.time() - linkage_start_time
print(f'Linkage Training time: {training_time}')

# plot the dendrogram
fig, ax = plt.subplots(figsize=(10, 6))

plt.title('Hierarchical Clustering Dendrogram (Euclidean)')
plt.xlabel('Data points') # uncomment this for datapoints on x axis -> too many points for readability with truncation
plt.ylabel('Distance')
plt.tight_layout()
dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=truncate_level)
plt.show()

# calculate the linkage matrix
Z = linkage(X, method='ward')

# plot the dendrogram
plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Datapoints')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='level', p=truncate_level)
plt.show()
