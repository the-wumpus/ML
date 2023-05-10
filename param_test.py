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


# model parameters
truncate_level = 5 # dendrogram truncate level
clusters = 10
iterations = 1000

# Begin kmeans analysis
km = KMeans(n_clusters=clusters, init='k-means++', n_init=500, max_iter=iterations, random_state=0)
km_train_start_time = time.time()
labels_km = km.fit_predict(X)
training_time = time.time() - km_train_start_time
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