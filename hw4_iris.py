# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# Homework 4

import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# import models and dimension reduction libraries
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA as KPCA
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the datasets and display the class labels
print("Loading the iris dataset...")
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# verifying the train/test split via bincount
print("Using 80:20 train/test split")
print('Instances in y:', np.bincount(y))
print('Instances in y_train:', np.bincount(y_train))
print('Instances in y_test:', np.bincount(y_test))

# standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# PCA testing
# Apply PCA to reduce the dimensions of the data
# measure the accuracy and testing time of PCA process
pca = PCA(n_components=2)
train_start_time = time.time()
X_train_pca = pca.fit_transform(X_train_std)
training_time = time.time() - train_start_time

# Transform the test set
transform_start_time = time.time()
X_test_pca = pca.transform(X_test_std)
transform_time = time.time() - transform_start_time

# Print the accuracy and time results
print(f"PCA fit time: {training_time:.3f} seconds")
print(f"PCA transform time: {transform_time:.3f} seconds")

# Train a decision tree classifier on the reduced dataset
dt = DecisionTreeClassifier()
fit_start_time = time.time()
dt.fit(X_train_pca, y_train)
dt_fit_time = time.time() - fit_start_time
print(f"Decision Tree fit time: {dt_fit_time:.3f} seconds")

# Make predictions on the test set
y_pred = dt.predict(X_test_pca)

# Calculate the accuracy of the classifier to 2 decmial places
accuracy = accuracy_score(y_test, y_pred)
formatted_accuracy = "{:.2f}".format(accuracy)
print(f"Accuracy: {formatted_accuracy }")
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# Plot the decision tree
plt.figure(figsize=(10,10))
plot_tree(dt, filled=True, rounded=True, feature_names=iris.feature_names)
plt.show()  
# end PCA testing

# LDA testing
# Apply LDA to reduce the dimensions of the data
# measure the accuracy and testing time of PCA process
lda = LDA(n_components=2)
train_start_time = time.time()
X_train_lda = lda.fit_transform(X_train_std, y_train)
training_time = time.time() - train_start_time

# Transform the test set
transform_start_time = time.time()
X_test_lda = lda.transform(X_test_std)
transform_time = time.time() - transform_start_time

# Print the accuracy and time results
print("\n")
print(f"LDA fit time: {training_time:.3f} seconds")
print(f"LDA transform time: {transform_time:.3f} seconds")

# Train a decision tree classifier on the reduced dataset
fit_start_time = time.time()
dt.fit(X_train_lda, y_train)
dt_fit_time = time.time() - fit_start_time
print(f"Decision Tree fit time: {dt_fit_time:.3f} seconds")

# Make predictions on the test set
y_pred = dt.predict(X_test_lda)

# Calculate the accuracy of the classifier to 2 decmial places
accuracy = accuracy_score(y_test, y_pred)
formatted_accuracy = "{:.2f}".format(accuracy)
print(f"Accuracy: {formatted_accuracy }")
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# Plot the decision tree
plt.figure(figsize=(10,10))
plot_tree(dt, filled=True, rounded=True, feature_names=iris.feature_names)
plt.show()  
# end LDA testing

# Kernel PCA testing
# Apply KPCA to reduce the dimensions of the data
# measure the accuracy and testing time of PCA process
# variations: kernels(poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’), gamma=3->15, n_components=None->2, eigen_solver='auto'
kpca = KPCA(n_components=2)
train_start_time = time.time()
X_train_kpca = kpca.fit_transform(X_train_std)
training_time = time.time() - train_start_time

# Transform the test set
transform_start_time = time.time()
X_test_kpca = kpca.transform(X_test_std)
transform_time = time.time() - transform_start_time

# Print the accuracy and time results
print("\n")
print(f"KPCA fit time: {training_time:.3f} seconds")
print(f"KPCA transform time: {transform_time:.3f} seconds")


# Train a decision tree classifier on the reduced dataset
fit_start_time = time.time()
dt.fit(X_train_kpca, y_train)
dt_fit_time = time.time() - fit_start_time
print(f"Decision Tree fit time: {dt_fit_time:.3f} seconds")

# Make predictions on the test set
y_pred = dt.predict(X_test_kpca)

# Calculate the accuracy of the classifier to 2 decmial places
accuracy = accuracy_score(y_test, y_pred)
formatted_accuracy = "{:.2f}".format(accuracy)
print(f"Accuracy: {formatted_accuracy }")
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# end Kernel PCA testing


