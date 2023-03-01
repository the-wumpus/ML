# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# Homework 3

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import utils for performance metrics
import time
import psutil
from sklearn.metrics import accuracy_score
# import models to test in 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the Breast Cancer  dataset and display the class labels and #features
# check-expect 30 (+R) with 2 labels (+-)
print("Loading the Wisconsin Breast Cancer dataset...")
digits = datasets.load_breast_cancer(return_X_y=False, as_frame=False)
X = digits.data
y = digits.target
print('Class labels:', np.unique(y))
#print('Features:', np.unique(X))


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train the model on the training set and measure the training using time function, and memory usage
memMb=psutil.Process().memory_info().rss / (1024 * 1024)

print ("Memory Usage %5.1f MByte" % (memMb))
# verifying the train/test split via bincount
print("Using 80:20 train/test split")
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



# Initialize the LR model, max_iter >= ~250 needed for convergence
#lr = LogisticRegression(C=100, solver='lbfgs', multi_class='multinomial',  max_iter=250)
lr = LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial',  max_iter=250)

#train the model
start_time = time.time()
lr.fit(X_train_std, y_train)
training_time = time.time() - start_time

# Make predictions on the training set and measure the accuracy
y_train_pred = lr.predict(X_train_std)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Make predictions on the testing set and measure the accuracy and testing time
start_time = time.time()
lr.fit(X_test_std, y_test)
y_test_pred = lr.predict(X_test_std)
testing_time = time.time() - start_time
testing_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy and time results
print("\nResults for Logistic Regression")
print(f"Training accuracy: {training_accuracy:.2f}")
print(f"Testing accuracy: {testing_accuracy:.2f}")
print(f"Training time: {training_time:.3f} seconds")
print(f"Testing time: {testing_time:.3f} seconds")

# generate classification metrics
y_pred=lr.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# END LR

# Initialize the SVM using RBF classifier
#svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=0.3)

#train the model
start_time = time.time()
svm.fit(X_train_std, y_train)
training_time = time.time() - start_time

# Make predictions on the training set and measure the accuracy
y_train_pred = svm.predict(X_train_std)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Make predictions on the testing set and measure the accuracy and testing time
start_time = time.time()
svm.fit(X_test_std, y_test)
y_test_pred = svm.predict(X_test_std)
testing_time = time.time() - start_time
testing_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy and time results
print("\nResults for SVM (RBF)")
print(f"Training accuracy: {training_accuracy:.2f}")
print(f"Testing accuracy: {testing_accuracy:.2f}")
print(f"Training time: {training_time:.3f} seconds")
print(f"Testing time: {testing_time:.3f} seconds")

# generate classification metrics
y_pred=svm.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# END SVM RBF

# Initialize the Decision Tree classifier
#tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1)

tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

#train the model
start_time = time.time()
tree_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions on the training set and measure the accuracy
y_train_pred = tree_model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

# Make predictions on the testing set and measure the accuracy and testing time
start_time = time.time()
tree_model.fit(X_test, y_test)
y_test_pred = tree_model.predict(X_test)
testing_time = time.time() - start_time
testing_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy and time results
print("\nResults for Decision Tree")
print(f"Training accuracy: {training_accuracy:.2f}")
print(f"Testing accuracy: {testing_accuracy:.2f}")
print(f"Training time: {training_time:.3f} seconds")
print(f"Testing time: {testing_time:.3f} seconds")

# generate classification metrics
y_pred=tree_model.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# Uncomment to plot the decision tree

#plt.figure(figsize=(20,20))
#plot_tree(tree_model, filled=True, rounded=True, feature_names=digits.feature_names)
#plt.show()  

# END Decision Tree



