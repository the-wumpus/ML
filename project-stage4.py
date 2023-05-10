# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# GABHS project stage 4

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
# from sklearn.linear_model import LogisticRegression

gabhs = pd.read_csv('GABHS-2021.csv')



# Split the data into features (X) and target (y)
X = gabhs.iloc[:, :-1].values
y = gabhs.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=1)
print('Class labels:', np.unique(y))
print('Features:', np.unique(X))

# verifying the train/test split via bincount
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

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
# end of PCA 

# Parameterize the classification models
# multiple models are used to compare the results
# LR = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr',max_iter=200)
DT = DecisionTreeClassifier(criterion='gini', splitter = "best", random_state=1)

# Fit the models to the training data
# LR.fit(X_train, y_train)
DT.fit(X_train_pca, y_train)

# Make predictions on the testing data
# y_pred = LR.predict(X_test)
y_pred = DT.predict(X_test_pca)

# end of decision tree model training

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Positive', 'Negative'])
plt.yticks([0, 1], ['Positive', 'Negative'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot the decision tree
plt.figure(figsize=(10,10))
plot_tree(DT, filled=True, rounded=True)
plt.show()  

# Calculate the learning rates

# learning_rate = LR.score(X_test, y_test)
# print('Learning Rate:', learning_rate)

# Calculate the number of misclassifications

# LR_misclassifications = (y_test != y_pred).sum()
# print('LR Misclassifications:', LR_misclassifications)

DT_misclassifications = (y_test != y_pred).sum()
print('DT Misclassifications:', DT_misclassifications)

# Calculate the accuracy of the model
# LR_accuracy = accuracy_score(y_test, y_pred)
# print('LR Accuracy:', LR_accuracy)

DT_accuracy = accuracy_score(y_test, y_pred)
print('DT Accuracy:', DT_accuracy)


# check the decision tree validation curve for different max_depth values
param_range = np.arange(1, 11)

# Compute the cross-validation scores for different max_depth values
max_depth = 4
train_scores, test_scores = [], []
for max_depth in param_range:
    DT.set_params(max_depth=max_depth)
    train_scores.append(cross_val_score(DT, X_train, y_train, cv=5, scoring='accuracy').mean())
    test_scores.append(cross_val_score(DT, X_test, y_test, cv=5, scoring='accuracy').mean())

# Plot the cross-validation curve
plt.figure()
plt.plot(param_range, train_scores, 'o-', label="Training score",
         color="darkorange", lw=2)
plt.plot(param_range, test_scores, 'o-', label="Cross-validation score",
         color="navy", lw=2)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()


