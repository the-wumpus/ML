# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# GABHS project

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

gabhs = pd.read_csv('C:\\Users\\xyzzy\\Desktop\\NMSU courses\\App ML\\project\\GABHS-2021.csv')

# Split the data into features (X) and target (y)
X = gabhs.iloc[:, :-1].values
y = gabhs.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Class labels:', np.unique(y))
print('Features:', np.unique(X))

# verifying the train/test split via bincount
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Create the logistic regression model
# model = LogisticRegression(C=100, solver='lbfgs', multi_class='multinomial',  max_iter=200)
model = LogisticRegression(C=100, solver='lbfgs', multi_class='ovr',max_iter=200)



# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

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

# Calculate the learning rate
learning_rate = model.score(X_test, y_test)
print('Learning Rate:', learning_rate)

# Calculate the number of misclassifications
misclassifications = (y_test != y_pred).sum()
print('Misclassifications:', misclassifications)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


