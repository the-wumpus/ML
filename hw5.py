# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# Homework 5

import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, Ridge, ElasticNet
from sklearn.datasets import fetch_california_housing


def LinearTest():

    # Load the datasets and display the class labels
    print("Loading the Cal Housing dataset...")
    cal_housing = fetch_california_housing()
    X = cal_housing.data
    y = cal_housing.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("80:20 train test split")

    # standardize the features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Legend for Metrics
    print("Legend")
    print("MSE: Mean Squared Error => 0.0 is best")
    print("R2: R Squared => 1.0 is best")
    print("MedAE: Median Absolute Error => 0.0 is best")

    # set testing parameters
    param_alpha = 0.5

    # Train RANSAC classifier 
    ransac = RANSACRegressor(random_state=42, min_samples = 0.95, residual_threshold = None)
    fit_start_time = time.time()
    ransac.fit(X_train_std, y_train)
    ransac_fit_time = time.time() - fit_start_time
    print(f"\nRANSAC fit time: {ransac_fit_time:.3f} seconds")

    # Make predictions on the test set
    y_pred = ransac.predict(X_test_std)

    # Calculate the accuracy of the classifier to 2 decmial places
    print(f"RANSAC MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RANSAC R2: {r2_score(y_test, y_pred):.2f}")
    print(f"RANSAC MedAE: {median_absolute_error(y_test, y_pred):.2f}")
    # End RANSAC

    # Train Lasso classifier
    lasso = Lasso(alpha=param_alpha, random_state=42)
    fit_start_time = time.time()
    lasso.fit(X_train_std, y_train)
    lasso_fit_time = time.time() - fit_start_time
    print(f"\nLasso fit time: {lasso_fit_time:.3f} seconds")

    # Make predictions on the test set
    y_pred = lasso.predict(X_test_std)

    # Calculate the accuracy of the classifier to 2 decmial places
    print(f"Lasso MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Lasso R2: {r2_score(y_test, y_pred):.2f}")
    print(f"Lasso MedAE: {median_absolute_error(y_test, y_pred):.2f}")
    # End Lasso

    # Train Ridge classifier
    ridge = Ridge(alpha=param_alpha, random_state=42)
    fit_start_time = time.time()
    ridge.fit(X_train_std, y_train)
    ridge_fit_time = time.time() - fit_start_time
    print(f"\nRidge fit time: {ridge_fit_time:.3f} seconds")

    # Make predictions on the test set
    y_pred = ridge.predict(X_test_std)

    # Calculate the accuracy of the classifier to 2 decmial places
    print(f"Ridge MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Ridge R2: {r2_score(y_test, y_pred):.2f}")
    print(f"Ridge MedAE: {median_absolute_error(y_test, y_pred):.2f}")
    # End Ridge

    # Train ElasticNet classifier
    elasticnet = ElasticNet(alpha=param_alpha, random_state=42)
    fit_start_time = time.time()
    elasticnet.fit(X_train_std, y_train)
    elasticnet_fit_time = time.time() - fit_start_time
    print(f"\nElasticNet fit time: {elasticnet_fit_time:.3f} seconds")

    # Make predictions on the test set
    y_pred = elasticnet.predict(X_test_std)

    # Calculate the accuracy of the classifier to 2 decmial places
    print(f"ElasticNet MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"ElasticNet R2: {r2_score(y_test, y_pred):.2f}")
    print(f"ElasticNet MedAE: {median_absolute_error(y_test, y_pred):.2f}")
    # End ElasticNet

    # Train Linear Regression classifier
    linear = LinearRegression()
    fit_start_time = time.time()
    linear.fit(X_train_std, y_train)
    linear_fit_time = time.time() - fit_start_time
    print(f"\nLinear Regression fit time: {linear_fit_time:.3f} seconds")

    # Make predictions on the test set
    y_pred = linear.predict(X_test_std)

    # Calculate the accuracy of the classifier to 2 decmial places
    print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Linear Regression R2: {r2_score(y_test, y_pred):.2f}")
    print(f"Linear Regression MedAE: {median_absolute_error(y_test, y_pred):.2f}")
    # End Linear Regression
    
def NonLinearTest():
    from sklearn.tree import DecisionTreeRegressor
    # Load the datasets and display the class labels
    print("Loading the Cal Housing dataset...")
    cal_housing = fetch_california_housing()
    X = cal_housing.data
    y = cal_housing.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("80:20 train test split")

    # Legend for Metrics
    print("Legend")
    print("MSE: Mean Squared Error => 0.0 is best")
    print("R2: R Squared => 1.0 is best")
    print("MedAE: Median Absolute Error => 0.0 is best")

    # Train NonLinear DT classifier 
    #non_linear = LinearRegression()
    non_linear = DecisionTreeRegressor(random_state=123, min_samples_split=25, criterion = "absolute_error" )
    fit_start_time = time.time()
    non_linear.fit(X_train, y_train)
    non_linear_fit_time = time.time() - fit_start_time
    print(f"\nNon-Linear DT Model fit time: {non_linear_fit_time:.3f} seconds")

    # Make predictions on the test set
    # Predict the target variable on the training and testing sets
    y_test_pred = non_linear.predict(X_test)

    # Calculate the accuracy of the classifier to 2 decmial places
    print(f"Decision Tree Regressor MSE: {mean_squared_error(y_test, y_test_pred):.2f}")
    print(f"Decision Tree Regressor R2: {r2_score(y_test, y_test_pred):.2f}")
    print(f"Decision Tree Regressor MedAE: {median_absolute_error(y_test, y_test_pred):.2f}")
    # End NonLinear regression test

def print_menu():
    print("\nChoose a models for testing")
    print("1. Run Linear Regression, RANSAC, Ridge, Lasso, and ElasticNet on the Housing dataset")
    print("2. Run a Decision Tree Regressor on the Housing dataset")
    print("3. Visualize the Housing dataset")
    print("4. Quit")
    
def visualize():
    cal_housing = fetch_california_housing()
    X = cal_housing.data
    y = cal_housing.target
    # visualize the dataset using scatterplotmatrix and heatmap
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    print(df.columns)
    print("shape of dataset: ", df.shape)
    scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
    plt.tight_layout()
    plt.show()
    cm = np.corrcoef(df.values.T)
    hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
    plt.tight_layout()
    plt.show()

while True:
    print_menu()
    choice = int(input("Enter your choice [1-4]: "))
    if choice == 1:
        # Do something for option 1
        print("Running Linear Models...")
        LinearTest()

    elif choice == 2:
        # Do something for option 2
        print("LRunning Non-Linear Model...")
        NonLinearTest()

    elif choice == 3:
        # Do something for option 3
        print("Visualizing the dataset...")
        visualize()

    elif choice == 4:
        # Do something for option 4
        print("Yes, I'm done.")
        quit()
    
       
    else:
        print("Invalid choice. Try again.")
