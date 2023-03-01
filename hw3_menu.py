# handle Classification model files
import pandas as pd
import numpy as np
from pathlib import Path

def run_file(file_name):
    with open(file_name) as f:
        exec(f.read())

def print_menu():
    print("\nChoose a dataset for testing")
    print("1. Run models on Digits dataset")
    print("2. Run models on the Wisconsin Breast Cancer dataset")
    print("3. Quit")
    

while True:
    print_menu()
    choice = int(input("Enter your choice [1-3]: "))
    if choice == 1:
        # Do something for option 1
        print("Loading Digits dataset...")
        run_file("hw3_digits.py")

    elif choice == 2:
        # Do something for option 2
        print("Loading Breast Cancer dataset...")
        run_file("hw3_breast_cancer.py")

    elif choice == 3:
        # Do something for option 5
        print("Yes, I'm done.")
        quit()
        
    else:
        print("Invalid choice. Try again.")
