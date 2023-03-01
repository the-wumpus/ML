Thomas Ehret
CS519 Applied ML
Dr Cao
NMSU Sp23
Homework 3

This HW project uses a menu file for ease of use and to prevent errors, 
particulary because my python code is not error resilient.

Instructions:
- invoke hw3_menu.py
- choose options
- menu file should gracefully exit terminal

models: LGR, SVM, KNN, DT, RBF, PNN
datasets loaded from sklearn dataset function
datasets: 
digits dataset 64 features 10 classes
wisconsin breast cancer dataset 30 features 2 classes
Decision Tree images are saved to files:
default parameters dt_digits.png dt_breast_cancer.png
modified parameters dt_digits2.png dt_breast_cancer2.png

 

	    Source Program flow 
          ----hw3_menu.py----			
         /                  \			
hw3_digits.py           hw3_breast_cancer.py	
        |                        |
        -------------------------
                   |
            return or exit


