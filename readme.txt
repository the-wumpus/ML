Thomas Ehret
CS519 Applied ML
Dr Cao
NMSU Sp23
Homework 2

This HW project uses a menu file for ease of use and to prevent errors, 
particulary because my python code is not error resilient.

Instructions:
- invoke slnnMenu.py
- choose options
- plot images run sequentially and will need to be closed for next to display
- menu file should gracefully exit terminal

models *.py: perceptron, adalineGD, adalineSGD
handler functions: 
snnPlot, slnnMenu.py, irisMain.py, irisMulti.py, digitsOVR.py, habermanMain.py
datasets: 
iris.data 4 features 3 classes
haberman.data 3 features 2 classes w/ overlapping features,
digits.data 8 features 10 labels  

	 Source Program flow <=-----------------
          ----slnnMenu.py----			|
         /        |          \			|
irisMain.py      OVR     habermanMain.py	|
    |             |               |		|
    V	          V               V             |
models      digits & iris       models		|
    |		|		 |		|
    V___________V________________V		|		
                |				|
		V				|	
            snnPlot----------------------------
