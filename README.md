# Optimization-based-on-GNI-Index-For-multi-objective-bandits
This project is mainly used to reproduce the result in the paper of "Multi-objective Bandits: Optimizing the Generalized Gini Index"

The code should be run in anaconda 3 and the package of cvxopt is needed. 

In order to draw figures similar to Fig 2 in the paper, some parameters in the file of LearningML.py should be revised, the list of these parameters is listed as follows:

1. K the number of arms
2. D the number of dimentions 
3. learn_Rate Learning_rate 
4. Iteration_num number of iterations in MO-OGDE and MO-LP 
