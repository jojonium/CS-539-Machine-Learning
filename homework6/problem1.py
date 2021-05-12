import numpy as np
from tree import DecisionTree
from collections import Counter
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Bootstrap Aggregation (Bagging Method) (50 points)
    In this problem, we will implement our first ensemble method: Bootstrap Aggregation (Bagging) of decision trees
    A list of all variables being used in this problem is provided at the end of this file.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your Dropbox automatically copied your solution from your desktop computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other students about this homework, only discuss high-level ideas or use pseudo-code. Don't discuss about the solution at the code level. For example, two students discuss about the solution of a function (which needs 5 lines of code to solve) and they then work on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences (variable names are different). In this case, the two students violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Note: we may use the Stanford Moss system to check your code for code similarity. https://theory.stanford.edu/~aiken/moss/
      Historical Data: in one year, we ended up finding 25% of the students in that class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #*******************************************
    # CHANGE HERE: if you have read and agree with the term above, change "False" to "True".
    Read_and_Agree = False
    #*******************************************
    return Read_and_Agree

#----------------------------------------------------
'''
    (Bootstrap Sampling) Given a dataset of features (X) and labels (Y), create a bootstrap sample of the same size from the dataset, by sampling WITH replacement (the same data record can be sampled twice). 
    ---- Inputs: --------
        * X: the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample.
        * Y: the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string.
    ---- Outputs: --------
        * Xs: the feature values of a bootstrap sample of the dataset, a numpy matrix of shape p by n. Xs[i,j] is the feature value of the i-th feature on the j-th data sample of the bootstrap.
        * Ys: the class labels of a bootstrap sample of the dataset, a numpy array of length n. Ys[i] is the class label of the i-th data sample of the bootstrap, which can be an int/float/string.
    ---- Hints: --------
        * You could use choice() function in numpy to generate random indices with or without replacement. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def bootstrap(X, Y):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    
    #########################################
    return Xs, Ys
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_bootstrap
        --- OR ---- 
        python3 -m nose -v test1.py:test_bootstrap
        --- OR ---- 
        python -m nose -v test1.py:test_bootstrap
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Build a Bagging Ensemble) Given a dataset of features (X) and labels (Y), create a bagging ensemble of multiple decision trees. 
    ---- Inputs: --------
        * X: the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample.
        * Y: the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string.
        * n_trees: the size of the ensemble (number of decision trees), an integer scalar.
    ---- Outputs: --------
        * Ts: the bagging ensemble of decision trees, each tree is trained on a bootstrap sample of the dataset, Ts[i] is the i-th decision tree.
    ---- Hints: --------
        * (Step 1): create a bootstrap sample from the dataset. 
        * (Step 2): use the bootstrap sample to train a decision tree and add it to the ensemble. 
        * You could use DecisionTree() function in tree.py to create a decision tree. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def bagging(X, Y, n_trees):
    Ts=[] # create an empty list to store the tree ensemble
    for _ in range(n_trees): # create one tree at a time
        pass # ignore this line
        #########################################
        ## INSERT YOUR CODE HERE (10 points)
    
        #########################################
    return Ts
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_bagging
        --- OR ---- 
        python3 -m nose -v test1.py:test_bagging
        --- OR ---- 
        python -m nose -v test1.py:test_bagging
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Predict Label of one data sample with Ensemble) Given a Bagging ensemble of decision trees (Ts), predict the label of one data sample (x) using majority vote by the trees. 
    ---- Inputs: --------
        * Ts: the bagging ensemble of decision trees, each tree is trained on a bootstrap sample of the dataset, Ts[i] is the i-th decision tree.
        * x: the feature values of one data instance, a numpy vector of length p. Xs[i] is the feature value of the i-th feature on the data instance.
    ---- Outputs: --------
        * y: the class labels of one data instance, a scalar of int/float/string.
    ---- Hints: --------
        * You could use predict_1() function in each decision tree to predict the label of one data sample. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def predict_1(Ts, x):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    
    #########################################
    return y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_predict_1
        --- OR ---- 
        python3 -m nose -v test1.py:test_predict_1
        --- OR ---- 
        python -m nose -v test1.py:test_predict_1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Predict Labels of all data samples) Given a Bagging ensemble of decision trees (Ts), predict the labels of all data samples (X) using majority vote by the trees. 
    ---- Inputs: --------
        * Ts: the bagging ensemble of decision trees, each tree is trained on a bootstrap sample of the dataset, Ts[i] is the i-th decision tree.
        * X: the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample.
    ---- Outputs: --------
        * Y: the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict(Ts, X):
    #########################################
    ## INSERT YOUR CODE HERE (20 points)
    
    #########################################
    return Y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_predict
        --- OR ---- 
        python3 -m nose -v test1.py:test_predict
        --- OR ---- 
        python -m nose -v test1.py:test_predict
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 1: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py
        --- OR ---- 
        python3 -m nose -v test1.py
        --- OR ---- 
        python -m nose -v test1.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 1 (50 points in total)--------------------- ... ok
        * (10 points) bootstrap ... ok
        * (10 points) bagging ... ok
        * (10 points) predict_1 ... ok
        * (20 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 4 tests in 0.586s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of samples in the training set, an integer scalar. 
* p:  the number of features in each sample, an integer scalar. 
* n_trees:  the size of the ensemble (number of decision trees), an integer scalar. 
* X:  the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample. 
* Y:  the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string. 
* Xs:  the feature values of a bootstrap sample of the dataset, a numpy matrix of shape p by n. Xs[i,j] is the feature value of the i-th feature on the j-th data sample of the bootstrap. 
* Ys:  the class labels of a bootstrap sample of the dataset, a numpy array of length n. Ys[i] is the class label of the i-th data sample of the bootstrap, which can be an int/float/string. 
* Ts:  the bagging ensemble of decision trees, each tree is trained on a bootstrap sample of the dataset, Ts[i] is the i-th decision tree. 
* x:  the feature values of one data instance, a numpy vector of length p. Xs[i] is the feature value of the i-th feature on the data instance. 
* y:  the class labels of one data instance, a scalar of int/float/string. 

'''
#--------------------------------------------