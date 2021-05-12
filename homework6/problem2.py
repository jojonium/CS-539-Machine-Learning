import numpy as np
from collections import Counter
from tree import DecisionTree
from problem1 import bootstrap
import problem1 as p1
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Random Forest (50 points)
    In this problem, we will implement our second ensemble method: Random Forest
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Sampling a subset of features) Given a bootstrap sampled dataset (Xs), create a set of randomly sampled features (m features), by sampling without replacement on the features (the same feature cannot be sampled twice). 
    ---- Inputs: --------
        * Xs: the feature values of a bootstrap sample of the dataset (X), a numpy matrix of shape p by n. Xs[i,j] is the feature value of the i-th feature on the j-th data sample of the bootstrap.
        * m: the number of features to be sampled for each tree in random forest, an integer scalar.
    ---- Outputs: --------
        * Xf: the sampled feature values of the bootstrap dataset (Xs), a numpy matrix of shape m by n. Xf[i,j] is the feature value of the i-th random feature on the j-th data sample of the bootstrap.
        * fid: the indices of the sampled features, a numpy vector of length m. fid[i] is the index of the i-th random feature, for example, if we sample two features (the third and the first feature) from Xs, the fid should be [2,0].
    ---- Hints: --------
        * You could use choice() function in numpy to generate random indices with or without replacement. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def sample_features(Xs, m):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    
    #########################################
    return Xf, fid
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_sample_features
        --- OR ---- 
        python3 -m nose -v test2.py:test_sample_features
        --- OR ---- 
        python -m nose -v test2.py:test_sample_features
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Build a Random Forest) Given a dataset of features (X) and labels (Y), create a random forest of multiple decision trees. In each decision tree, we first sample a bootstrap of the dataset (Xs) and a subset of features (Xf) randomly sampled from Xs, and then the decision tree is trained on Xf. 
    ---- Inputs: --------
        * X: the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample.
        * Y: the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string.
        * n_trees: the size of the ensemble (number of decision trees), an integer scalar.
        * m: the number of features to be sampled for each tree in random forest, an integer scalar.
    ---- Outputs: --------
        * Ts: the ensemble of random forest, each tree is trained on a bootstrap sample of the dataset with a set of random features, Ts[i] is the i-th decision tree in the random forest.
        * Fs: the feature IDs of random forest, an integer matrix of shape m by p. Fs[i] is the list of feature indices (fid) sampled for the i-th decision tree in the random forest. Fs[i,j] is the index of the j-th sampled feature for the i-th decision tree.
    ---- Hints: --------
        * (Step 1): create a bootstrap sample (Xs) from the dataset (X). 
        * (Step 2): sample a subset of features (Xf) on the bootstrap samples (Xs). 
        * (Step 3): use the sampled features Xf to train a decision tree to get the tree (t) and feature ID list (fid). 
        * (Step 4): add the tree (t) to the ensemble (Ts) and add feature IDs (fid) to the feature IDs (Fs). 
        * You could use DecisionTree() function in tree.py to create a decision tree. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def random_forest(X, Y, n_trees, m):
    Ts=[] # create an empty list to store the tree ensemble
    Fs= np.empty((n_trees,m),dtype=int) # create an empty matrix to store a list of feature ids for each tree
    for i in range(n_trees): # create one tree at a time
        pass # ignore this line
        #########################################
        ## INSERT YOUR CODE HERE (10 points)
    
        #########################################
    return Ts, Fs
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_random_forest
        --- OR ---- 
        python3 -m nose -v test2.py:test_random_forest
        --- OR ---- 
        python -m nose -v test2.py:test_random_forest
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Predict Label of one data sample using Random Forest) Given a random forest (Ts and Fs), predict the label (y) of one data sample (x) using majority vote by the trees. 
    ---- Inputs: --------
        * Ts: the ensemble of random forest, each tree is trained on a bootstrap sample of the dataset with a set of random features, Ts[i] is the i-th decision tree in the random forest.
        * Fs: the feature IDs of random forest, an integer matrix of shape m by p. Fs[i] is the list of feature indices (fid) sampled for the i-th decision tree in the random forest. Fs[i,j] is the index of the j-th sampled feature for the i-th decision tree.
        * x: the feature values of one data instance, a numpy vector of length p. Xs[i] is the feature value of the i-th feature on the data instance.
    ---- Outputs: --------
        * y: the class labels of one data instance, a scalar of int/float/string.
    ---- Hints: --------
        * You could use predict_1() function in each decision tree to predict the label of one data sample. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def predict_1(Ts, Fs, x):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    
    #########################################
    return y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_predict_1
        --- OR ---- 
        python3 -m nose -v test2.py:test_predict_1
        --- OR ---- 
        python -m nose -v test2.py:test_predict_1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Predict Labels of multiple data samples using Random Forest) Given a random forest (Ts and Fs), predict the labels (Y) of all the data samples in (X) using majority vote by the trees. 
    ---- Inputs: --------
        * Ts: the ensemble of random forest, each tree is trained on a bootstrap sample of the dataset with a set of random features, Ts[i] is the i-th decision tree in the random forest.
        * Fs: the feature IDs of random forest, an integer matrix of shape m by p. Fs[i] is the list of feature indices (fid) sampled for the i-th decision tree in the random forest. Fs[i,j] is the index of the j-th sampled feature for the i-th decision tree.
        * X: the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample.
    ---- Outputs: --------
        * Y: the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string.
    ---- Hints: --------
        * You could use predict_1() function in each decision tree to predict the label of one data sample. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict(Ts, Fs, X):
    #########################################
    ## INSERT YOUR CODE HERE (20 points)
    
    #########################################
    return Y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_predict
        --- OR ---- 
        python3 -m nose -v test2.py:test_predict
        --- OR ---- 
        python -m nose -v test2.py:test_predict
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 2: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py
        --- OR ---- 
        python3 -m nose -v test2.py
        --- OR ---- 
        python -m nose -v test2.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 2 (50 points in total)--------------------- ... ok
        * (10 points) sample_features ... ok
        * (10 points) random_forest ... ok
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
* m:  the number of features to be sampled for each tree in random forest, an integer scalar. 
* n_trees:  the size of the ensemble (number of decision trees), an integer scalar. 
* X:  the feature values of a dataset of samples, a numpy matrix of shape p by n. X[i,j] is the feature value of the i-th feature on the j-th data sample. 
* Y:  the class labels of a dataset of samples, a numpy array of length n. Y[i] is the class label of the i-th data sample, which can be an int/float/string. 
* Xs:  the feature values of a bootstrap sample of the dataset (X), a numpy matrix of shape p by n. Xs[i,j] is the feature value of the i-th feature on the j-th data sample of the bootstrap. 
* Xf:  the sampled feature values of the bootstrap dataset (Xs), a numpy matrix of shape m by n. Xf[i,j] is the feature value of the i-th random feature on the j-th data sample of the bootstrap. 
* fid:  the indices of the sampled features, a numpy vector of length m. fid[i] is the index of the i-th random feature, for example, if we sample two features (the third and the first feature) from Xs, the fid should be [2,0]. 
* Ts:  the ensemble of random forest, each tree is trained on a bootstrap sample of the dataset with a set of random features, Ts[i] is the i-th decision tree in the random forest. 
* Fs:  the feature IDs of random forest, an integer matrix of shape m by p. Fs[i] is the list of feature indices (fid) sampled for the i-th decision tree in the random forest. Fs[i,j] is the index of the j-th sampled feature for the i-th decision tree. 
* x:  the feature values of one data instance, a numpy vector of length p. Xs[i] is the feature value of the i-th feature on the data instance. 
* y:  the class labels of one data instance, a scalar of int/float/string. 

'''
#--------------------------------------------