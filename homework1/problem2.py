import numpy as np

# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Support Vector Machine (with Linear Kernel) 
    In this problem, you will implement the SVM classification method.  We will optimize the parameters using gradient descent method. 
    A list of all variables being used in this problem is provided at the end of this file.
'''

#---------------------------------------------------
'''
    compute the f(x) of the linear model on one data instance x. 
    Inputs: 
        * x: the feature vector of one training data sample, a numpy vector of length p.
        * w: the weights of the SVM model, a numpy float vector of length p.
        * b: the bias of the SVM model, a float scalar.
    Outputs: 
        * fx: f(x), the output of linear model on the data instance (x), a float scalar.
    --------------------------------------
    Example:
        suppose we have a two dimensional feature space (p=2).
        If x is a data sample with 2 features:
        x = [1, 2]
        So now we want to compute the f(x) = w*x +b in the linear model.
        Suppose w = [0.1, 0.2] and b = -0.5
        The result f(x) = w*x + b =  0.1*1 + 0.2*2 - 0.5 = 0 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_fx(x, w, b):
    #########################################
    ## INSERT YOUR CODE HERE
    fx = sum(a * b for [a, b] in zip(x, w)) + b
    #########################################
    return fx
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_fx
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    compute the g(x) of the linear model on one data instance x. 
    Inputs: 
        * x: the feature vector of one training data sample, a numpy vector of length p.
        * w: the weights of the SVM model, a numpy float vector of length p.
        * b: the bias of the SVM model, a float scalar.
    Outputs: 
        * gx: g(x), the predicted label (-1 or 1) of SVM model on the data instance (x), a float scalar.
    --------------------------------------
    Example:
        suppose we have a two dimensional feature space (p=2).
        If x is a data sample with 2 features:
        x = [1, 2]
        So now we want to compute the f(x) = w*x +b in the linear model.
        Suppose w = [0.1, 0.2] and b = -0.4
        f(x) = w*x + b =  0.1*1 + 0.2*2 - 0.4 = 0.1
        The result g(x) = sign(f(x))= sign(0.1) = 1 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_gx(x, w, b):
    #########################################
    ## INSERT YOUR CODE HERE
    gx = 1 if (compute_fx(x, w, b) >= 0) else -1
    #########################################
    return gx
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_gx
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the gradient of loss function of SVM w.r.t. w and b (on one training instance). 
    Inputs: 
        * x: the feature vector of one training data sample, a numpy vector of length p.
        * y: the label of one training data sample, a float scalar (1 or -1).
        * w: the weights of the SVM model, a numpy float vector of length p.
        * b: the bias of the SVM model, a float scalar.
        * l: short for lambda = 1/ (n C), which is the weight of the L2 regularization term.
    Outputs: 
        * dL_dw: the gradient of the weights, a numpy float vector of length p. The i-th element is  d L / d w[i].
        * dL_db: the gradient of the bias, a float scalar.
    Hints: 
        * This problem can be solved using 6 line(s) of code.
'''
#---------------------
def compute_gradient(x, y, w, b, l=0.01):
    #########################################
    ## INSERT YOUR CODE HERE
    if 1 - y * compute_fx(x, w, b) > 0:
        dL_dw = l * w - y * x
        dL_db = -y
    else:
        dL_dw = l * w
        dL_db = 0
    #########################################
    return dL_dw, dL_db
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_gradient
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Update the parameter w using the gradient descent. 
    Inputs: 
        * w: the weights of the SVM model, a numpy float vector of length p.
        * dL_dw: the gradient of the weights, a numpy float vector of length p. The i-th element is  d L / d w[i].
        * lr: the learning rate, a float scalar, controlling the speed of gradient descent.
    Outputs: 
        * w: the weights of the SVM model, a numpy float vector of length p.
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_w(w, dL_dw, lr=0.01):
    #########################################
    ## INSERT YOUR CODE HERE
    w = w - lr * dL_dw
    #########################################
    return w
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_update_w
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Update the parameter b using the gradient descent. 
    Inputs: 
        * b: the bias of the SVM model, a float scalar.
        * dL_db: the gradient of the bias, a float scalar.
        * lr: the learning rate, a float scalar, controlling the speed of gradient descent.
    Outputs: 
        * b: the bias of the SVM model, a float scalar.
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_b(b, dL_db, lr=0.01):
    #########################################
    ## INSERT YOUR CODE HERE
    b = b - lr * dL_db
    #########################################
    return b
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_update_b
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Train the SVM model using Stochastic Gradient Descent (SGD). 
    Inputs: 
        * X: the feature matrix of all data samples, a numpy matrix of shape n by p.
        * Y: the labels of all data samples, a numpy vector of length n. If the i-th instance is positive labeled, Y[i]= 1, otherwise -1.
        * lr: the learning rate, a float scalar, controlling the speed of gradient descent.
        * C: the weight of the hinge loss.
        * n_epoch: the number of rounds to iterate through all training examples.
    Outputs: 
        * w: the weights of the SVM model, a numpy float vector of length p.
        * b: the bias of the SVM model, a float scalar.
    Hints: 
        * You could use the functions that you have implemented above to build the solution. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def train(X, Y, lr=0.01, C=1.0, n_epoch=10):
    n,p = X.shape
    l = 1./(n * C) #l is the weight of the L2 regularization term. 
    w,b = np.random.rand(p), 0. # initialize the weight with a random vector and bias with 0
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        indices = np.random.permutation(n) # shuffle the indices of all instances
        for i in indices: # iterate through each random instance (x,y)
            x=X[i] # the feature vector of the i-th random instance
            y=Y[i] # the label of the i-th random instance
            #########################################
            ## INSERT YOUR CODE HERE
            dL_dw, dL_db = compute_gradient(x, y, w, b, l)
            w = update_w(w, dL_dw, lr)
            b = update_b(b, dL_db, lr)
            #########################################
    return w, b
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_train
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Predict the labels of a set of data instances. Suppose the weights w and bias b of the SVM are given, use the model to predict the labels of all the instances in matrix X. 
    Inputs: 
        * X: the feature matrix of all data samples, a numpy matrix of shape n by p.
        * w: the weights of the SVM model, a numpy float vector of length p.
        * b: the bias of the SVM model, a float scalar.
    Outputs: 
        * Y: the labels of all data samples, a numpy vector of length n. If the i-th instance is positive labeled, Y[i]= 1, otherwise -1.
    Hints: 
        * You could use np.where() to simplify the code. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict(X, w, b):
    #########################################
    ## INSERT YOUR CODE HERE
    Y = np.where(np.dot(X, w) + b > 0, 1, -1)
    #########################################
    return Y
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_predict
        ---------------------------------------------------
    '''
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_svm
        ---------------------------------------------------
    '''
    

#--------------------------------------------

''' 
    TEST problem 2: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 2 (30 points in total)--------------------- ... ok
        * (1 point) compute_fx ... ok 
        * (1 points) compute_gx ... ok 
        * (5 points) compute_gradient ... ok 
        * (5 points) update_w ... ok 
        * (5 points) update_b ... ok 
        * (5 point) train ... ok 
        * (3 points) predict ... ok 
        * (5 point) SVM ... ok 
        ----------------------------------------------------------------------
        Ran 8 tests in 0.959s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* p:  the number of dimensions in the feature space. 
* n:  the number of data instances. 
* X:  the feature matrix of all data samples, a numpy matrix of shape n by p. 
* Y:  the labels of all data samples, a numpy vector of length n. If the i-th instance is positive labeled, Y[i]= 1, otherwise -1. 
* w:  the weights of the SVM model, a numpy float vector of length p. 
* b:  the bias of the SVM model, a float scalar. 
* x:  the feature vector of one training data sample, a numpy vector of length p. 
* y:  the label of one training data sample, a float scalar (1 or -1). 
* fx:  f(x), the output of linear model on the data instance (x), a float scalar. 
* gx:  g(x), the predicted label (-1 or 1) of SVM model on the data instance (x), a float scalar. 
* C:  the weight of the hinge loss. 
* l:  short for lambda = 1/ (n C), which is the weight of the L2 regularization term. 
* dL_dw:  the gradient of the weights, a numpy float vector of length p. The i-th element is  d L / d w[i]. 
* dL_db:  the gradient of the bias, a float scalar. 
* lr:  the learning rate, a float scalar, controlling the speed of gradient descent. 
* n_epoch:  the number of rounds to iterate through all training examples. 

'''
#--------------------------------------------