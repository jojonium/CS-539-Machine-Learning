import torch as th
import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Softmax Regression (with PyTorch) (12 points)
    In this problem, you will implement the softmax regression for multi-class classification problems.
The main goal of this problem is to get familiar with the PyTorch package for deep learning methods.

    -------------------------
    Package(s) to Install:
        Please install python version 3.7 or above and the following package(s):
        * torch (for building deep learning models)
    How to Install:
        * torch: To install 'torch' using pip, you could type in the terminal: 
            python3 -m pip install torch
    -------------------------
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
    Read_and_Agree = True
    #*******************************************
    return Read_and_Agree

#----------------------------------------------------
'''
    Given a softmax regression model with parameters W and b, please compute the linear logits z on a mini-batch of data samples x1, x2, ... x_batch_size. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the global gradients of the weights dL_dW and the biases dL_db in the PyTorch tensors. 
    ---- Inputs: --------
        * x: the feature vectors of a mini-batch of data samples, a float torch tensor of shape (batch_size, p).
        * W: the weight matrix of softmax regression, a float torch Tensor of shape (p by c).
        * b: the bias values of softmax regression, a float torch vector of length c.
    ---- Outputs: --------
        * z: the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c).
    ---- Hints: --------
        * When computing z values, in order to connect the global gradients dL_dz with dL_dW and dL_db, you may want to use the operators in PyTorch, instead of in NumPy or Python. For example, np.dot() is the numpy product of two numpy arrays, which will only compute the values z correctly, but cannot connect the global gradients of the torch tensors W and b. Instead, you may want to find the PyTorch version of dot product for two torch tensors. 
        * For PyTorch tensors, A@B represents the matrix multiplication between two torch matrices A and B. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z(x, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    z = x@W + b
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_compute_z
        --- OR ---- 
        python3 -m nose -v test1.py:test_compute_z
        --- OR ---- 
        python -m nose -v test1.py:test_compute_z
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Suppose we are given a softmax regression model and we have already computed the linear logits z on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average loss of the softmax regression model on the mini-batch of training samples. In the mean time, please also connect the global gradients of the linear logits z (dL_dz) with the loss L correctly. 
    ---- Inputs: --------
        * z: the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c).
        * y: the labels of a mini-batch of data samples, a torch integer vector of length batch_size. The value of each element can be 0,1,2, ..., or (c-1).
    ---- Outputs: --------
        * L: the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar.
    ---- Hints: --------
        * The loss L is a scalar, computed from the average of the cross entropy loss on all samples in the mini-batch. For example, if the loss on the four training samples are 0.1, 0.2, 0.3, 0.4, then the final loss L is the average of these numbers as (0.1+0.2+0.3+0.4)/4 = 0.25. 
        * You could use CrossEntropyLoss in PyTorch to compute the loss. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(z, y):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    L = th.nn.CrossEntropyLoss(reduction='mean')(z, y)
    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test1.py:test_compute_L
        --- OR ---- 
        python -m nose -v test1.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Gradient Descent) Suppose we are given a softmax regression model with parameters (W and b) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the weights W on the mini-batch of data samples. Assume that we have already created an optimizer for the parameter W and b. Please update the weights W and b using gradient descent. After the update, the global gradients of W and b should be set to all zeros. 
    ---- Inputs: --------
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (W and b).
    ---- Hints: --------
        * Although the parameters W and b are NOT given explicitly in the input of this function, but we can assume the W and b are already properly configured in the optimizer. So the optimizer is configured to handle the parameters W and b. 
        * Although the gradients of the parameters dL_dW and dL_db are NOT given explicitly in the input of this function, but we can assume that in the PyTorch tensors W and b, the gradients are already properly computed and are stored in W.grad (for dL_dW) and b.grad (for dL_db). 
        * Although the learning rate is NOT given explicitly in the input of this function, but we can assume that the optimizer was already configured with the learning rate parameter. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def update_parameters(optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    optimizer.step()
    optimizer.zero_grad()
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_update_parameters
        --- OR ---- 
        python3 -m nose -v test1.py:test_update_parameters
        --- OR ---- 
        python -m nose -v test1.py:test_update_parameters
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training Softmax Regression) Given a training dataset X (features), Y (labels) in a data loader, train the softmax regression model using mini-batch stochastic gradient descent: iteratively update the weights W and biases b using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples. 
    ---- Inputs: --------
        * data_loader: the PyTorch loader of a dataset.
        * c: the number of classes in the classification task, an integer scalar.
        * p: the number of input features.
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * W: the weight matrix of softmax regression, a float torch Tensor of shape (p by c).
        * b: the bias values of softmax regression, a float torch vector of length c.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits and loss. 
        * Step 2 Back propagation: compute the gradients of W and b. 
        * Step 3 Gradient descent: update the parameters W and b using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(data_loader, c, p, alpha=0.001, n_epoch=100):
    W = th.randn(p,c, requires_grad=True) # initialize W randomly using standard normal distribution
    b = th.zeros(c, requires_grad=True) # initialize b as all zeros
    optimizer = th.optim.SGD([W,b], lr=alpha) # SGD optimizer
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        for mini_batch in data_loader: # iterate through the dataset, with one mini-batch of random training samples (x,y) at a time
            x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
            y=mini_batch[1] # the labels of the samples in a mini-batch
            #########################################
            ## INSERT YOUR CODE HERE (2 points)
            update_parameters(optimizer)
            z = compute_z(x, W, b)
            L = compute_L(z, y)
            L.backward()
            #########################################
    return W, b
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_train
        --- OR ---- 
        python3 -m nose -v test1.py:test_train
        --- OR ---- 
        python -m nose -v test1.py:test_train
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using Softmax Regression)  Given a trained softmax regression model with parameters W and b. Suppose we have a mini-batch of test data samples. Please use the softmax regression model to predict the labels. 
    ---- Inputs: --------
        * x: the feature vectors of a mini-batch of data samples, a float torch tensor of shape (batch_size, p).
        * W: the weight matrix of softmax regression, a float torch Tensor of shape (p by c).
        * b: the bias values of softmax regression, a float torch vector of length c.
    ---- Outputs: --------
        * y_predict: the predicted labels of a mini-batch of test data samples, a torch integer vector of length batch_size. y_predict[i] represents the predicted label on the i-th test sample in the mini-batch.
    ---- Hints: --------
        * This is a multi-class classification task, for each sample, the label should be predicted as the index of the largest value of each row of the linear logit z. 
        * You could use the argmax() function in PyTorch to return the indices of the largest values in a tensor. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def predict(x, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    z = compute_z(x, W, b)
    y_predict = th.argmax(z, 1)
    #########################################
    return y_predict
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
        ----------- Problem 1 (12 points in total)--------------------- ... ok
        * (2 points) compute_z ... ok
        * (2 points) compute_L ... ok
        * (2 points) update_parameters ... ok
        * (2 points) train ... ok
        * (4 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 5 tests in 1.489s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of data instance in the training set. 
* p:  the number of input features. 
* c:  the number of classes in the classification task, an integer scalar. 
* batch_size:  the number of samples in a mini-batch, an integer scalar. 
* x:  the feature vectors of a mini-batch of data samples, a float torch tensor of shape (batch_size, p). 
* y:  the labels of a mini-batch of data samples, a torch integer vector of length batch_size. The value of each element can be 0,1,2, ..., or (c-1). 
* W:  the weight matrix of softmax regression, a float torch Tensor of shape (p by c). 
* b:  the bias values of softmax regression, a float torch vector of length c. 
* z:  the linear logits on a mini-batch of data samples, a float torch tensor of shape (batch_size, c). 
* a:  the softmax activations on a mini-batch of data samples, a float torch tensor of shape (batch_size, c). 
* L:  the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar. 
* data_loader:  the PyTorch loader of a dataset. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* y_predict:  the predicted labels of a mini-batch of test data samples, a torch integer vector of length batch_size. y_predict[i] represents the predicted label on the i-th test sample in the mini-batch. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (W and b). 

'''
#--------------------------------------------