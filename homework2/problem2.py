import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Softmax Regression (30 points)
    In this problem, you will implement the softmax regression for multi-class classification problems.
The main goal of this problem is to extend the logistic regression method to solving multi-class classification problems.
We will get familiar with computing gradients of vectors/matrices.
We will use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters.

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Value Forward Function 1) Given a softmax regression model with parameters W and b, please compute the linear logit z(x) on a data sample x. 
    ---- Inputs: --------
        * x: the feature vector of a data instance, a float numpy vector of length p.
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
        * b: the bias values of softmax regression, a float numpy vector of length c.
    ---- Outputs: --------
        * z: the linear logits, a float numpy vector of length c..
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z(x, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    z = W.dot(x) + b
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_z
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_z
        --- OR ---- 
        python -m nose -v test2.py:test_compute_z
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 1.1) Suppose we are given a softmax regression model with parameters W and b. Suppose we have already computed the linear logits z(x) on a training sample x.  Please compute partial gradients of the linear logits z(x) w.r.t. the biases b. 
    ---- Inputs: --------
        * c: the number of classes in the classification task, an integer scalar.
    ---- Outputs: --------
        * dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape c by c.  Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias b[j],   d_z[i] / d_b[j].
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dz_db(c):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    dz_db = np.diag(np.full(c, 1))
    #########################################
    return dz_db
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_dz_db
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_dz_db
        --- OR ---- 
        python -m nose -v test2.py:test_compute_dz_db
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradients 1.1) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the local gradient 1.1 (dz_db) and the global gradients of the loss L w.r.t. the linear logits z(x) (dL_dz). Please compute the partial gradients of the loss L w.r.t. biases b using chain rule. 
    ---- Inputs: --------
        * dL_dz: the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy vector of length c.  The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z[i],  d_L / d_z[i]..
        * dz_db: the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape c by c.  Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias b[j],   d_z[i] / d_b[j].
    ---- Outputs: --------
        * dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i].
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_db(dL_dz, dz_db):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    dL_db = dz_db.dot(dL_dz)
    #########################################
    return dL_db
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_dL_db
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_dL_db
        --- OR ---- 
        python -m nose -v test2.py:test_compute_dL_db
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 1.2) Suppose we are given a softmax regression model with parameters W and b. Suppose we have already computed the linear logits z(x) on a training sample x.  Please compute partial gradients of the linear logits z(x) w.r.t. the weights W. 
    ---- Inputs: --------
        * x: the feature vector of a data instance, a float numpy vector of length p.
        * c: the number of classes in the classification task, an integer scalar.
    ---- Outputs: --------
        * dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float tensor of shape (c by c by p).  The (i,j,k)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k],   d_z[i] / d_W[j,k].
    ---- Hints: --------
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def compute_dz_dW(x, c):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    dz_db: np.ndarray = np.diag(np.full(c, 1))
    dz_dW = [[x if j == 1 else np.zeros((len(x))) for j in i] for i in dz_db]
    #########################################
    return dz_dW
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_dz_dW
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_dz_dW
        --- OR ---- 
        python -m nose -v test2.py:test_compute_dz_dW
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 1.2) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the local gradients local gradient 1.2 (dz_dW) and the global gradients of the loss L w.r.t. the linear logits z(x) (dL_dz). Please compute the partial gradient of the loss L w.r.t. the weights W using chain rule. 
    ---- Inputs: --------
        * dL_dz: the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy vector of length c.  The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z[i],  d_L / d_z[i]..
        * dz_dW: the partial gradient of logits z w.r.t. the weight matrix W, a numpy float tensor of shape (c by c by p).  The (i,j,k)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k],   d_z[i] / d_W[j,k].
    ---- Outputs: --------
        * dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j].
    ---- Hints: --------
        * You could use np.tensordot(A,B) to compute the dot product of two tensors A and B. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_dW(dL_dz, dz_dW):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    dL_dW = np.tensordot(dL_dz, dz_dW, 1)
    #########################################
    return dL_dW
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_dL_dW
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_dL_dW
        --- OR ---- 
        python -m nose -v test2.py:test_compute_dL_dW
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Value Forward Function 2) Suppose we are given a softmax regression model and we have already computed the linear logits z(x) on a data sample x. Please compute the softmax activation on the data sample, i.e., a(x). 
    ---- Inputs: --------
        * z: the linear logits, a float numpy vector of length c..
    ---- Outputs: --------
        * a: the softmax activations, a float numpy vector of length c..
    ---- Hints: --------
        * You could use np.exp(x) to compute the element-wise exponentials of vector x. 
        * When computing exp(z), you need to be careful about overflowing cases. When an element of z (say z[i]) is a large number (say 1000),  the computer can no longer store the result of exp(z[i]) in a floating-point number. In this case, we may want to avoid computing exp(z) directly. Instead, you could find the largest value in z (say max_z) and subtract every element with max_z and then you could compute exp() on the vector (z-max_z) directly. The result will be correct, but will no longer suffer from overflow problems. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_a(z):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    zprime = (z - max(z)) if abs(max(z)) >= 1000 else z
    sigma = sum([np.exp(zi) for zi in zprime])
    a = np.exp(zprime) / sigma
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_a
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_a
        --- OR ---- 
        python -m nose -v test2.py:test_compute_a
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradients 2) Suppose we are given a softmax regression model and we have already computed the linear logits z(x) and activations a(x) on a training sample (x). Please compute the partial gradients of the softmax activations a(x) w.r.t. the linear logits z(x). 
    ---- Inputs: --------
        * a: the softmax activations, a float numpy vector of length c..
    ---- Outputs: --------
        * da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c).  The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] ).
    ---- Hints: --------
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def compute_da_dz(a):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return da_dz
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_da_dz
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_da_dz
        --- OR ---- 
        python -m nose -v test2.py:test_compute_da_dz
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradients 2) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the gradient of the loss L w.r.t. the activations a(x) and the partial gradients of activations a(x) w.r.t. the linear logits z(x). Please compute the partial gradients of the loss L w.r.t. the linear logits z(x) using chain rule. 
    ---- Inputs: --------
        * dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy vector of length c.  The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i],  d_L / d_a[i]..
        * da_dz: the partial gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c).  The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] ).
    ---- Outputs: --------
        * dL_dz: the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy vector of length c.  The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z[i],  d_L / d_z[i]..
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_dz(dL_da, da_dz):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return dL_dz
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_dL_dz
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_dL_dz
        --- OR ---- 
        python -m nose -v test2.py:test_compute_dL_dz
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Value Forward Function 3) Suppose we are given a softmax regression model and we have already computed the activations a(x) on a training sample x. Suppose the label of the training sample is y. Please compute the loss function of the softmax regression model on the training sample. 
    ---- Inputs: --------
        * a: the softmax activations, a float numpy vector of length c..
        * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
    ---- Outputs: --------
        * L: the multi-class cross entropy loss, a float scalar..
    ---- Hints: --------
        * You could use np.log(x) to compute the natural log of x. 
        * When computing log(a[i]), you need to be careful about a corner case where log(0) is not defined in math. Now the question is how can any activation (a) become 0? It is mathematically impossible, because the output of the softmax function (activation) should be 0<a[i]<1. However, in the above function (compute_a), when some linear logit z[i] is a number much larger than all the other elements in z (say z[j]), the activation of all the other elements (a[j]) will be very very small. Then the computer can no longer store these small numbers accurately in floating-point numbers. Instead, computer will store 0 as the activation a[i]. Then we have a problem in this function. We need to handle the specially case when a[j] = 0. To solve this problem, we need to avoid computing log(0) by assigning the final result of L directly. In this case, the log(a[j]) should be a very large negative number (say -10000...000 ), though it should not be negative infinity. So the loss L = -log(a[j]) should be a very large positive number. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(a, y):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_L
        --- OR ---- 
        python -m nose -v test2.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local/Global Gradient 3) Suppose we are given a softmax regression model and we have already computed the activations a(x) on a training sample x. Suppose the label of the training sample is y. Please compute the partial gradients of the loss function (L) w.r.t. the activations (a). 
    ---- Inputs: --------
        * a: the softmax activations, a float numpy vector of length c..
        * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
    ---- Outputs: --------
        * dL_da: the partial gradients of the loss function L w.r.t. the activations a, a float numpy vector of length c.  The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i],  d_L / d_a[i]..
    ---- Hints: --------
        * If you want to create an all-zero array of the same shape as x, you could use np.zeros_like(x) to create the all-zero matrix. 
        * When computing 1/a[i], you need to be careful about a corner case where 1/0 is not defined in math. Now the question is how can any of the activations (a) become 0? It is mathematically impossible, because the output of the softmax function (activations) should be 0<a[i]<1. However, in the compute_a() function, when some element of linear logits z (say z[i]) is much larger than all the other linear logits (say z[j]), the activations of all the other elements (a[j]) will be very very small. Then the computer can no longer store these small numbers accurately in floating-point numbers. Instead, computer will store 0 as the activation. Then we have a problem in this function. We need to handle the specially case when a[j] = 0.  To solve this problem, we need to avoid computing 1/a[j] by assigning the final result of 1/a[j] directly. In this case, when a[j] is a very small positive number (say exp(-900)), then 1/a[j] should be a very large positive number, though it should not be positive infinity. In this case, (-1/a[j]) will be a very large negative number (say -100000...000), though it should not be negative infinity. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_dL_da(a, y):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return dL_da
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_dL_da
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_dL_da
        --- OR ---- 
        python -m nose -v test2.py:test_compute_dL_da
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Forward Pass) Suppose we are given a softmax regression model with parameter W and b. Given a data sample (x), please compute the activations a(x) on the sample. 
    ---- Inputs: --------
        * x: the feature vector of a data instance, a float numpy vector of length p.
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
        * b: the bias values of softmax regression, a float numpy vector of length c.
    ---- Outputs: --------
        * a: the softmax activations, a float numpy vector of length c..
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def forward(x, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_forward
        --- OR ---- 
        python3 -m nose -v test2.py:test_forward
        --- OR ---- 
        python -m nose -v test2.py:test_forward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Back Propagation) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a(x) on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters W and b on the data sample using back propagation. 
    ---- Inputs: --------
        * x: the feature vector of a data instance, a float numpy vector of length p.
        * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        * a: the softmax activations, a float numpy vector of length c..
    ---- Outputs: --------
        * dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j].
        * dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i].
    ---- Hints: --------
        * Step 1: compute all the local gradients by re-using the above functions. 
        * Step 2: use the local gradients to build global gradients for the parameters W and b. 
        * This problem can be solved using 7 line(s) of code.
'''
#---------------------
def backward(x, y, a):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return dL_dW, dL_db
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_backward
        --- OR ---- 
        python3 -m nose -v test2.py:test_backward
        --- OR ---- 
        python -m nose -v test2.py:test_backward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Gradient Descent 1) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the partial gradients of the loss w.r.t. the biases b on the data sample. Please update the biases b using gradient descent. 
    ---- Inputs: --------
        * b: the bias values of softmax regression, a float numpy vector of length c.
        * dL_db: the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i].
        * alpha: the step-size parameter of gradient descent, a float scalar.
    ---- Outputs: --------
        * b: the bias values of softmax regression, a float numpy vector of length c.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_b(b, dL_db, alpha=0.001):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return b
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_update_b
        --- OR ---- 
        python3 -m nose -v test2.py:test_update_b
        --- OR ---- 
        python -m nose -v test2.py:test_update_b
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Gradient Descent 2) Suppose we are given a softmax regression model with parameters (W and b) and we have a training data sample (x,y).  Suppose we have already computed the partial gradients of the loss w.r.t. the weights W on the data sample. Please update the weights W using gradient descent. 
    ---- Inputs: --------
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
        * dL_dW: the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j].
        * alpha: the step-size parameter of gradient descent, a float scalar.
    ---- Outputs: --------
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_W(W, dL_dW, alpha=0.001):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return W
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_update_W
        --- OR ---- 
        python3 -m nose -v test2.py:test_update_W
        --- OR ---- 
        python -m nose -v test2.py:test_update_W
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training Softmax Regression) Given a training dataset X (features), Y (labels), train the softmax regression model using stochastic gradient descent: iteratively update the weights W and biases b using the gradients on each random data sample.  We repeat n_epoch passes over all the training instances. 
    ---- Inputs: --------
        * X: the feature matrix of training instances, a float numpy matrix of shape (n by p).
        * Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1.
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
        * b: the bias values of softmax regression, a float numpy vector of length c.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits, activations and loss. 
        * Step 2 Back propagation: compute the gradients of W and b. 
        * Step 3 Gradient descent: update the parameters W and b using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    n,p = X.shape # n: the number of training samples, p: the number of input features
    c = max(Y) + 1 # the number of classes
    W = np.random.randn(c,p) # initialize W randomly using standard normal distribution
    b= np.zeros(c) # initialize b as all zeros
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        indices = np.random.permutation(n) # shuffle the indices of all samples
        for i in indices: # iterate through each random training sample (x,y)
            x=X[i] # the feature vector of the i-th random sample
            y=Y[i] # the label of the i-th random sample
            #########################################
            ## INSERT YOUR CODE HERE (1 points)
    
            #########################################
    return W, b
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_train
        --- OR ---- 
        python3 -m nose -v test2.py:test_train
        --- OR ---- 
        python -m nose -v test2.py:test_train
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using Softmax Regression 1) Given a trained softmax regression model with parameters W and b. Suppose we have a test sample x. Please use the softmax regression model to predict the label of x and the probabilities of the label being in each of the classes, i.e. the activation a(x). . 
    ---- Inputs: --------
        * x: the feature vector of a data instance, a float numpy vector of length p.
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
        * b: the bias values of softmax regression, a float numpy vector of length c.
    ---- Outputs: --------
        * y: the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        * a: the softmax activations, a float numpy vector of length c..
    ---- Hints: --------
        * If we have multiple elements in the activations being the largest at the same time (for example, [0.5, 0.5,0] have two largest values), we can break the tie by choosing the element with the smallest index. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def inference(x, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return y, a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_inference
        --- OR ---- 
        python3 -m nose -v test2.py:test_inference
        --- OR ---- 
        python -m nose -v test2.py:test_inference
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using Softmax Regression 2) Given a trained softmax regression model with parameters W and b. Suppose we have a test dataset Xtest (features). For each data sample x in Xtest, use the softmax regression model to predict the label of x and the probabilities of the label being in each of the classes, i.e. the activation a(x). 
    ---- Inputs: --------
        * Xtest: the feature matrix of test instances, a float numpy matrix of shape (n_test by p).
        * W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p).
        * b: the bias values of softmax regression, a float numpy vector of length c.
    ---- Outputs: --------
        * Ytest: the predicted labels of test data samples, an integer numpy array of length ntest. Ytest[i] represents the predicted label on the i-th test sample.
        * P: the predicted probabilities of test data samples to be in different classes, a float numpy matrix of shape (n_test,c).  P[i,j] is the probability of the i-th data sample to have the j-th class label.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict(Xtest, W, b):
    n_test = Xtest.shape[0] # the number of test samples
    c = W.shape[0] # the number of classes
    Ytest = np.zeros(n_test) # initialize label vector as all zeros
    P = np.zeros((n_test,c)) # initialize label probability matrix as all zeros
    for i in range(n_test): # iterate through each test sample
        x=Xtest[i] # the feature vector of the i-th data sample
        #########################################
        ## INSERT YOUR CODE HERE (5 points)
    
        #########################################
    return Ytest, P
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
        ----------- Problem 2 (30 points in total)--------------------- ... ok
        * (1 points) compute_z ... ok
        * (1 points) compute_dz_db ... ok
        * (1 points) compute_dL_db ... ok
        * (1 points) compute_dz_dW ... ok
        * (2 points) compute_dL_dW ... ok
        * (2 points) compute_a ... ok
        * (2 points) compute_da_dz ... ok
        * (2 points) compute_dL_dz ... ok
        * (2 points) compute_L ... ok
        * (2 points) compute_dL_da ... ok
        * (1 points) forward ... ok
        * (4 points) backward ... ok
        * (1 points) update_b ... ok
        * (1 points) update_W ... ok
        * (1 points) train ... ok
        * (1 points) inference ... ok
        * (5 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 17 tests in 1.189s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* p:  the number of input features. 
* c:  the number of classes in the classification task, an integer scalar. 
* x:  the feature vector of a data instance, a float numpy vector of length p. 
* y:  the label of one data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1). 
* W:  the weight matrix of softmax regression, a float numpy matrix of shape (c by p). 
* b:  the bias values of softmax regression, a float numpy vector of length c. 
* z:  the linear logits, a float numpy vector of length c.. 
* a:  the softmax activations, a float numpy vector of length c.. 
* L:  the multi-class cross entropy loss, a float scalar.. 
* dL_da:  the partial gradients of the loss function L w.r.t. the activations a, a float numpy vector of length c.  The i-th element dL_da[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a[i],  d_L / d_a[i].. 
* dL_dz:  the partial gradients of the loss function L w.r.t. the linear logits z, a float numpy vector of length c.  The i-th element dL_dz[i] represents the partial gradient of the loss function L w.r.t. the i-th linear logit z[i],  d_L / d_z[i].. 
* da_dz:  the partial gradient of the activations a w.r.t. the logits z, a float numpy matrix of shape (c by c).  The (i,j)-th element of da_dz represents the partial gradient ( d_a[i]  / d_z[j] ). 
* dz_dW:  the partial gradient of logits z w.r.t. the weight matrix W, a numpy float tensor of shape (c by c by p).  The (i,j,k)-th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k],   d_z[i] / d_W[j,k]. 
* dz_db:  the partial gradient of the logits z w.r.t. the biases b, a float matrix of shape c by c.  Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias b[j],   d_z[i] / d_b[j]. 
* dL_dW:  the partial gradients of the loss function L w.r.t. the weight matrix W, a numpy float matrix of shape (c by p).  The i,j-th element dL_dW[i,j] represents the partial gradient of the loss function L w.r.t. the i,j-th weight W[i,j],  d_L / d_W[i,j]. 
* dL_db:  the partial gradient of the loss function L w.r.t. the biases b, a float numpy vector of length c.  The i-th element dL_db[i] represents the partial gradient of the loss function w.r.t. the i-th bias,  d_L / d_b[i]. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* n:  the number of data instance in the training set. 
* n_test:  the number of data instance in the test set. 
* X:  the feature matrix of training instances, a float numpy matrix of shape (n by p). 
* Y:  the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1. 
* Xtest:  the feature matrix of test instances, a float numpy matrix of shape (n_test by p). 
* Ytest:  the predicted labels of test data samples, an integer numpy array of length ntest. Ytest[i] represents the predicted label on the i-th test sample. 
* P:  the predicted probabilities of test data samples to be in different classes, a float numpy matrix of shape (n_test,c).  P[i,j] is the probability of the i-th data sample to have the j-th class label. 

'''
#--------------------------------------------
