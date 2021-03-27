import problem1 as lr
import problem2 as sr
import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: Two-layer Fully Connected Neural Network for multi-class classification (40 points)
    In this problem, you will implement a multi-class classification method using fully-connected neural network with two layers.
The main goal of this problem is to extend the softmax regression method to multiple layers.
In the first layer, we will use sigmoid function as the activation function to convert the linear logits into a non-linear activations.
In the second layer, we will use softmax as the activation function. 
We will use multi-class cross entropy as the loss function and stochastic gradient descent to train the model parameters.

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Value Forward Function 1.1) Given a fully-connected neural network with parameters W1, b1, W2 and b2, please compute the linear logits in the first layer on a data sample x (i.e., z1(x)). 
    ---- Inputs: --------
        * x: the feature vector of a data sample, a float numpy vector of length p.
        * W1: the weight matrix of the 1st layer, a float numpy matrix of shape (h by p).
        * b1: the bias values of the 1st layer, a float numpy vector of length h.
    ---- Outputs: --------
        * z1: the linear logits of the 1st layer, a float numpy vector of length h.
    ---- Hints: --------
        * You can re-use the functions that you have implemented in the previous problem.  For example, sr.compute_a() represents the compute_a function you implemented for softmax regression.  Here 'sr' represents 'softmax regression'. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z1(x, W1, b1):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return z1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_z1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_z1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_z1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 1.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the linear logits z1(x) in the first layer on a training sample x. Please compute the partial gradient of the linear logits z1(x) in the first layer w.r.t. the biases b1 in the first layer. 
    ---- Inputs: --------
        * h: the number of outputs in the 1st layer (or the number of hidden neurons in the first layer).
    ---- Outputs: --------
        * dz1_db1: the partial gradient of the logits z1 w.r.t. the biases b1, a float matrix of shape (h, h).  Each (i,j)-th element represents the partial gradient of the i-th logit z1[i] w.r.t. the j-th bias b1[i]:  d_z1[i] / d_b1[j].
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dz1_db1(h=3):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dz1_db1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dz1_db1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dz1_db1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dz1_db1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 1.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z1(x) in the first layer and the local gradient of the linear logits z1(x) w.r.t. the biases b1 on a training sample x. Please compute the partial gradient of the loss L w.r.t. the biases b1 in the first layer using chain rule. 
    ---- Inputs: --------
        * dL_dz1: the partial gradient ofthe loss L w.r.t. the logits z1, a float numpy vector of length h.  The i-th element of represents the partial gradient ( d_L  / d_z1[i] ).
        * dz1_db1: the partial gradient of the logits z1 w.r.t. the biases b1, a float matrix of shape (h, h).  Each (i,j)-th element represents the partial gradient of the i-th logit z1[i] w.r.t. the j-th bias b1[i]:  d_z1[i] / d_b1[j].
    ---- Outputs: --------
        * dL_db1: the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i].
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_db1(dL_dz1, dz1_db1):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_db1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_db1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_db1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_db1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 1.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the linear logits z1(x) in the first layer on a training sample x. Please compute the partial gradients of the linear logits z1(x) in the first layer w.r.t. the weights W1 in the 1st layer. 
    ---- Inputs: --------
        * x: the feature vector of a data sample, a float numpy vector of length p.
        * h: the number of outputs in the 1st layer (or the number of hidden neurons in the first layer).
    ---- Outputs: --------
        * dz1_dW1: the partial gradient of logits z1 w.r.t. the weight matrix W1, a numpy float tensor of shape (h by h by p).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z1[i]) w.r.t. the weight W1[j,k]:   d_z1[i] / d_W1[j,k].
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dz1_dW1(x, h=3):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dz1_dW1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dz1_dW1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dz1_dW1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dz1_dW1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 1.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z1(x) in the first layer and the local gradient of the linear logits z1(x) w.r.t. the weights W1 on a training sample x. Please compute the partial gradient of the loss L w.r.t. the weights W1 in the first layer using chain rule. 
    ---- Inputs: --------
        * dL_dz1: the partial gradient ofthe loss L w.r.t. the logits z1, a float numpy vector of length h.  The i-th element of represents the partial gradient ( d_L  / d_z1[i] ).
        * dz1_dW1: the partial gradient of logits z1 w.r.t. the weight matrix W1, a numpy float tensor of shape (h by h by p).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z1[i]) w.r.t. the weight W1[j,k]:   d_z1[i] / d_W1[j,k].
    ---- Outputs: --------
        * dL_dW1: the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j].
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_dW1(dL_dz1, dz1_dW1):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_dW1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_dW1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_dW1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_dW1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Value Forward Function 1.2) Suppose we are given a fully-connected neural network and we have already computed the linear logits z1(x) in the first layer on a data sample x. Please compute the element-wise sigmoid activations a1(x) in the first layer on the data sample. Here we use element-wise sigmoid function to transform linear logits z1(x) into activations a1(x). 
    ---- Inputs: --------
        * z1: the linear logits of the 1st layer, a float numpy vector of length h.
    ---- Outputs: --------
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
    ---- Hints: --------
        * In this function, we want to compute the element-wise sigmoid: computing sigmoid on each element of z1 and put them into a vector a1, so that a1[i] = sigmoid(z1[i]). 
        * You could reuse the functions in logistic regression, for example lr.function_name(). Here 'lr' represents 'logistic regression'. 
        * This function is slightly different from the sigmoid function in logistic regression. In logistic regression, the input z is a scalar, but here the input z1 is a vector. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_a1(z1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return a1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_a1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_a1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_a1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer on a training sample x. Please compute the partial gradients of the activations a1(x) in the first layer w.r.t. the linear logits z1(x) in the first layer. 
    ---- Inputs: --------
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
    ---- Outputs: --------
        * da1_dz1: the partial gradient of the activations a1 w.r.t. the logits z1, a float numpy matrix of shape (h, h).  The (i,j)-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[j] ).
    ---- Hints: --------
        * Remember that the activations a1(x) in the first layer are computed using element-wise sigmoid function. Each a1[i] was computed by sigmoid(z1[i]). So a1(x) is NOT computed by softmax on z1(x). In this case, you cannot use the compute_da_dz() function in ether softmax regression or logistic regression. You need to implement this gradient function from scratch. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_da1_dz1(a1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return da1_dz1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_da1_dz1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_da1_dz1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_da1_dz1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the activation a1(x) in the first layer and the local gradient of the activations a1(x) w.r.t. the linear logits z1(x) on a training sample x. Please compute the partial gradients of the loss L w.r.t. the linear logits z1(x) in the first layer using chain rule. 
    ---- Inputs: --------
        * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] ).
        * da1_dz1: the partial gradient of the activations a1 w.r.t. the logits z1, a float numpy matrix of shape (h, h).  The (i,j)-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[j] ).
    ---- Outputs: --------
        * dL_dz1: the partial gradient ofthe loss L w.r.t. the logits z1, a float numpy vector of length h.  The i-th element of represents the partial gradient ( d_L  / d_z1[i] ).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression.. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_dz1(dL_da1, da1_dz1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return dL_dz1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_dz1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_dz1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_dz1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Value Forward Function 2.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer on a data sample x, please compute the linear logits z2(x) in the second layer on the data sample. 
    ---- Inputs: --------
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
        * b2: the bias values of the 2nd layer, a float numpy vector of length c.
    ---- Outputs: --------
        * z2: the linear logits in the second layer, a float numpy vector of length c.
    ---- Hints: --------
        * The activations a1(x) in the first layer is used as the input to the second layer for computing z2(x). 
        * You could re-use the functions in softmax regression, for example sr.function_name(). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z2(a1, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_z2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_z2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_z2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 2.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the biases b2 in the second layer. 
    ---- Inputs: --------
        * c: the number of classes in the classification task, an integer scalar.
    ---- Outputs: --------
        * dz2_db2: the partial gradient of the logits z2 w.r.t. the biases b2, a float matrix of shape (c, c).  Each (i,j)-th element represents the partial gradient of the i-th logit z2[i] w.r.t. the j-th bias b2[j]:  d_z2[i] / d_b2[j].
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dz2_db2(c):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dz2_db2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dz2_db2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dz2_db2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dz2_db2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 2.1.1) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z2(x) in the second layer and the local gradient of the linear logits z2(x) w.r.t. the biases b2 on a training sample x. Please compute the partial gradients of the loss L w.r.t. the biases b2 in the second layer using chain rule. 
    ---- Inputs: --------
        * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] ).
        * dz2_db2: the partial gradient of the logits z2 w.r.t. the biases b2, a float matrix of shape (c, c).  Each (i,j)-th element represents the partial gradient of the i-th logit z2[i] w.r.t. the j-th bias b2[j]:  d_z2[i] / d_b2[j].
    ---- Outputs: --------
        * dL_db2: the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i].
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_db2(dL_dz2, dz2_db2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_db2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_db2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_db2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_db2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 2.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the weights W2 in the second layer. 
    ---- Inputs: --------
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
        * c: the number of classes in the classification task, an integer scalar.
    ---- Outputs: --------
        * dz2_dW2: the partial gradient of logits z2 w.r.t. the weight matrix W2, a numpy float tensor of shape (c by c by h).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z2[i]) w.r.t. the weight W2[j,k]:   d_z2[i] / d_W2[j,k].
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dz2_dW2(a1, c):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return dz2_dW2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dz2_dW2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dz2_dW2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dz2_dW2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 2.1.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss L w.r.t. the linear logits z2(x) in the second layer and the local gradient of the linear logits z2(x) w.r.t. the weights w2 on a training sample x. Please compute the partial gradients of the loss L w.r.t. the weights W2 in the second layer using chain rule. 
    ---- Inputs: --------
        * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] ).
        * dz2_dW2: the partial gradient of logits z2 w.r.t. the weight matrix W2, a numpy float tensor of shape (c by c by h).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z2[i]) w.r.t. the weight W2[j,k]:   d_z2[i] / d_W2[j,k].
    ---- Outputs: --------
        * dL_dW2: the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j].
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_dW2(dL_dz2, dz2_dW2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_dW2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_dW2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_dW2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_dW2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 2.1.3) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the activations a1(x) in the first layer and the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradients of the linear logits z2(x) in the second layer w.r.t. the activations a1(x) in the first layer. 
    ---- Inputs: --------
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
    ---- Outputs: --------
        * dz2_da1: the partial gradient of the logits z2 w.r.t. the inputs a1, a float numpy matrix of shape (c, h).  The (i,j)-th element represents the partial gradient ( d_z2[i]  / d_a1[j] ).
    ---- Hints: --------
        * The activations a1(x) in the first layer is used as the input to the second layer for computing z2(x). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dz2_da1(W2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return dz2_da1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dz2_da1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dz2_da1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dz2_da1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 2.1.3) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss w.r.t. the linear logits z2(x) and the local gradients of the linear logits z2(x) w.r.t. the activations a1(x) in the first layer on a training sample x. Please compute the partial gradient of the loss function L w.r.t. the activations a1(x) in the first layer using chain rule. 
    ---- Inputs: --------
        * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] ).
        * dz2_da1: the partial gradient of the logits z2 w.r.t. the inputs a1, a float numpy matrix of shape (c, h).  The (i,j)-th element represents the partial gradient ( d_z2[i]  / d_a1[j] ).
    ---- Outputs: --------
        * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] ).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_da1(dL_dz2, dz2_da1):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_da1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_da1
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_da1
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_da1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Value Forward Function 2.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the linear logits z2(x) in the second layer on a data sample x, please compute the softmax activations a2(x) in the second layer on the data sample. 
    ---- Inputs: --------
        * z2: the linear logits in the second layer, a float numpy vector of length c.
    ---- Outputs: --------
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
    ---- Hints: --------
        * you could re-use the functions in softmax regression, for example sr.function_name(). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_a2(z2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return a2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_a2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_a2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_a2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local Gradient 2.2) Suppose we are given a fully-connected neural network and we have already computed the activations a2(x) in the second layer on a training sample x. Please compute the partial gradients of the softmax activations a2(x) w.r.t. the logits z2(x) in the second layer. 
    ---- Inputs: --------
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
    ---- Outputs: --------
        * da2_dz2: the partial gradient of the activations a2 w.r.t. the logits z2, a float numpy matrix of shape (c by c).  The (i,j)-th element represents the partial gradient ( d_a2[i]  / d_z2[j] ).
    ---- Hints: --------
        * you could re-use the functions in softmax regression, for example sr.function_name(). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_da2_dz2(a2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return da2_dz2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_da2_dz2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_da2_dz2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_da2_dz2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Global Gradient 2.2) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Suppose we have already computed the global gradients of the loss w.r.t. the activations a2(x) and the local gradients of the activations a2(x) w.r.t.  the linear logits z2(x) in the second layer on a training sample x. Please compute the partial gradient of the loss function L w.r.t. the linear logits z2(x) in the first layer using chain rule. 
    ---- Inputs: --------
        * dL_da2: the partial gradients of the loss function L w.r.t. the activations a2, a float numpy vector of length c.  The i-th element dL_da2[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a2[i]:  d_L / d_a2[i].
        * da2_dz2: the partial gradient of the activations a2 w.r.t. the logits z2, a float numpy matrix of shape (c by c).  The (i,j)-th element represents the partial gradient ( d_a2[i]  / d_z2[j] ).
    ---- Outputs: --------
        * dL_dz2: the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] ).
    ---- Hints: --------
        * you could re-use the functions in softmax regression, for example sr.function_name(). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_dz2(dL_da2, da2_dz2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_dz2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_dz2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_dz2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_dz2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Value Forward Function 3) Suppose we are given a fully-connected neural network and we have already computed the activations a2(x) in the second layer on a training sample x. Suppose the label of the training sample is y. Please compute the loss on the training sample using multi-class cross entropy loss. 
    ---- Inputs: --------
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
        * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
    ---- Outputs: --------
        * L: the multi-class cross entropy loss, a float scalar.
    ---- Hints: --------
        * you could re-use the functions in softmax regression, for example sr.function_name(). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(a2, y):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_L
        --- OR ---- 
        python -m nose -v test3.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Local/Global Gradient 3) Suppose we are given a fully-connected neural network and we have already computed the activations a2(x) in the second layer. Suppose the label of the training sample is y. Please compute the partial gradients of the loss L w.r.t. the activations a2(x) in the second layer. 
    ---- Inputs: --------
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
        * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
    ---- Outputs: --------
        * dL_da2: the partial gradients of the loss function L w.r.t. the activations a2, a float numpy vector of length c.  The i-th element dL_da2[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a2[i]:  d_L / d_a2[i].
    ---- Hints: --------
        * you could re-use the functions in softmax regression, for example sr.function_name(). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_dL_da2(a2, y):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_da2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_dL_da2
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_dL_da2
        --- OR ---- 
        python -m nose -v test3.py:test_compute_dL_da2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Forward Pass) Suppose we are given a fully-connected neural network with parameters W1, b1, W2 and b2. Given a data sample (x), please compute the activations a1(x) and a2(x) on the sample. 
    ---- Inputs: --------
        * x: the feature vector of a data sample, a float numpy vector of length p.
        * W1: the weight matrix of the 1st layer, a float numpy matrix of shape (h by p).
        * b1: the bias values of the 1st layer, a float numpy vector of length h.
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
        * b2: the bias values of the 2nd layer, a float numpy vector of length c.
    ---- Outputs: --------
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
    ---- Hints: --------
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def forward(x, W1, b1, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return a1, a2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_forward
        --- OR ---- 
        python3 -m nose -v test3.py:test_forward
        --- OR ---- 
        python -m nose -v test3.py:test_forward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Back Propagation in the second layer: Suppose we are given a fully-connected neural network with parameters (W1, b1, W2 and b2) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a1(x) in the first layer and the activations a2(x) in the second layer on the data sample in the forward-pass. Please compute the global gradients of the loss L w.r.t. the parameters W2, b2 and the activation a1(x) on the data sample using back propagation. 
    ---- Inputs: --------
        * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
    ---- Outputs: --------
        * dL_db2: the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i].
        * dL_dW2: the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j].
        * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] ).
    ---- Hints: --------
        * It's easier to follow a certain order to compute all the gradients: dL_da2, da2_dz2, dL_dz2, dz2_db2, dL_db2 .... 
        * This problem can be solved using 10 line(s) of code.
'''
#---------------------
def backward_layer2(y, a1, a2, W2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_db2, dL_dW2, dL_da1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_backward_layer2
        --- OR ---- 
        python3 -m nose -v test3.py:test_backward_layer2
        --- OR ---- 
        python -m nose -v test3.py:test_backward_layer2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Back Propagation in the first layer: Suppose we are given a fully-connected neural network with parameters (W1, b1, W2 and b2) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a1(x) in the first layer on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters W1 and b1 on the data sample using back propagation. 
    ---- Inputs: --------
        * x: the feature vector of a data sample, a float numpy vector of length p.
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
        * dL_da1: the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] ).
    ---- Outputs: --------
        * dL_db1: the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i].
        * dL_dW1: the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j].
    ---- Hints: --------
        * It's easier to follow a certain order to compute all the gradients: da1_dz1, dL_dz1, dz1_db1, dL_db1 .... 
        * This problem can be solved using 7 line(s) of code.
'''
#---------------------
def backward_layer1(x, a1, dL_da1):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return dL_db1, dL_dW1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_backward_layer1
        --- OR ---- 
        python3 -m nose -v test3.py:test_backward_layer1
        --- OR ---- 
        python -m nose -v test3.py:test_backward_layer1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Back Propagation (layer 2 and 1): Suppose we are given a fully-connected neural network with parameters (W1, b1, W2 and b2) and we have a training data sample (x) with label (y).  Suppose we have already computed the activations a1(x) in the first layer and the activations a2(x) in the second layer on the data sample in the forward-pass. Please compute the global gradients of the loss w.r.t. the parameters W1, b1, W2 and b2 on the data sample using back propagation. 
    ---- Inputs: --------
        * x: the feature vector of a data sample, a float numpy vector of length p.
        * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        * a1: the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i].
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
    ---- Outputs: --------
        * dL_db2: the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i].
        * dL_dW2: the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j].
        * dL_db1: the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i].
        * dL_dW1: the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j].
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def backward(x, y, a1, a2, W2):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    
    #########################################
    return dL_db2, dL_dW2, dL_db1, dL_dW1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_backward
        --- OR ---- 
        python3 -m nose -v test3.py:test_backward
        --- OR ---- 
        python -m nose -v test3.py:test_backward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Train a Fully-Connected Neural Network) Given a training dataset, train the Fully Connected Neural Network by iteratively updating the weights W and biases b using the gradients computed over each data instance. 
    ---- Inputs: --------
        * X: the feature matrix of training instances, a float numpy matrix of shape (n by p).
        * Y: the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1.
        * h: the number of outputs in the 1st layer (or the number of hidden neurons in the first layer).
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * W1: the weight matrix of the 1st layer, a float numpy matrix of shape (h by p).
        * b1: the bias values of the 1st layer, a float numpy vector of length h.
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
        * b2: the bias values of the 2nd layer, a float numpy vector of length c.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits, activations and loss. 
        * Step 2 Back propagation: compute the gradients of W1, b1, W2 and b2. 
        * Step 3 Gradient descent: update the parameters W1, b1, W2 and b2 using gradient descent. 
        * This problem can be solved using 6 line(s) of code.
'''
#---------------------
def train(X, Y, h=3, alpha=0.01, n_epoch=100):
    n,p = X.shape # n: the number of training samples, p: the number of features
    c = max(Y) + 1 # number of classes
    W1 = np.random.randn(h,p) # initialize W1 randomly using standard normal distribution
    b1= np.zeros(h) # initialize b1 as all zeros
    W2 = np.random.randn(c,h) # initialize W2 randomly using standard normal distribution
    b2= np.zeros(c) # initialize b2 as all zeros
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        indices = np.random.permutation(n) # shuffle the indices of all samples
        for i in indices: # iterate through each random training sample (x,y)
            x=X[i] # the feature vector of the i-th random sample
            y=Y[i] # the label of the i-th random sample
            #########################################
            ## INSERT YOUR CODE HERE (2 points)
    
            #########################################
    return W1, b1, W2, b2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_train
        --- OR ---- 
        python3 -m nose -v test3.py:test_train
        --- OR ---- 
        python -m nose -v test3.py:test_train
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using Fully-Connected Network 1) Given a trained full-connected neural network with parameters W1, b1, W2 and b2. Suppose we have a test sample x. Please use the model to predict the label of x and the probabilities of the label being in each of the classes, i.e. the activation a2(x). 
    ---- Inputs: --------
        * x: the feature vector of a data sample, a float numpy vector of length p.
        * W1: the weight matrix of the 1st layer, a float numpy matrix of shape (h by p).
        * b1: the bias values of the 1st layer, a float numpy vector of length h.
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
        * b2: the bias values of the 2nd layer, a float numpy vector of length c.
    ---- Outputs: --------
        * y: the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
        * a2: the softmax activations in the 2nd layer, a float numpy vector of length c.
    ---- Hints: --------
        * If we have multiple elements in the activations being the largest at the same time (for example, [0.5, 0.5,0] have two largest values), we can break the tie by choosing the element with the smallest index. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def inference(x, W1, b1, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (1 points)
    
    #########################################
    return y, a2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_inference
        --- OR ---- 
        python3 -m nose -v test3.py:test_inference
        --- OR ---- 
        python -m nose -v test3.py:test_inference
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using Fully-Connected Network 2) Given a trained full-connected neural network with parameters W1, b1, W2 and b2. Suppose we have a test dataset Xtest (features). For each data sample x in Xtest, use the model to predict the label of x and the probabilities of the label being in each of the classes, i.e. the activation a2(x). 
    ---- Inputs: --------
        * Xtest: the feature matrix of test instances, a float numpy matrix of shape (n_test by p).
        * W1: the weight matrix of the 1st layer, a float numpy matrix of shape (h by p).
        * b1: the bias values of the 1st layer, a float numpy vector of length h.
        * W2: the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h).
        * b2: the bias values of the 2nd layer, a float numpy vector of length c.
    ---- Outputs: --------
        * Ytest: the predicted labels of test data, an integer numpy array of length n_test Each element can be 0, 1, ..., or (c-1).
        * P: the predicted probabilities of test data to be in different classes, a float numpy matrix of shape (n_test,c). Each (i,j) element is between 0 and 1, indicating the probability of the i-th instance having the j-th class label.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict(Xtest, W1, b1, W2, b2):
    n_test = Xtest.shape[0] # number of test samples
    c = W2.shape[0] # number of classes
    Ytest = np.zeros(n_test) # initialize the labels as all zeros
    P = np.zeros((n_test,c)) # initialize the class probability matrix as all zeros
    for i in range(n_test): # iterate through each test instance
        x=Xtest[i] # the feature vector of the i-th data sample
        #########################################
        ## INSERT YOUR CODE HERE (4 points)
    
        #########################################
    return Ytest, P
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_predict
        --- OR ---- 
        python3 -m nose -v test3.py:test_predict
        --- OR ---- 
        python -m nose -v test3.py:test_predict
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 3: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py
        --- OR ---- 
        python3 -m nose -v test3.py
        --- OR ---- 
        python -m nose -v test3.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 3 (40 points in total)--------------------- ... ok
        * (1 points) compute_z1 ... ok
        * (1 points) compute_dz1_db1 ... ok
        * (1 points) compute_dL_db1 ... ok
        * (1 points) compute_dz1_dW1 ... ok
        * (1 points) compute_dL_dW1 ... ok
        * (2 points) compute_a1 ... ok
        * (2 points) compute_da1_dz1 ... ok
        * (2 points) compute_dL_dz1 ... ok
        * (2 points) compute_z2 ... ok
        * (1 points) compute_dz2_db2 ... ok
        * (1 points) compute_dL_db2 ... ok
        * (2 points) compute_dz2_dW2 ... ok
        * (1 points) compute_dL_dW2 ... ok
        * (2 points) compute_dz2_da1 ... ok
        * (1 points) compute_dL_da1 ... ok
        * (2 points) compute_a2 ... ok
        * (1 points) compute_da2_dz2 ... ok
        * (1 points) compute_dL_dz2 ... ok
        * (1 points) compute_L ... ok
        * (1 points) compute_dL_da2 ... ok
        * (1 points) forward ... ok
        * (1 points) backward_layer2 ... ok
        * (1 points) backward_layer1 ... ok
        * (3 points) backward ... ok
        * (2 points) train ... ok
        * (1 points) inference ... ok
        * (4 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 27 tests in 10.389s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* p:  the number of input features. 
* c:  the number of classes in the classification task, an integer scalar. 
* x:  the feature vector of a data sample, a float numpy vector of length p. 
* y:  the label of a data sample, an integer scalar value. The values can be 0,1,2, ..., or (c-1). 
* h:  the number of outputs in the 1st layer (or the number of hidden neurons in the first layer). 
* W1:  the weight matrix of the 1st layer, a float numpy matrix of shape (h by p). 
* b1:  the bias values of the 1st layer, a float numpy vector of length h. 
* W2:  the weight matrix of the 2nd layer, a float numpy matrix of shape (c by h). 
* b2:  the bias values of the 2nd layer, a float numpy vector of length c. 
* z1:  the linear logits of the 1st layer, a float numpy vector of length h. 
* a1:  the element-wise sigmoid activations in the 1st layer, a float numpy vector of length h.  The i-th element represents the sigmoid of the i-th logit z1[i]. 
* z2:  the linear logits in the second layer, a float numpy vector of length c. 
* a2:  the softmax activations in the 2nd layer, a float numpy vector of length c. 
* L:  the multi-class cross entropy loss, a float scalar. 
* dL_da2:  the partial gradients of the loss function L w.r.t. the activations a2, a float numpy vector of length c.  The i-th element dL_da2[i] represents the partial gradient of the loss function L w.r.t. the i-th activation a2[i]:  d_L / d_a2[i]. 
* da2_dz2:  the partial gradient of the activations a2 w.r.t. the logits z2, a float numpy matrix of shape (c by c).  The (i,j)-th element represents the partial gradient ( d_a2[i]  / d_z2[j] ). 
* dL_dz2:  the partial gradient of the loss L w.r.t. the logits z2, a float numpy vector of length c.  The i-th element represents the partial gradient ( d_L  / d_z2[i] ). 
* dz2_dW2:  the partial gradient of logits z2 w.r.t. the weight matrix W2, a numpy float tensor of shape (c by c by h).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z2[i]) w.r.t. the weight W2[j,k]:   d_z2[i] / d_W2[j,k]. 
* dL_dW2:  the partial gradient of the loss L w.r.t. the weight matrix W2, a numpy float matrix of shape (c by h).  The (i,j)-th element represents the partial gradient of the loss L w.r.t. the weight W2[i,j]:   d_L / d_W2[i,j]. 
* dz2_db2:  the partial gradient of the logits z2 w.r.t. the biases b2, a float matrix of shape (c, c).  Each (i,j)-th element represents the partial gradient of the i-th logit z2[i] w.r.t. the j-th bias b2[j]:  d_z2[i] / d_b2[j]. 
* dL_db2:  the partial gradient of the loss L w.r.t. the biases b2, a float vector of length c.  Each i-th element represents the partial gradient of loss L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]. 
* dz2_da1:  the partial gradient of the logits z2 w.r.t. the inputs a1, a float numpy matrix of shape (c, h).  The (i,j)-th element represents the partial gradient ( d_z2[i]  / d_a1[j] ). 
* dL_da1:  the partial gradient of the loss L w.r.t. the activations a1, a float numpy vector of shape h.  The i-th element represents the partial gradient ( d_L  / d_a1[i] ). 
* da1_dz1:  the partial gradient of the activations a1 w.r.t. the logits z1, a float numpy matrix of shape (h, h).  The (i,j)-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[j] ). 
* dL_dz1:  the partial gradient ofthe loss L w.r.t. the logits z1, a float numpy vector of length h.  The i-th element of represents the partial gradient ( d_L  / d_z1[i] ). 
* dz1_dW1:  the partial gradient of logits z1 w.r.t. the weight matrix W1, a numpy float tensor of shape (h by h by p).  The (i,j,k)-th element represents the partial gradient of the i-th logit (z1[i]) w.r.t. the weight W1[j,k]:   d_z1[i] / d_W1[j,k]. 
* dL_dW1:  the partial gradients of the loss function L w.r.t. the weights W1, a float numpy matrix of shape (h by p).  The i,j-th element represents the partial gradient of the loss function L w.r.t. the i,j-th weight W1[i,j]:  d_L / d_W1[i,j]. 
* dz1_db1:  the partial gradient of the logits z1 w.r.t. the biases b1, a float matrix of shape (h, h).  Each (i,j)-th element represents the partial gradient of the i-th logit z1[i] w.r.t. the j-th bias b1[i]:  d_z1[i] / d_b1[j]. 
* dL_db1:  the partial gradients of the loss function L w.r.t. the biases b1, a float numpy vector of length h.  The i-th element represents the partial gradient of the loss function L w.r.t. the i-th bias b2[i]:  d_L / d_b2[i]. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* n:  the number of data instance in the training set. 
* n_test:  the number of data instance in the test set. 
* X:  the feature matrix of training instances, a float numpy matrix of shape (n by p). 
* Y:  the labels of training instance, a numpy integer numpy array of length n. The values can be 0, 1, ..., or c-1. 
* Xtest:  the feature matrix of test instances, a float numpy matrix of shape (n_test by p). 
* Ytest:  the predicted labels of test data, an integer numpy array of length n_test Each element can be 0, 1, ..., or (c-1). 
* P:  the predicted probabilities of test data to be in different classes, a float numpy matrix of shape (n_test,c). Each (i,j) element is between 0 and 1, indicating the probability of the i-th instance having the j-th class label. 

'''
#--------------------------------------------