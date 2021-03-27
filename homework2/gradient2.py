from problem2 import *
import numpy as np

#-----------------------------------------------------------------
# gradient checking 
#-----------------------------------------------------------------


#-----------------------------------------------------------------
def check_da_dz(z, delta=1e-7):
    '''
        Compute local gradient of the softmax function using gradient checking.
        Input:
            z: the logit values of softmax regression, a float numpy vector of length c. Here c is the number of classes
            delta: a small number for gradient check, a float scalar.
        Output:
            da_dz: the approximated local gradient of the activations w.r.t. the logits, a float numpy matrix of shape (c by c). 
                   The (i,j)-th element represents the partial gradient ( d a[i]  / d z[j] )
    '''
    c = len(z) # number of classes
    da_dz = np.zeros((c,c))
    for i in range(c):
        for j in range(c):
            d = np.zeros(c)
            d[j] = delta
            da_dz[i,j] = (compute_a(z+d)[i] - compute_a(z)[i]) / delta
    return da_dz 

#-----------------------------------------------------------------
def check_dL_da(a, y, delta=1e-7):
    '''
        Compute local gradient of the multi-class cross-entropy function w.r.t. the activations using gradient checking.
        Input:
            a: the activations of a training instance, a float numpy vector of length c. Here c is the number of classes. 
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_da: the approximated local gradients of the loss function w.r.t. the activations, a float numpy vector of length c.
    '''
    c = a.shape[0] # number of classes
    dL_da = np.zeros(c) # initialize the vector as all zeros
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        dL_da[i] = ( compute_L(a+d,y) - compute_L(a,y)) / delta
    return dL_da 

#--------------------------
def check_dz_dW(x, W, b, delta=1e-7):
    '''
        compute the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of length c.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dW: the approximated local gradient of the logits w.r.t. the weight matrix computed by gradient checking, a numpy float matrix of shape (c by c by p). 
                   The i,j,k -th element of dz_dW represents the partial gradient of the i-th logit (z[i]) w.r.t. the weight W[j,k]:   d_z[i] / d_W[j,k]
    '''
    c,p = W.shape # number of classes and features
    dz_dW = np.zeros((c,c,p))
    for i in range(c):
        for j in range(c):
            for k in range(p):
                d = np.zeros((c,p))
                d[j,k] = delta
                dz_dW[i,j,k] = (compute_z(x,W+d, b)[i] - compute_z(x, W, b))[i] / delta
    return dz_dW


#--------------------------
def check_dz_db(x, W, b, delta=1e-7):
    '''
        compute the local gradient of the logit function using gradient check.
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of length c.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_db: the approximated local gradient of the logits w.r.t. the biases using gradient check, a float vector of length c.
                   Each element dz_db[i,j] represents the partial gradient of the i-th logit z[i] w.r.t. the j-th bias:  d_z[i] / d_b[j]
    '''
    c,p = W.shape # number of classes and features
    dz_db = np.zeros((c,c))
    for i in range(c):
        for j in range(c):
            d = np.zeros(c) 
            d[j] = delta
            dz_db[i,j] = (compute_z(x,W, b+d)[i] - compute_z(x, W, b)[i]) / delta
    return dz_db


#-----------------------------------------------------------------
def check_dL_dW(x,y,W,b,delta=1e-7):
    '''
       Compute the gradient of the loss function w.r.t. the weights W using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of length c.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dW: the approximated gradients of the loss function w.r.t. the weight matrix, a numpy float matrix of shape (c by p). 
    '''
    c, p = W.shape    
    dL_dW = np.zeros((c,p))
    for i in range(c):
        for j in range(p):
            d = np.zeros((c,p))
            d[i,j] = delta
            a1 = forward(x,W+d,b) 
            a2 = forward(x,W,b) 
            L1 = compute_L(a1,y)
            L2 = compute_L(a2,y)
            dL_dW[i,j] = (L1 - L2 ) / delta
    return dL_dW


#-----------------------------------------------------------------
def check_dL_db(x,y,W,b,delta=1e-7):
    '''
       Compute the gradient of the loss function w.r.t. the bias b using gradient checking.
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0,1,2, ..., or (c-1).
            W: the weight matrix of softmax regression, a float numpy matrix of shape (c by p). Here c is the number of classes.
            b: the bias values of softmax regression, a float numpy vector of length c.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_db: the approximated gradients of the loss function w.r.t. the biases, a float vector of length c.
    '''
    c, p = W.shape    
    dL_db =np.zeros(c)
    for i in range(c):
        d = np.zeros(c)
        d[i] = delta
        a1 = forward(x,W,b+d) 
        a2 = forward(x,W,b) 
        L1 = compute_L(a1,y)
        L2 = compute_L(a2,y)
        dL_db[i] = ( L1 - L2) / delta
    return dL_db 

