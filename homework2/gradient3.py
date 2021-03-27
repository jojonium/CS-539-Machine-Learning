from problem3 import *
import numpy as np

#-----------------------------------------------------------------
# gradient checking 
#-----------------------------------------------------------------


#--------------------------
def check_da1_dz1(z1,delta= 1e-7):
    '''
        Compute local gradient of the sigmoid activations a using gradient check.
        Input:
            z1: the input logits values of activation function, a float vector of shape p by 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            da1_dz1: the approximated local gradient of the activations a1 w.r.t. the logits z1, a float numpy vector of shape p by 1. 
                   The i-th element of da1_dz1 represents the partial gradient ( d_a1[i]  / d_z1[i] )
    '''
    h = z1.shape[0]
    da1_dz1 = np.zeros((h,h)) 
    for i in range(h):
        for j in range(h):
            d = np.zeros(h)
            d[j] = delta
            da1_dz1[i,j] = (compute_a1(z1+d)[i] - compute_a1(z1)[i]) / delta
    return da1_dz1 

#--------------------------
def check_dL_dW2(x,y, W1,b1,W2,b2, delta= 1e-7):
    '''
        Compute gradient of the weights W1 a using gradient check.
        Input:
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dW1: the approximated gradient of the loss L w.r.t. the weights W1
    '''
    c,h = W2.shape
    dL_dW2 = np.zeros((c,h)) 
    for i in range(c):
        for j in range(h):
            d = np.zeros((c,h))
            d[i,j] = delta
            a1, a2 = forward(x, W1, b1, W2+d, b2)
            L = sr.compute_L(a2,y)
            a1, a2 = forward(x, W1, b1, W2, b2)
            dL_dW2[i,j] = (L - sr.compute_L(a2,y)) / delta
    return dL_dW2 

#--------------------------
def check_dL_dW1(x,y, W1,b1,W2,b2, delta= 1e-7):
    '''
        Compute gradient of the weights W1 a using gradient check.
        Input:
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dW1: the approximated gradient of the loss L w.r.t. the weights W1
    '''
    h,p = W1.shape
    dL_dW1 = np.zeros((h,p)) 
    for i in range(h):
        for j in range(p):
            d = np.zeros((h,p))
            d[i,j] = delta
            a1, a2 = forward(x, W1+d, b1, W2, b2)
            L = sr.compute_L(a2,y)
            a1, a2 = forward(x, W1, b1, W2, b2)
            dL_dW1[i,j] = (L - sr.compute_L(a2,y)) / delta
    return dL_dW1 
