from problem4 import *
import sys
import math
import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
'''
    Unit test 4:
    This file includes unit tests for problem4.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 4 (36 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_xh():
    ''' (2 points) compute_xh'''
    # a mini-batch of 2 time sequences, suppose p=3
    # at time step t, the input features of the two sequences are
    xt = th.tensor([ 
                        [1.,2.,3.], # the input features of the first time sequence of the mini-batch at time step t
                        [4.,5.,6.] # the input features of the second time sequence of the mini-batch at time step t
                   ],requires_grad=True)
    # the hidden states at the end of the t-1 time step, h=4
    ht_1 = th.tensor([ 
                        [.1,.2,.3,.4], # the hidden states of the first time sequence of the mini-batch at the end of time step t-1
                        [.5,.6,.7,.8] # the hidden states of the second time sequence of the mini-batch at the end of time step t-1
                     ],requires_grad=True)
    xh = compute_xh(xt,ht_1)
    assert np.allclose(xh.size(),[2,7])
    # check if the values are computed  correctly
    xh_true = [
                [1.,2.,3.,  .1,.2,.3,.4],
                [4.,5.,6.,  .5,.6,.7,.8]
              ]
    assert np.allclose(xh.data,xh_true)
    # check if the gradients are connected correctly
    # create a toy loss function: the sum of all elements 
    L = xh.sum()
    # back propagation to compute the gradients
    L.backward()
    dL_dx = np.ones((2,3))
    dL_dh = np.ones((2,4))
    assert np.allclose(xt.grad,dL_dx)
    assert np.allclose(ht_1.grad,dL_dh)
    # test the function with random input sizes
    n = np.random.randint(2,10)
    p = np.random.randint(2,10)
    h = np.random.randint(2,10)
    xt = th.randn(n,p) 
    ht_1 = th.randn(n,h) 
    xh = compute_xh(xt,ht_1)
    assert np.allclose(xh.size(), [n,p+h])
#---------------------------------------------------
def test_compute_z_f():
    ''' (2 points) compute_z_f'''
    # a mini-batch of 2 time sequences (n=2), 
    # suppose p=3, h=4
    xh = th.tensor([
                [1.,2.,3.,  .1,.2,.3,.4],
                [4.,5.,6.,  .5,.6,.7,.8]
              ],requires_grad=True)
    W = th.tensor([
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
              ],requires_grad=True)
    b = th.tensor([ .4,.5,.6,.7], requires_grad=True)
    z = compute_z_f(xh,W, b) 
    z_true = [[ 0.0,  0.1,  0.2,  0.3],
              [-0.7, -0.6, -0.5, -0.4]]
    assert np.allclose(z.data,z_true,atol=0.01)
    # check if the gradients are connected correctly
    # create a toy loss function: the sum of all elements 
    L = z.sum()
    # back propagation to compute the gradients
    L.backward()
    dL_dh = [[ 0.4,  0.4,  0.4, -4., -4., -4., -4.],
             [ 0.4,  0.4,  0.4, -4., -4., -4., -4.]]
    dL_dW = [[5.0, 5.0, 5.0, 5.0],
             [7.0, 7.0, 7.0, 7.0],
             [9.0, 9.0, 9.0, 9.0],
             [0.6, 0.6, 0.6, 0.6],
             [0.8, 0.8, 0.8, 0.8],
             [1.0, 1.0, 1.0, 1.0],
             [1.2, 1.2, 1.2, 1.2]]
    dL_db = [2., 2., 2., 2.]
    assert np.allclose(xh.grad,dL_dh)
    assert np.allclose(W.grad,dL_dW)
    assert np.allclose(b.grad,dL_db)
    # test the function with random input sizes
    n = np.random.randint(2,10)
    p = np.random.randint(2,10)
    h = np.random.randint(2,10)
    xh = th.randn(n,p+h) 
    W = th.randn(p+h,h)
    b = th.randn(h)
    z = compute_z_f(xh,W,b)
    assert np.allclose(z.size(), [n,h])
#---------------------------------------------------
def test_compute_f_t():
    ''' (2 points) compute_f_t'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    z = th.tensor([
                   #---------- the hidden states for the first time sequence in the mini-batch at time step t ------
                   [0.0, 0.2, 1000.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.5,-0.2,-1000.],
                    ], requires_grad=True)
    f = compute_f_t(z)
    assert type(f) == th.Tensor 
    f_true =[[0.5000, 0.5498, 1.],
             [0.6225, 0.4502, 0.]]
    assert np.allclose(f.data,f_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = f.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_dz = [[0.2500, 0.2475, 0.],
             [0.2350, 0.2475, 0.]]
    assert np.allclose(z.grad, dL_dz, atol= 0.001)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    z = th.randn(n,h)
    f = compute_f_t(z)
    assert np.allclose(f.size(),(n,h))
#---------------------------------------------------
def test_compute_z_i():
    ''' (2 points) compute_z_i'''
    # a mini-batch of 2 time sequences (n=2), 
    # suppose p=3, h=4
    xh = th.tensor([
                [1.,2.,3.,  .1,.2,.3,.4],
                [4.,5.,6.,  .5,.6,.7,.8]
              ],requires_grad=True)
    W = th.tensor([
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
              ],requires_grad=True)
    b = th.tensor([ .4,.5,.6,.7], requires_grad=True)
    z = compute_z_i(xh,W, b) 
    z_true = [[ 0.0,  0.1,  0.2,  0.3],
              [-0.7, -0.6, -0.5, -0.4]]
    assert np.allclose(z.data,z_true,atol=0.01)
    # check if the gradients are connected correctly
    # create a toy loss function: the sum of all elements 
    L = z.sum()
    # back propagation to compute the gradients
    L.backward()
    dL_dh = [[ 0.4,  0.4,  0.4, -4., -4., -4., -4.],
             [ 0.4,  0.4,  0.4, -4., -4., -4., -4.]]
    dL_dW = [[5.0, 5.0, 5.0, 5.0],
             [7.0, 7.0, 7.0, 7.0],
             [9.0, 9.0, 9.0, 9.0],
             [0.6, 0.6, 0.6, 0.6],
             [0.8, 0.8, 0.8, 0.8],
             [1.0, 1.0, 1.0, 1.0],
             [1.2, 1.2, 1.2, 1.2]]
    dL_db = [2., 2., 2., 2.]
    assert np.allclose(xh.grad,dL_dh)
    assert np.allclose(W.grad,dL_dW)
    assert np.allclose(b.grad,dL_db)
    # test the function with random input sizes
    n = np.random.randint(2,10)
    p = np.random.randint(2,10)
    h = np.random.randint(2,10)
    xh = th.randn(n,p+h) 
    W = th.randn(p+h,h)
    b = th.randn(h)
    z = compute_z_i(xh,W,b)
    assert np.allclose(z.size(), [n,h])
#---------------------------------------------------
def test_compute_i_t():
    ''' (2 points) compute_i_t'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    z = th.tensor([
                   #---------- the hidden states for the first time sequence in the mini-batch at time step t ------
                   [0.0, 0.2, 1000.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.5,-0.2,-1000.],
                    ], requires_grad=True)
    i = compute_i_t(z)
    assert type(i) == th.Tensor 
    i_true =[[0.5000, 0.5498, 1.],
             [0.6225, 0.4502, 0.]]
    assert np.allclose(i.data,i_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = i.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_dz = [[0.2500, 0.2475, 0.],
             [0.2350, 0.2475, 0.]]
    assert np.allclose(z.grad, dL_dz, atol= 0.001)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    z = th.randn(n,h)
    i = compute_i_t(z)
    assert np.allclose(i.size(),(n,h))
#---------------------------------------------------
def test_compute_z_c():
    ''' (2 points) compute_z_c'''
    # a mini-batch of 2 time sequences (n=2), 
    # suppose p=3, h=4
    xh = th.tensor([
                [1.,2.,3.,  .1,.2,.3,.4],
                [4.,5.,6.,  .5,.6,.7,.8]
              ],requires_grad=True)
    W = th.tensor([
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
              ],requires_grad=True)
    b = th.tensor([ .4,.5,.6,.7], requires_grad=True)
    z = compute_z_c(xh,W, b) 
    z_true = [[ 0.0,  0.1,  0.2,  0.3],
              [-0.7, -0.6, -0.5, -0.4]]
    assert np.allclose(z.data,z_true,atol=0.01)
    # check if the gradients are connected correctly
    # create a toy loss function: the sum of all elements 
    L = z.sum()
    # back propagation to compute the gradients
    L.backward()
    dL_dh = [[ 0.4,  0.4,  0.4, -4., -4., -4., -4.],
             [ 0.4,  0.4,  0.4, -4., -4., -4., -4.]]
    dL_dW = [[5.0, 5.0, 5.0, 5.0],
             [7.0, 7.0, 7.0, 7.0],
             [9.0, 9.0, 9.0, 9.0],
             [0.6, 0.6, 0.6, 0.6],
             [0.8, 0.8, 0.8, 0.8],
             [1.0, 1.0, 1.0, 1.0],
             [1.2, 1.2, 1.2, 1.2]]
    dL_db = [2., 2., 2., 2.]
    assert np.allclose(xh.grad,dL_dh)
    assert np.allclose(W.grad,dL_dW)
    assert np.allclose(b.grad,dL_db)
    # test the function with random input sizes
    n = np.random.randint(2,10)
    p = np.random.randint(2,10)
    h = np.random.randint(2,10)
    xh = th.randn(n,p+h) 
    W = th.randn(p+h,h)
    b = th.randn(h)
    z = compute_z_c(xh,W,b)
    assert np.allclose(z.size(), [n,h])
#---------------------------------------------------
def test_compute_C_c():
    ''' (2 points) compute_C_c'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    z = th.tensor([
                   #---------- the hidden states for the first time sequence in the mini-batch at time step t ------
                   [0.0, 0.2, 1000.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.5,-0.2,-1000.],
                    ], requires_grad=True)
    C = compute_C_c(z)
    assert type(C) == th.Tensor 
    C_true =[[ 0.0000,  0.1974,  1.],
              [ 0.4621, -0.1974, -1.]]
    assert np.allclose(C.data,C_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = C.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_dz = [[1.0000, 0.961, 0.],
             [0.7864, 0.961, 0.]]    
    assert np.allclose(z.grad, dL_dz, atol= 0.001)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    z = th.randn(n,h)
    o = compute_C_c(z)
    assert np.allclose(o.size(),(n,h))
#---------------------------------------------------
def test_compute_Ct():
    ''' (2 points) compute_Ct'''
    
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    f = th.tensor([
                        #---------- the forget gates for the first time sequence in the mini-batch at time step t ------
                        [0.1, 0.2, 0.3],
                        #---------- the forget gates for the second time sequence in the mini-batch at time step t ------
                        [0.4, 0.5, 0.6],
                  ], requires_grad=True)
    C_old = th.tensor([
                        #---------- the input gates for the first time sequence in the mini-batch at time step t ------
                        [-0.5,-0.6,-0.7],
                        #---------- the input gates for the second time sequence in the mini-batch at time step t ------
                        [ 0.5, 0.6, 0.7],
                    ], requires_grad=True)
    i = th.tensor([
                        #---------- the input gates for the first time sequence in the mini-batch at time step t ------
                        [0.4, 0.5, 0.6],
                        #---------- the input gates for the second time sequence in the mini-batch at time step t ------
                        [0.7, 0.8, 0.9],
                  ], requires_grad=True)
    C_c = th.tensor([
                        #---------- the input gates for the first time sequence in the mini-batch at time step t ------
                        [ 0.5, 0.6, 0.7],
                        #---------- the input gates for the second time sequence in the mini-batch at time step t ------
                        [-0.5,-0.6,-0.7],
                    ], requires_grad=True)
    C = compute_Ct(f,i,C_c,C_old)
    assert type(C) == th.Tensor 
    assert np.allclose(C.size(),[2,3])
    C_true =[[ 0.15,  0.18,  0.21],
             [-0.15, -0.18, -0.21]]
    assert np.allclose(C.data,C_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = C.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_df = [[-0.5, -0.6, -0.7],
             [ 0.5,  0.6,  0.7]]
    dL_di = [[ 0.5,  0.6,  0.7],
             [-0.5, -0.6, -0.7]]
    dL_dC_c = [[0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9]]
    dL_dC_old = [[0.1, 0.2, 0.3],
                 [0.4, 0.5, 0.6]]
    assert np.allclose(f.grad, dL_df, atol= 0.01)
    assert np.allclose(i.grad, dL_di, atol= 0.01)
    assert np.allclose(C_c.grad, dL_dC_c, atol= 0.01)
    assert np.allclose(C_old.grad, dL_dC_old, atol= 0.01)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    f = th.randn(n,h)
    i = th.randn(n,h)
    C_c = th.randn(n,h)
    C_old = th.randn(n,h)
    C = compute_Ct(f,i,C_c,C_old)
    assert np.allclose(C.size(),(n,h))
#---------------------------------------------------
def test_compute_z_o():
    ''' (2 points) compute_z_o'''
    # a mini-batch of 2 time sequences (n=2), 
    # suppose p=3, h=4
    xh = th.tensor([
                [1.,2.,3.,  .1,.2,.3,.4],
                [4.,5.,6.,  .5,.6,.7,.8]
              ],requires_grad=True)
    W = th.tensor([
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [0.1,0.1,0.1,0.1],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
                    [-1.,-1.,-1.,-1.],
              ],requires_grad=True)
    b = th.tensor([ .4,.5,.6,.7], requires_grad=True)
    z = compute_z_o(xh,W, b) 
    z_true = [[ 0.0,  0.1,  0.2,  0.3],
              [-0.7, -0.6, -0.5, -0.4]]
    assert np.allclose(z.data,z_true,atol=0.01)
    # check if the gradients are connected correctly
    # create a toy loss function: the sum of all elements 
    L = z.sum()
    # back propagation to compute the gradients
    L.backward()
    dL_dh = [[ 0.4,  0.4,  0.4, -4., -4., -4., -4.],
             [ 0.4,  0.4,  0.4, -4., -4., -4., -4.]]
    dL_dW = [[5.0, 5.0, 5.0, 5.0],
             [7.0, 7.0, 7.0, 7.0],
             [9.0, 9.0, 9.0, 9.0],
             [0.6, 0.6, 0.6, 0.6],
             [0.8, 0.8, 0.8, 0.8],
             [1.0, 1.0, 1.0, 1.0],
             [1.2, 1.2, 1.2, 1.2]]
    dL_db = [2., 2., 2., 2.]
    assert np.allclose(xh.grad,dL_dh)
    assert np.allclose(W.grad,dL_dW)
    assert np.allclose(b.grad,dL_db)
    # test the function with random input sizes
    n = np.random.randint(2,10)
    p = np.random.randint(2,10)
    h = np.random.randint(2,10)
    xh = th.randn(n,p+h) 
    W = th.randn(p+h,h)
    b = th.randn(h)
    z = compute_z_o(xh,W,b)
    assert np.allclose(z.size(), [n,h])
#---------------------------------------------------
def test_compute_o_t():
    ''' (2 points) compute_o_t'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    z = th.tensor([
                   #---------- the hidden states for the first time sequence in the mini-batch at time step t ------
                   [0.0, 0.2, 1000.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.5,-0.2,-1000.],
                    ], requires_grad=True)
    o = compute_o_t(z)
    assert type(o) == th.Tensor 
    o_true =[[0.5000, 0.5498, 1.],
             [0.6225, 0.4502, 0.]]
    assert np.allclose(o.data,o_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = o.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_dz = [[0.2500, 0.2475, 0.],
             [0.2350, 0.2475, 0.]]
    assert np.allclose(z.grad, dL_dz, atol= 0.001)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    z = th.randn(n,h)
    o = compute_o_t(z)
    assert np.allclose(o.size(),(n,h))
#---------------------------------------------------
def test_compute_ht():
    ''' (2 points) compute_ht'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    C = th.tensor([
                   #---------- the new cell states for the first time sequence in the mini-batch at time step t ------
                   [0.0, 1., 2.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.5,-1.,-2.],
                    ], requires_grad=True)
    o = th.tensor([
                   #---------- the new cell states for the first time sequence in the mini-batch at time step t ------
                   [0.3, .1, 1.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.8, 1., 0.],
                    ], requires_grad=True)
    ht = compute_ht(C,o)
    assert type(ht) == th.Tensor 
    ht_true =[[ 0.0000,  0.0762,  0.9640],
              [ 0.3697, -0.7616, -0.0000]]
    assert np.allclose(ht.data,ht_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = ht.sum()
    # back propagation
    L.backward()
    # the gradient for C and o
    dL_dC = [[0.3000, 0.0420, 0.0707],
             [0.6292, 0.4200, 0.0000]]
    dL_do = [[ 0.0000,  0.7616,  0.9640],
             [ 0.4621, -0.7616, -0.9640]]
    assert np.allclose(C.grad, dL_dC, atol= 0.01)
    assert np.allclose(o.grad, dL_do, atol= 0.01)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    C = th.randn(n,h)
    o = th.rand(n,h)
    ht = compute_ht(C,o)
    assert np.allclose(ht.size(),(n,h))
#---------------------------------------------------
def test_step():
    ''' (2 points) step'''
    
    # 2 time sequences of 3 input features at the current time step t 
    # n = 2, p = 3 
    xt = th.tensor([
                   #---------- the first time sequence in the mini-batch at time step t ------
                   [0.2,0.4,0.6],
                   #---------- the second time sequence in the mini-batch at time step t ------
                   [0.3,0.6,0.9],
                    ])
    # hidden states of 2 neurons after the previous step t-1
    # h = 2
    ht_1 = th.tensor([[ 0.5,-0.4],  # the hidden states for the first time sequence in the mini-batch
                      [-0.3, 0.6]], # the hidden states for the second time sequence in the mini-batch
                     requires_grad=True) 
    W_f = th.tensor([[ 0.1, 0.2],
                     [ 0.2, 0.4],
                     [ 0.3, 0.6],
                     [ 0.4, 0.8],
                     [ 0.5, 1.0]],
                   requires_grad=True) 
    b_f = th.tensor([0.2,-0.2], requires_grad=True)
    W_i = th.tensor([[-1.0, 1.0],
                     [ 1.0,-1.0],
                     [-1.0, 1.0],
                     [ 1.0,-1.0],
                     [-1.0, 1.0]],
                   requires_grad=True) 
    b_i = th.tensor([0.5,-0.5], requires_grad=True)
    W_c = th.tensor([[ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1]],
                   requires_grad=True) 
    b_c = th.tensor([-0.2, 0.2], requires_grad=True)
    W_o = th.tensor([[ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3]],
                   requires_grad=True) 
    b_o = th.tensor([-0.1, 0.1], requires_grad=True)
    # cell states of 2 cells after the previous step t-1
    Ct_1 = th.tensor([[ 1.0,-1.0],  # the cell states for the first time sequence in the mini-batch
                      [-0.6, 0.6]], # the cell states for the second time sequence in the mini-batch
                     requires_grad=True) 
    Ct, ht = step(xt,ht_1,Ct_1,W_f,b_f,W_i,b_i,W_c,b_c, W_o,b_o) 
    # check if the values are correct
    assert type(Ct) == th.Tensor 
    assert type(ht) == th.Tensor 
    assert np.allclose(Ct.size(),(2,2))
    assert np.allclose(ht.size(),(2,2))
    Ct_true= [[ 0.8440, -0.5034],
              [-0.2621,  0.7226]]
    ht_true= [[ 0.4362, -0.2882],
              [-0.1848,  0.4174]]
    assert np.allclose(Ct.data,Ct_true, atol = 0.01)
    assert np.allclose(ht.data,ht_true, atol = 0.01)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in Ct and ht)
    L = Ct.sum()+ht.sum()
    # back propagation
    L.backward()
    # the gradients 
    dL_dWo = [[ 0.0165,  0.0188],
              [ 0.0329,  0.0376],
              [ 0.0494,  0.0565],
              [ 0.0953, -0.0955],
              [-0.0948,  0.1252]]
    dL_dbo = [0.1081, 0.0262]
    dL_dWi = [[ 0.0720,  0.0511],
              [ 0.1440,  0.1021],
              [ 0.2160,  0.1532],
              [-0.0152,  0.0141],
              [ 0.0791,  0.0277]]
    dL_dbi = [0.2671, 0.2013]
    dL_dWc = [[ 0.2683,  0.3356],
              [ 0.5366,  0.6713],
              [ 0.8049,  1.0069],
              [ 0.3490, -0.0842],
              [-0.1688,  0.3840]]
    dL_dbc = [1.1883, 1.2385]
    dL_dWf = [[-0.0014, -0.0218],
              [-0.0029, -0.0436],
              [-0.0043, -0.0654],
              [ 0.2220, -0.2300],
              [-0.2549,  0.2442]]
    dL_dbf = [ 0.1002, -0.1926]
    dL_dht_1 = [[0.2619, 0.2452],
                [0.3510, 0.2075]]
    dL_dCt_1 = [[0.8241, 0.8754],
                [1.1549, 1.0356]]
    assert np.allclose(W_o.grad, dL_dWo, atol= 0.01)
    assert np.allclose(b_o.grad, dL_dbo, atol= 0.01)
    assert np.allclose(W_i.grad, dL_dWi, atol= 0.01)
    assert np.allclose(b_i.grad, dL_dbi, atol= 0.01)
    assert np.allclose(W_c.grad, dL_dWc, atol= 0.01)
    assert np.allclose(b_c.grad, dL_dbc, atol= 0.01)
    assert np.allclose(W_f.grad, dL_dWf, atol= 0.01)
    assert np.allclose(b_f.grad, dL_dbf, atol= 0.01)
    assert np.allclose(ht_1.grad, dL_dht_1, atol= 0.01)
    assert np.allclose(Ct_1.grad, dL_dCt_1, atol= 0.01)
#---------------------------------------------------
def test_compute_z():
    ''' (2 points) compute_z'''
    # batch_size = 4
    # number of classes c = 3
    # number of hidden states h = 2
    # NOTE: this is just an example, in real cases, the values in ht will always be -1 <= ht <= 1
    ht =th.tensor([[1.,1.], # the first sample in the mini-batch
                   [2.,2.], # the second sample in the mini-batch
                   [3.,3.], # the third sample in the mini-batch
                   [4.,4.]],# the fourth sample in the mini-batch
                 requires_grad=True)
    # weight matrix of shape (2 x 3) or (h x c)
    W = th.tensor([[ 0.5, 0.1,-0.2],
                   [-0.6, 0.0, 0.3]],requires_grad=True)
    # bias vector of length 3 (c)
    b = th.tensor([0.2,-0.3,-0.1],requires_grad=True) 
    z = compute_z(ht,W,b)
    assert type(z) == th.Tensor 
    assert np.allclose(z.size(), (4,3)) # batch_size x c 
    z_true = [[ 0.1,-0.2, 0.0], # linear logits for the first sample in the mini-batch
              [ 0.0,-0.1, 0.1], # linear logits for the second sample in the mini-batch
              [-0.1, 0.0, 0.2], # linear logits for the third sample in the mini-batch
              [-0.2, 0.1, 0.3]] # linear logits for the fourth sample in the mini-batch
    assert np.allclose(z.data,z_true, atol = 1e-2)
    assert z.requires_grad
    # check if the gradients of W is connected to z correctly
    L = th.sum(z) # compute the sum of all elements in z
    L.backward() # back propagate gradient to W and b
    # now the gradients dL_dW should be
    dL_dW_true = [[10,10,10],
                  [10,10,10]]
    # here [10,10] in each column of dL_dW is computed as the sum of the gradients in all the four samples.
    # for the 1st sample, the gradient is x = [1,1]
    # for the 2nd sample, the gradient is x = [2,2]
    # for the 3rd sample, the gradient is x = [3,3]
    # for the 4th sample, the gradient is x = [4,4]
    # so the sum of the gradients  will be [10,10] for each class (column of dL_dW matrix)
    assert np.allclose(W.grad,dL_dW_true, atol=0.1)
    # now the gradients of dL_db should be
    dL_db_true = [4,4,4]
    # here each element (4) of dL_db is computed as the sum of the gradients in all the four samples: 1+1+1+1 = 4
    assert np.allclose(b.grad,dL_db_true, atol=0.1)
#---------------------------------------------------
def test_forward():
    ''' (2 points) forward'''
    # 2 time sequences of 3 time steps with 2 input features at each time step 
    # n = 3, l=3 p = 2
    x = th.tensor([
                     #---------- the first time sequence in the mini-batch ------
                     [
                       [1.,0.], # the first time step of the time sequence
                       [0.,1.], # the second time step of the time sequence
                       [1.,0.]  # the third time step of the time sequence
                     ],
                     #---------- the second time sequence in the mini-batch ------
                     [
                       [1.,0.], # the first time step of the time sequence
                       [1.,0.], # the second time step of the time sequence
                       [0.,1.]  # the third time step of the time sequence
                     ],
                     #---------- the third time sequence in the mini-batch ------
                     [
                       [0.,1.], # the first time step of the time sequence
                       [1.,0.], # the second time step of the time sequence
                       [0.,1.]  # the third time step of the time sequence
                     ]
                     #------------------------------------------------------------
                   ])
    #---------------------------
    # Layer 1: Recurrent layer
    #---------------------------
    # 2 hidden states 
    # h = 2
    W_f = th.tensor([[ 0.1, 0.2],
                     [ 0.2, 0.4],
                     [ 0.3, 0.6],
                     [ 0.5, 1.0]],
                   requires_grad=True) 
    b_f = th.tensor([0.2,-0.2], requires_grad=True)
    W_i = th.tensor([[-1.0, 1.0],
                     [ 1.0,-1.0],
                     [ 1.0,-1.0],
                     [-1.0, 1.0]],
                   requires_grad=True) 
    b_i = th.tensor([0.5,-0.5], requires_grad=True)
    W_c = th.tensor([[ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1]],
                   requires_grad=True) 
    b_c = th.tensor([-0.2, 0.2], requires_grad=True)
    W_o = th.tensor([[ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3]],
                   requires_grad=True) 
    b_o = th.tensor([-0.1, 0.1], requires_grad=True)
    # initial hidden and cell states of 2 neurons on 3 time sequences 
    ht = th.zeros(3,2,requires_grad=True) 
    Ct = th.zeros(3,2,requires_grad=True) 
    #---------------------------
    # Layer 2: Fully-connected layer
    #---------------------------
    W = th.tensor([
                    [-1., 1.,-1.], 
                    [-1.,-1., 1.]],
                    requires_grad=True)
    b = th.tensor([0.,0.1,-0.1], requires_grad=True)
    z = forward(x,ht,Ct,W_f,b_f,W_i,b_i,W_c,b_c,W_o,b_o,W,b)
    assert type(z) == th.Tensor 
    assert np.allclose(z.size(),(3,3))
    z_true =[[-0.3315,  0.0932, -0.0932],
             [-0.3394,  0.1516, -0.1516],
             [-0.3257,  0.2012, -0.2012]]
    assert np.allclose(z.data,z_true, atol=1e-3)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = z.sum()
    # back propagation
    L.backward()
    # the gradient for the parameters
    dL_dWo = [[-0.0849, -0.1115],
              [-0.1673, -0.1080],
              [-0.0264, -0.0220],
              [-0.0310, -0.0248]]
    dL_dbo = [-0.2522, -0.2194]
    dL_dWi = [[-0.1347, -0.1648],
              [-0.0789, -0.0928],
              [-0.0161, -0.0182],
              [-0.0177, -0.0200]]
    dL_dbi = [-0.2137, -0.2576]
    dL_dWc = [[-0.8382, -1.2995],
              [-1.4577, -0.3363],
              [-0.1555, -0.1116],
              [-0.2071, -0.1103]]
    dL_dbc = [-2.2959, -1.6357]
    dL_dWf = [[-0.0641, -0.0531],
              [-0.0512, -0.0901],
              [-0.0124, -0.0130],
              [-0.0123, -0.0179]]
    dL_dbf = [-0.1153, -0.1432]
    dL_dW = [[0.5713, 0.5713, 0.5713],
             [0.4253, 0.4253, 0.4253]]
    dL_db =  [3., 3., 3.]
    dL_dht = [[-0.0726, -0.0863],
              [-0.0703, -0.0843],
              [-0.1151, -0.1247]]
    dL_dCt = [[-0.2187, -0.1790],
              [-0.2143, -0.1775],
              [-0.2122, -0.1928]]
    assert np.allclose(W_o.grad, dL_dWo, atol= 0.01)
    assert np.allclose(b_o.grad, dL_dbo, atol= 0.01)
    assert np.allclose(W_i.grad, dL_dWi, atol= 0.01)
    assert np.allclose(b_i.grad, dL_dbi, atol= 0.01)
    assert np.allclose(W_c.grad, dL_dWc, atol= 0.01)
    assert np.allclose(b_c.grad, dL_dbc, atol= 0.01)
    assert np.allclose(W_f.grad, dL_dWf, atol= 0.01)
    assert np.allclose(b_f.grad, dL_dbf, atol= 0.01)
    assert np.allclose(ht.grad, dL_dht, atol= 0.01)
    assert np.allclose(Ct.grad, dL_dCt, atol= 0.01)
    # test the function with random input sizes
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    h = np.random.randint(2,10) # number of hidden states 
    l = np.random.randint(2,10) # number of time steps in a sequence 
    p = np.random.randint(2,10) # number of input features at each time step 
    c = np.random.randint(2,10) # number of classes in the classification task 
    x  = th.randn(n,l,p)
    ht = th.randn(n,h)
    Ct = th.randn(n,h)
    W_f  = th.randn(p+h,h)
    b_f  = th.randn(h)
    W_i  = th.randn(p+h,h)
    b_i  = th.randn(h)
    W_o  = th.randn(p+h,h)
    b_o  = th.randn(h)
    W_c  = th.randn(p+h,h)
    b_c  = th.randn(h)
    W = th.randn(h,c) 
    b = th.randn(c) 
    z = forward(x,ht,Ct,W_f,b_f,W_i,b_i,W_c,b_c,W_o,b_o,W,b)
    assert np.allclose(z.size(),(n,c))
#---------------------------------------------------
def test_compute_L():
    ''' (2 points) compute_L'''
    # batch_size = 4
    # number of classes c = 3
    # linear logits in a mini-batch:  shape (4 x 3) or (batch_size x c)
    z = th.tensor([[ 0.1,-0.2, 0.0], # linear logits for the first sample in the mini-batch
                   [ 0.0,-0.1, 0.1], # linear logits for the second sample in the mini-batch
                   [-0.1, 0.0, 0.2], # linear logits for the third sample in the mini-batch
                   [-0.2, 0.1, 0.3]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.LongTensor([1,2,1,0])
    L = compute_L(z,y)
    assert type(L) == th.Tensor 
    assert L.requires_grad
    assert np.allclose(L.detach().numpy(),1.2002,atol=1e-4) 
    # check if the gradients of z is connected to L correctly
    L.backward() # back propagate gradient to W and b
    dL_dz_true = [[ 0.0945, -0.1800,  0.0855],
                  [ 0.0831,  0.0752, -0.1582],
                  [ 0.0724, -0.1700,  0.0977],
                  [-0.1875,  0.0844,  0.1031]]
    assert np.allclose(z.grad,dL_dz_true, atol=0.01)
    #-----------------------------------------    
    # batch_size = 3
    # number of classes c = 3
    # linear logits in a mini-batch:  shape (3 x 3) or (batch_size x c)
    z = th.tensor([[  0.1,-1000, 1000], # linear logits for the first sample in the mini-batch
                   [  0.0, 1100, 1000], # linear logits for the second sample in the mini-batch
                   [-2000,-1900,-5000]], requires_grad=True) # linear logits for the last sample in the mini-batch
    y = th.LongTensor([2,1,1])
    L = compute_L(z,y)
    assert np.allclose(L.data,0,atol=1e-4) 
    #-----------------------------------------    
    # batch_size = 2
    # number of classes c = 3
    # linear logits in a mini-batch:  shape (2 x 3) or (batch_size x c)
    z = th.tensor([[  0.1,-1000, 1000], # linear logits for the first sample in the mini-batch
                   [-2000,-1900,-5000]], requires_grad=True) # linear logits for the last sample in the mini-batch
    y = th.LongTensor([0,2])
    L = compute_L(z,y)
    assert L.data >100
    assert L.data < float('inf')
#---------------------------------------------------
def test_update_parameters():
    ''' (2 points) update_parameters'''
    #---------------------------
    # Layer 1: Recurrent layer
    #---------------------------
    # 2 hidden states 
    # h = 2
    W_f = th.tensor([[ 0.1, 0.2],
                     [ 0.2, 0.4],
                     [ 0.3, 0.6],
                     [ 0.5, 1.0]],
                   requires_grad=True) 
    b_f = th.tensor([0.2,-0.2], requires_grad=True)
    W_i = th.tensor([[-1.0, 1.0],
                     [ 1.0,-1.0],
                     [ 1.0,-1.0],
                     [-1.0, 1.0]],
                   requires_grad=True) 
    b_i = th.tensor([0.5,-0.5], requires_grad=True)
    W_c = th.tensor([[ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1],
                     [ 0.4, 0.1]],
                   requires_grad=True) 
    b_c = th.tensor([-0.2, 0.2], requires_grad=True)
    W_o = th.tensor([[ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3],
                     [ 0.5, 0.3]],
                   requires_grad=True) 
    b_o = th.tensor([-0.1, 0.1], requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer
    #---------------------------
    W = th.tensor([
                    [-1., 1.,-1.], 
                    [-1.,-1., 1.]],
                    requires_grad=True)
    b = th.tensor([0.,0.1,-0.1], requires_grad=True)
    # create a toy loss function: the sum of all elements in all parameters 
    L = W_f.sum()+ b_f.sum() + W_i.sum() + b_i.sum() + W_o.sum()+b_o.sum()+ W_c.sum()+b_c.sum()+W.sum()+b.sum()
    # back propagation to compute the gradients
    L.backward()
    # now the gradients for all parameters should be all-ones
    # let's try updating the parameters with gradient descent
    # create an optimizer for the parameters with learning rate = 0.1
    optimizer = th.optim.SGD([W_f,b_f,W_i,b_i,W_o,b_o,W_c,b_c,W,b], lr=0.1)
    # now perform gradient descent using SGD
    update_parameters(optimizer)
    # let's check the new values of the parameters 
    W_f_ = [[ 0.0, 0.1],
            [ 0.1, 0.3],
            [ 0.2, 0.5],
            [ 0.4, 0.9]]
    b_f_= [0.1,-0.3]
    W_i_= [[-1.1, 0.9],
           [ 0.9,-1.1],
           [ 0.9,-1.1],
           [-1.1, 0.9]]
    b_i_= [0.4,-0.6]
    W_c_= [[ 0.3, 0.0],
           [ 0.3, 0.0],
           [ 0.3, 0.0],
           [ 0.3, 0.0]]
    b_c_= [-0.3, 0.1]
    W_o_= [[ 0.4, 0.2],
           [ 0.4, 0.2],
           [ 0.4, 0.2],
           [ 0.4, 0.2]]
    b_o_= [-0.2, 0.0]
    W_= [ [-1.1, 0.9,-1.1], 
          [-1.1,-1.1, 0.9]]
    b_= [-0.1,0.0,-0.2]
    assert np.allclose(W_f.data,W_f_,atol=1e-2) 
    assert np.allclose(b_f.data,b_f_,atol=1e-2) 
    assert np.allclose(W_i.data,W_i_,atol=1e-2) 
    assert np.allclose(b_i.data,b_i_,atol=1e-2) 
    assert np.allclose(W_o.data,W_o_,atol=1e-2) 
    assert np.allclose(b_o.data,b_o_,atol=1e-2) 
    assert np.allclose(W_c.data,W_c_,atol=1e-2) 
    assert np.allclose(b_c.data,b_c_,atol=1e-2) 
    assert np.allclose(W.data,W_,atol=1e-2) 
    assert np.allclose(b.data,b_,atol=1e-2) 
    assert np.allclose(W_f.grad,np.zeros((4,2)),atol=1e-2) 
    assert np.allclose(b_f.grad,np.zeros(2),atol=1e-2) 
    assert np.allclose(W_i.grad,np.zeros((4,2)),atol=1e-2) 
    assert np.allclose(b_i.grad,np.zeros(2),atol=1e-2) 
    assert np.allclose(W_o.grad,np.zeros((4,2)),atol=1e-2) 
    assert np.allclose(b_o.grad,np.zeros(2),atol=1e-2) 
    assert np.allclose(W_c.grad,np.zeros((4,2)),atol=1e-2) 
    assert np.allclose(b_c.grad,np.zeros(2),atol=1e-2) 
    assert np.allclose(W.grad,np.zeros((2,3)),atol=1e-2) 
    assert np.allclose(b.grad,np.zeros(3),atol=1e-2) 
#---------------------------------------------------
def test_train():
    ''' (2 points) train'''
    # n = 4, l=3, p = 2, c=3
    X  = [
          [ # instance 0
            [0.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ], 
          [ # instance 1
            [0.,1.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ],
          [ # instance 2
            [1.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ],
          [ # instance 3
            [1.,0.], # time step 0 
            [0.,1.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ] 
         ]
    Y = [0,1,2,0]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.LongTensor(Y)
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    h=64
    n=2
    loader = th.utils.data.DataLoader(d, batch_size = n,shuffle=True)
    W_f,b_f,W_i,b_i,W_c,b_c,W_o,b_o,W,b = train(loader,p=2,h=h,n = n,c=3,n_epoch=100)
    ht = th.zeros(4,h) # initialize the hidden states as all zero
    Ct = th.zeros(4,h) # initialize the cells states as all zero
    z = forward(th.Tensor(X),ht,Ct,W_f,b_f,W_i,b_i,W_c,b_c,W_o,b_o,W,b)
    assert z[0,0] == max(z[0])
    assert z[1,1] == max(z[1])
    assert z[2,2] == max(z[2])
    assert z[3,0] == max(z[3])
#---------------------------------------------------
def test_predict():
    ''' (2 points) predict'''
    # n = 4, l=3, p = 2, c=3
    X  = [
          [ # instance 0
            [0.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ], 
          [ # instance 1
            [0.,1.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ],
          [ # instance 2
            [1.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ],
          [ # instance 3
            [1.,0.], # time step 0 
            [0.,1.], # time step 1
            [0.,0.], # time step 2
            [0.,0.], # time step 3
            [0.,0.], # time step 4
            [0.,0.], # time step 5
            [0.,0.], # time step 6
            [0.,0.], # time step 7
            [0.,0.], # time step 8
            [0.,0.]  # time step 9
          ] 
         ]
    Y = [0,1,2,0]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.LongTensor(Y)
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    h=64
    n=2
    loader = th.utils.data.DataLoader(d, batch_size = n,shuffle=True)
    W_f,b_f,W_i,b_i,W_c,b_c,W_o,b_o,W,b = train(loader,p=2,h=h,n = n,c=3,n_epoch=100)
    y_predict = predict(th.Tensor(X),W_f,b_f,W_i,b_i,W_c,b_c,W_o,b_o,W,b)
    assert np.allclose(y_predict, Y)

