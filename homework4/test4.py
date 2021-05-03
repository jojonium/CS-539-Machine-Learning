from problem4 import *
import sys
import math
from game import *
from torch import Tensor
import random
'''
    Unit test 4:
    This file includes unit tests for problem4.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 4 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z1():
    ''' (3 points) compute_z1'''
    # 2 images of 6 by 8 pixels with 3 color/input channels  (shape: 3 channel x 6 height x 8 width )
    # n = 2, c = 3, h = 6, w = 8 
    S = th.tensor([
                   #---------- the first image in the mini-batch ------
                   [[[0.,0.,1.,0.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,0.,1.,0.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,0.,1.,0.,0.],
                     [0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,0.,1.,0.,0.]], # the first/red channel of the image 
                    [[0.,1.,1.,1.,0.,1.,0.,0.],
                     [0.,0.,0.,0.,1.,1.,1.,0.],
                     [0.,1.,1.,1.,0.,1.,0.,0.],
                     [0.,0.,1.,0.,1.,1.,1.,0.],
                     [0.,1.,1.,1.,0.,0.,0.,0.],
                     [0.,0.,1.,0.,1.,1.,1.,0.]],# the second/green channel of the image
                    [[0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,0.,1.,0.,0.,1.,0.,0.],
                     [0.,0.,1.,0.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,0.,1.,0.,0.,1.,0.,0.],
                     [0.,1.,0.,1.,0.,1.,0.,0.]]], # the third/blue channel of the image
                   #---------- the second image in the mini-batch ------
                   [[[1.,0.,1.,0.,0.,0.,1.,0.],
                     [1.,0.,1.,0.,0.,1.,0.,1.],
                     [1.,0.,1.,0.,0.,0.,1.,0.],
                     [0.,1.,0.,0.,0.,1.,0.,1.],
                     [1.,0.,1.,0.,0.,1.,0.,1.],
                     [0.,1.,0.,0.,0.,1.,0.,1.]], # the first/red channel of the image 
                    [[0.,1.,0.,0.,0.,1.,1.,1.],
                     [1.,1.,1.,0.,0.,0.,0.,0.],
                     [0.,1.,0.,0.,0.,1.,1.,1.],
                     [1.,1.,1.,0.,0.,0.,1.,0.],
                     [0.,0.,0.,0.,0.,1.,1.,1.],
                     [1.,1.,1.,0.,0.,0.,1.,0.]],# the second/green channel of the image
                    [[1.,0.,1.,0.,0.,1.,0.,1.],
                     [0.,1.,0.,0.,0.,0.,1.,0.],
                     [1.,0.,1.,0.,0.,0.,1.,0.],
                     [1.,0.,1.,0.,0.,1.,0.,1.],
                     [0.,1.,0.,0.,0.,0.,1.,0.],
                     [0.,1.,0.,0.,0.,1.,0.,1.]]] # the third/blue channel of the image
                    ])
    # 2 filters of shape 3 x 3 with 3 channels    (shape: 2 filters x 3 channels x 3 hight x 3 width)
    W = th.tensor( [
                        #---------- the first filter with 3 input channels ------
                        [
                         [[1.,0.,1.],
                          [1.,0.,1.],
                          [1.,0.,1.]], # the first channel of the filter, trying to match a red-colored pattern '| |' 
                         [[0.,1.,0.],
                          [1.,1.,1.],
                          [0.,1.,0.]], # the second channel of the filter, trying to match a green-colored pattern '+'
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]]  # the third channel of the filter, trying to match a blue-colored pattern 'X'
                         ],
                        #---------- the second filter with 3 input channels ------
                        [
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the first channel of the filter, trying to match a red-colored pattern 'O' 
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [1.,1.,1.]], # the second channel of the filter, trying to match a green-colored pattern '='
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [0.,1.,0.]]  # the third channel of the filter, trying to match a blue-colored pattern 'X'
                         ]
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b = th.tensor([-15.,   # the bias for the first filter
                   -13.],  # the bias for the second filter
                   requires_grad=True)
    z = compute_z1(S,W,b)
    assert type(z) == Tensor 
    assert np.allclose(z.size(),(2,2,4,6))
               #---------- the output on the first image in the mini-batch ------
    z_true =  [[[[-10.,  -8.,  -4., -11.,   1., -13.],
                 [-11.,  -4.,  -8.,  -5.,  -6.,  -9.],
                 [ -8.,  -8.,  -4.,  -8.,  -5., -10.],
                 [-13.,   1., -11.,  -4.,  -8., -10.]], # outputs of the first filter
                [[ -9.,   1.,  -7.,  -4.,  -6.,  -8.],
                 [ -7., -11.,  -4.,  -5.,  -4.,  -5.],
                 [ -6.,  -3.,  -5.,  -5.,  -9.,  -8.],
                 [ -8.,  -6.,  -4.,  -7.,   1.,  -9.]]], # outputs of the second filter
               #---------- the output on the second image in the mini-batch ------
               [[[  1., -13.,  -9., -13., -10.,  -8.],
                 [ -6.,  -9., -12., -11., -11.,  -4.],
                 [ -5., -10., -11., -13.,  -8.,  -8.],
                 [ -8., -10., -13.,  -9., -13.,   1.]], # outputs of the first filter
                [[ -6.,  -8., -11.,  -9.,  -9.,   1.],
                 [ -4.,  -5., -10., -13.,  -7., -11.],
                 [ -9.,  -8., -12., -10.,  -6.,  -3.],
                 [  1.,  -9.,  -9., -11.,  -8.,  -6.]]]] # outputs of the second filter
    assert np.allclose(z.data,z_true, atol=0.1)
    #-----------------------
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in z)
    L = z.sum()
    # back propagation
    L.backward()
    # check the gradients of w, b
    dL_dW_true = [[[[19., 21., 20.],
                    [21., 22., 21.],
                    [20., 21., 19.]],
                   [[23., 26., 22.],
                    [21., 24., 21.],
                    [22., 26., 23.]],
                   [[19., 21., 20.],
                    [16., 19., 17.],
                    [18., 20., 18.]]],
                  [[[19., 21., 20.],
                    [21., 22., 21.],
                    [20., 21., 19.]],
                   [[23., 26., 22.],
                    [21., 24., 21.],
                    [22., 26., 23.]],
                   [[19., 21., 20.],
                    [16., 19., 17.],
                    [18., 20., 18.]]]]
    assert np.allclose(W.grad, dL_dW_true, atol= 0.1)
    assert np.allclose(b.grad, [48,48], atol= 0.1)
    #-----------------------
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    s = np.random.randint(2,5) # size of the filter 
    h = s+np.random.randint(5,20) # hight of the image 
    w = s+np.random.randint(5,20) # width of the image 
    l = np.random.randint(2,10) # number of channels 
    n_filters = np.random.randint(2,10) # number of filters 
    S  = th.randn(n,l,h,w)
    W  = th.randn(n_filters, l,s,s)
    b = th.randn(n_filters)
    z = compute_z1(S,W,b) 
    assert np.allclose(z.size(),(n,n_filters, h-s+1,w-s+1))
#---------------------------------------------------
def test_compute_a1():
    ''' (2 points) compute_a1'''
    
    # n=2, 2 filters, h = 4, w = 6
               #---------- the linear logits on the first image in the mini-batch ------
    z =th.tensor([[[[-10.,  -8.,  -4., -11.,   1., -13.],
                    [-11.,  -4.,  -8.,  -5.,  -6.,  -9.],
                    [ -8.,  -8.,  -4.,  -8.,  -5., -10.],
                    [-13.,   1., -11.,  -4.,  -8., -10.]], # outputs of the first filter
                   [[ -9.,   1.,  -7.,  -4.,  -6.,  -8.],
                    [ -7., -11.,  -4.,  -5.,  -4.,  -5.],
                    [ -6.,  -3.,  -5.,  -5.,  -9.,  -8.],
                    [ -8.,  -6.,  -4.,  -7.,   1.,  -9.]]], # outputs of the second filter
                  #---------- the linear logits on the second image in the mini-batch ------
                  [[[  1., -13.,  -9., -13., -10.,  -8.],
                    [ -6.,  -9., -12., -11., -11.,  -4.],
                    [ -5., -10., -11., -13.,  -8.,  -8.],
                    [ -8., -10., -13.,  -9., -13.,   1.]], # outputs of the first filter
                   [[ -6.,  -8., -11.,  -9.,  -9.,   1.],
                    [ -4.,  -5., -10., -13.,  -7., -11.],
                    [ -9.,  -8., -12., -10.,  -6.,  -3.],
                    [  1.,  -9.,  -9., -11.,  -8.,  -6.]]]], # outputs of the second filter
                   requires_grad=True)
    a = compute_a1(z)
    # check value 
    assert type(a) == Tensor 
    assert np.allclose(a.size(),(2,2,4,6))
               #---------- the activations on the first image in the mini-batch ------
    a_true =  [[[[0., 0., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0., 0.]],
                [[0., 1., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0.]]],
               #---------- the activations on the second image in the mini-batch ------
               [[[1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 1.]],
                [[0., 0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.]]]]
    assert np.allclose(a.data, a_true)
    #-----------------------
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in a)
    L = a.sum()
    # back propagation
    L.backward()
    # check the gradients dL_dz, which happens to equal to a_true in this test case.
    assert np.allclose(z.grad, a_true, atol= 0.1)
    #-----------------------
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20) # hight of the image 
    w = np.random.randint(5,20) # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    z = th.randn(n,n_filters,h,w)
    a = compute_a1(z) 
    assert np.allclose(a.size(),(n,n_filters, h,w))
#---------------------------------------------------
def test_compute_p():
    ''' (2 points) compute_p'''
                   #---------- the activations on the first image in the mini-batch ------
    a = th.tensor([[[[0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 0.]],
                    [[0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1., 0.]]],
                   #---------- the activations on the second image in the mini-batch ------
                   [[[1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 1.]],
                    [[0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0.]]]],
                    requires_grad=True)
    p = compute_p(a)
    # check value 
    assert type(p) == Tensor 
    assert np.allclose(p.size(),(2,2,2,3))
              #---------- the pooled features on the first image in the mini-batch ------
    p_true  = [[[[0., 0., 1.],
                 [1., 0., 0.]],
                [[1., 0., 0.],
                 [0., 0., 1.]]],
              #---------- the pooled features on the second image in the mini-batch ------
               [[[1., 0., 0.],
                 [0., 0., 1.]],
                [[0., 0., 1.],
                 [1., 0., 0.]]]]
    assert np.allclose(p.data, p_true)
    #-----------------------
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in p)
    L = p.sum()
    # back propagation
    L.backward()
    # check the gradients of w, b
    dL_da_true = [[[[1., 0., 1., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 1., 0.],
                    [0., 1., 0., 0., 0., 0.]],
                   [[0., 1., 1., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [1., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0.]]],
                  [[[1., 0., 1., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 0.],
                    [1., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1.]],
                   [[1., 0., 1., 0., 0., 1.],
                    [0., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 1., 0.],
                    [1., 0., 0., 0., 0., 0.]]]]
    assert np.allclose(a.grad, dL_da_true, atol= 0.1)
    #-----------------------
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20)*2 # hight of the image 
    w = np.random.randint(5,20)*2 # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    a = th.randn(n,n_filters,h,w)
    p = compute_p(a)
    assert np.allclose(p.size(),(n,n_filters, h/2,w/2))
    #-----------------------
    # check gradient with multiple max values
    a = th.tensor([[[[ 0., 1.],
                     [ 1., 0.]]]],requires_grad=True)
    p = compute_p(a)
    t = p.sum()
    t.backward()
    dL_da_true = [[[[ 0., 1.],
                    [ 0., 0.]]]]
    assert np.allclose(a.grad,dL_da_true,atol=1e-2)
#---------------------------------------------------
def test_flatten():
    ''' (2 points) flatten'''
               #---------- the pooling results on the first image in the mini-batch ------
    p = th.tensor([[[[0., 0., 1.],
                     [1., 0., 0.]],
                    [[1., 0., 0.],
                     [0., 0., 1.]]],
               #---------- the pooling results on the second image in the mini-batch ------
                   [[[1., 0., 0.],
                     [0., 0., 1.]],
                    [[0., 0., 1.],
                     [1., 0., 0.]]]],requires_grad=True)
    f = flatten(p)
    # check value 
    assert type(f) == Tensor 
    assert np.allclose(f.size(),(2,12))
    f_true  = [[0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1.], # flat feature of the first image in the mini-batch
               [1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0.]] # flat feature of the second image in the mini-batch
    assert np.allclose(f.data, f_true,atol=0.1) 
    #-----------------------
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in f)
    L = f.sum()
    # back propagation
    L.backward()
    # check the gradients of w, b
    dL_dp_true = [[[[1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.]]],
                  [[[1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.]]]]
    assert np.allclose(p.grad, dL_dp_true, atol= 0.1)
    #-----------------------
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20) # hight of the image 
    w = np.random.randint(5,20) # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    p = th.randn(n,n_filters,h,w)
    f = flatten(p)
    assert np.allclose(f.size(),(n,n_filters*h*w))
#---------------------------------------------------
def test_compute_z2():
    ''' (2 points) compute_z2'''
    # batch_size = 4
    # number of input features  = 2
    # input feature to the second layer on one mini-batch: 4 (batch_size) by 2 (p) matrix
    f = th.tensor([[1.,1.], # the first sample in the mini-batch
                   [2.,2.], # the second sample in the mini-batch
                   [3.,3.], # the third sample in the mini-batch
                   [4.,4.]])# the fourth sample in the mini-batch
    # weights of length 2
    W2 = th.tensor([[0.1,0.2,0.3],
                    [0.6,0.5,0.4]],requires_grad=True)
    # bias
    b2 = th.tensor([-0.1,0, 0.1],requires_grad=True) 
    z2 = compute_z2(f,W2,b2)
    assert type(z2) == th.Tensor 
    assert np.allclose(z2.size(), (4,3)) # batch_size
    z_true = [[0.6, 0.7, 0.8],
              [1.3, 1.4, 1.5],
              [2.0, 2.1, 2.2],
              [2.7, 2.8, 2.9]]
    assert np.allclose(z2.data,z_true, atol = 1e-1)
    assert z2.requires_grad
    #-----------------------
    # check if the gradients of W is connected to z correctly
    L = th.sum(z2) # compute the sum of all elements in z
    L.backward() # back propagate gradient to W and b
    # now the gradients dL_dW should be
    dL_dW_true = [[10., 10., 10.],
                  [10., 10., 10.]]
    # here [10,10] of dL_dW is computed as the sum of the gradients in all the four samples.
    # for the 1st sample, the gradient is x = [1,1]
    # for the 2nd sample, the gradient is x = [2,2]
    # for the 3rd sample, the gradient is x = [3,3]
    # for the 4th sample, the gradient is x = [4,4]
    # so the sum of the gradients  will be [10,10]
    assert np.allclose(W2.grad,dL_dW_true, atol=0.1)
    # now the gradients of dL_db should be: 4
    # here dL_db is computed as the sum of the gradients in all the four samples: 1+1+1+1 = 4
    assert np.allclose(b2.grad,[4,4,4], atol=0.1)
    #-----------------------
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    nf = np.random.randint(5,20) # number of flat features 
    c = np.random.randint(2,10) # number of actions 
    x = th.randn(n,nf)
    W = th.randn(nf,c)
    b = th.randn(c)
    z = compute_z2(x,W,b)
    assert np.allclose(z.size(),(n,c))
#---------------------------------------------------
def test_compute_Q():
    ''' (3 points) compute_Q'''
    
    # The shape of the tensors are as follows:
    # x: 2x3x4x4  (2 images, 3 channels, height 4, width 4)
    # Convolutional Layer:  2 filters of size 3x3 
    # z1: 2x2x2x2 (2 images, 2 filter channels, height 4, width 4)
    # a1: 2x2x2x2 (2 images, 2 filter channels, 2 height, 2 width)
    # p: 2x2x1x1 (2 images, 2 filter channels, 1 height, 1 width) 
    # f: 2x2 (2 images, 2 flattened features)
    # Q: 2x3 (2 images, 3 actions)
    S = th.tensor([
                    #---------- the first image in the mini-batch (game state type 1) ------
                    [ 
                        # the first/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.],
                         [1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ],
                    #---------- the second image in the mini-batch (game state type 2) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ] 
                    #----------------------------------------------------
                ])
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 input channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'object 1' detector ------
                        [
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the second channel (green color) of the filter
                         [[0.,0.,0.],
                          [1.,1.,1.],
                          [0.,0.,0.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'Object 2' detector ------
                        [
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the first channel of the filter
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the second channel of the filter
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ],
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-11.,  # the bias for the first filter
                   -11.], # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer (3 actions)
    #---------------------------
    W2 = th.tensor([[ 1., 0.,0.],
                    [-1., 0.,0.]], requires_grad=True)
    b2= th.tensor([0.,1.,2.],requires_grad=True) 
    Q = compute_Q(S,W1,b1,W2,b2) 
    # check value 
    assert type(Q) == Tensor 
    assert np.allclose(Q.size(),(2,3))
    Q_true  = [[ 1.,1.,2.],
               [-1.,1.,2.]]
    assert np.allclose(Q.data, Q_true)
    #---------------------------
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in p)
    L = Q.sum()
    # back propagation
    L.backward()
    # check the gradients
    dL_dW1 = [[[[ 1.,  0.,  1.],
                [ 0.,  1.,  0.],
                [ 1.,  0.,  1.]],
               [[ 0.,  1.,  0.],
                [ 1.,  0.,  1.],
                [ 0.,  1.,  0.]],
               [[ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 0.,  0.,  0.]]],
              [[[ 0., -1.,  0.],
                [-1.,  0., -1.],
                [ 0., -1.,  0.]],
               [[-1.,  0., -1.],
                [ 0., -1.,  0.],
                [-1.,  0., -1.]],
               [[-1., -1., -1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]]]
    dL_dW2 = [[1., 1., 1.],
              [1., 1., 1.]]
    assert np.allclose(W1.grad, dL_dW1, atol= 0.1)
    assert np.allclose(b1.grad, [1,-1], atol= 0.1)
    assert np.allclose(W2.grad, dL_dW2, atol= 0.1)
    assert np.allclose(b2.grad, [2,2,2], atol= 0.1)
    #---------------------------
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    s1 = np.random.randint(1,3)*2+1 # size of the filter 
    c0 = np.random.randint(2,4) # number of color channels 
    c1 = np.random.randint(2,4) # number of filters 
    c = np.random.randint(2,4) # number of actions 
    h1 = np.random.randint(2,4) # hight after CONV layer 
    w1 = np.random.randint(2,4) 
    h = h1*2 + s1 - 1 # hight of the image
    w = w1*2 + s1 - 1 # width of the image
    n_flat_features = c1*h1*w1
    S  = th.randn(n,c0,h,w)
    W1  = th.randn(c1,c0,s1,s1)
    b1 = th.randn(c1)
    W2  = th.randn(n_flat_features,c)
    b2 = th.zeros(c)
    Q= compute_Q(S,W1,b1,W2,b2) 
    assert np.allclose(Q.size(),(n,c))
#---------------------------------------------------
def test_compute_Qt():
    ''' (3 points) compute_Qt'''
    S_new = th.tensor([
                    #---------- the first image in the mini-batch (game state type 1) ------
                    [ 
                        # the first/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.],
                         [1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ],
                    #---------- the second image in the mini-batch (game state type 2) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ] 
                    #----------------------------------------------------
                ])
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 input channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'object 1' detector ------
                        [
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the second channel (green color) of the filter
                         [[0.,0.,0.],
                          [1.,1.,1.],
                          [0.,0.,0.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'Object 2' detector ------
                        [
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the first channel of the filter
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the second channel of the filter
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ],
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-11.,  # the bias for the first filter
                   -11.], # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer (3 actions)
    #---------------------------
    W2 = th.tensor([[ 1., 0.,0.],
                    [-1., 0.,0.]], requires_grad=True)
    b2= th.tensor([0.,1.,2.],requires_grad=True)
    R = th.tensor([.1,1000.])
    T = th.tensor([False,True])
    Qt = compute_Qt(S_new,R,T,W1,b1,W2,b2,gamma=0.5) 
    # check value 
    assert type(Qt) == Tensor 
    assert np.allclose(Qt.size(),(2,))
    Qt_true  = [1.1,1000.]
    assert np.allclose(Qt.data, Qt_true,atol=0.1)
    # check if the gradients are disconnected correctly
    assert Qt.requires_grad == False
    # check with another gamma value
    Qt = compute_Qt(S_new,R,T,W1,b1,W2,b2,gamma=0.9) 
    # check value 
    Qt_true  = [1.9,1000.]
    assert np.allclose(Qt.data, Qt_true,atol=0.1)
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    s1 = np.random.randint(1,3)*2+1 # size of the filter 
    c0 = np.random.randint(2,4) # number of color channels 
    c1 = np.random.randint(2,4) # number of filters 
    c = np.random.randint(2,4) # number of actions 
    h1 = np.random.randint(2,4) # hight after second CONV layer 
    w1 = np.random.randint(2,4) 
    h = h1*2 + s1 - 1 # hight of the image
    w = w1*2 + s1 - 1 # width of the image
    n_flat_features = c1*h1*w1
    R = th.randn(n)
    S_new  = th.randn(n,c0,h,w)
    T = th.tensor(np.random.randn(n)>0)
    W1  = th.randn(c1,c0,s1,s1)
    b1 = th.randn(c1)
    W2  = th.randn(n_flat_features,c)
    b2 = th.zeros(c)
    Qt = compute_Qt(S_new,R, T, W1,b1,W2,b2) 
    assert np.allclose(Qt.size(),(n,))
#---------------------------------------------------
def test_update_Q():
    ''' (3 points) update_Q'''
    # The shape of the tensors are as follows:
    # x: 2x3x4x4  (2 images, 3 channels, height 4, width 4)
    # Convolutional Layer:  2 filters of size 3x3 
    # z1: 2x2x2x2 (2 images, 2 filter channels, height 4, width 4)
    # a1: 2x2x2x2 (2 images, 2 filter channels, 2 height, 2 width)
    # p: 2x2x1x1 (2 images, 2 filter channels, 1 height, 1 width) 
    # f: 2x2 (2 images, 2 flattened features)
    # Q: 2x3 (2 images, 3 actions)
    S = th.tensor([
                    #---------- the first image in the mini-batch (game state type 1) ------
                    [ 
                        # the first/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.],
                         [1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ],
                    #---------- the second image in the mini-batch (game state type 2) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ] 
                    #----------------------------------------------------
                ])
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 input channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'object 1' detector ------
                        [
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the second channel (green color) of the filter
                         [[0.,0.,0.],
                          [1.,1.,1.],
                          [0.,0.,0.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'Object 2' detector ------
                        [
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the first channel of the filter
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the second channel of the filter
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ],
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-11.,  # the bias for the first filter
                   -11.], # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer (3 actions)
    #---------------------------
    W2 = th.tensor([[ 1., 0.,0.],
                    [-1., 0.,0.]], requires_grad=True)
    b2= th.tensor([0.,1.,2.],requires_grad=True) 
    # create an optimizer for the parameters with learning rate = 0.1
    optimizer = th.optim.SGD([W1,b1,W2,b2], lr=0.5)
    A = th.LongTensor([2,0]) # the actions chosen
    Qt = th.tensor([-2.,4.]) # the target Q values 
    # update Q
    update_Q(S,A,Qt,W1,b1,W2,b2,optimizer)
    W1_true =[[[[ 1.0,  0.0,  1.0],
                [ 0.0,  1.0,  0.0],
                [ 1.0,  0.0,  1.0]],
               [[ 0.0,  1.0,  0.0],
                [ 1.0,  0.0,  1.0],
                [ 0.0,  1.0,  0.0]],
               [[ 0.0,  0.0,  0.0],
                [ 1.0,  1.0,  1.0],
                [ 0.0,  0.0,  0.0]]],
              [[[ 0.0, -1.5,  0.0],
                [-1.5,  0.0, -1.5],
                [ 0.0, -1.5,  0.0]],
               [[-1.5,  0.0, -1.5],
                [ 0.0, -1.5,  0.0],
                [-1.5,  0.0, -1.5]],
               [[-1.5, -1.5, -1.5],
                [ 0.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0]]]]
    W2_true = [[ 1.0,  0., -2.],
               [ 1.5,  0.,  0.]]
    b1_true = [-11., -13.5]
    b2_true = [2.5, 1.,0.]
    assert np.allclose(W1.data,W1_true,atol=1e-2) 
    assert np.allclose(b1.data,b1_true,atol=1e-2) 
    assert np.allclose(W2.data,W2_true,atol=1e-2) 
    assert np.allclose(b2.data,b2_true,atol=1e-2) 
    assert np.allclose(W1.grad,np.zeros((2,3,3,3)),atol=1e-2) 
    assert np.allclose(b1.grad,np.zeros(2),atol=1e-2) 
    assert np.allclose(W2.grad,np.zeros((2,3)),atol=1e-2) 
    assert np.allclose(b2.grad,[0,0,0],atol=1e-2) 
#---------------------------------------------------
def test_predict_q():
    ''' (2 points) predict_q'''
    s = th.tensor( [ 
                        # the first/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.],
                         [1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ])
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 input channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'object 1' detector ------
                        [
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the second channel (green color) of the filter
                         [[0.,0.,0.],
                          [1.,1.,1.],
                          [0.,0.,0.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'Object 2' detector ------
                        [
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the first channel of the filter
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the second channel of the filter
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ],
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-11.,  # the bias for the first filter
                   -11.], # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer (3 actions)
    #---------------------------
    W2 = th.tensor([[ 1., 0.,0.],
                    [-1., 0.,0.]], requires_grad=True)
    b2= th.tensor([0.,1.,2.],requires_grad=True) 
    q  = predict_q(s,W1,b1,W2,b2) 
    # check value 
    assert type(q) == th.Tensor
    assert np.allclose(q.size(),(3,))
    q_true  = [ 1.,1.,2.]
    assert np.allclose(q.data, q_true,atol=0.1)
#---------------------------------------------------
def test_sample_action():
    ''' (8 points) sample_action'''
    s = th.tensor( [ 
                        # the first/red channel of the image 
                        [[1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,1.,0.,0.],
                         [1.,0.,1.,0.],
                         [0.,1.,0.,0.],
                         [0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.],
                         [1.,1.,1.,0.],
                         [0.,0.,0.,0.],
                         [0.,0.,0.,0.]], 
                    ])
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 input channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'object 1' detector ------
                        [
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the second channel (green color) of the filter
                         [[0.,0.,0.],
                          [1.,1.,1.],
                          [0.,0.,0.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'Object 2' detector ------
                        [
                         [[0.,1.,0.],
                          [1.,0.,1.],
                          [0.,1.,0.]], # the first channel of the filter
                         [[1.,0.,1.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the second channel of the filter
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ],
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-11.,  # the bias for the first filter
                   -11.], # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer (3 actions)
    #---------------------------
    W2 = th.tensor([[ 1., 0.,0.],
                    [-1., 0.,0.]], requires_grad=True)
    b2= th.tensor([0.,1.,2.],requires_grad=True) 
    count = np.zeros(3)
    N =400
    for _ in range(N):
        a= sample_action(s,W1,b1,W2,b2,e=0.75)
        count[a]+=1
    assert np.allclose(count/N,[.25,.25,.5], atol = 0.1)
    #-----------------------
    class ReplayMemory:
        def __init__(self, N = 1000):
            self.N = N # capacity
            self.S = [] 
            self.A = [] 
            self.S_new = [] 
            self.R = [] 
            self.T = [] 
            self.i = 0 # position
        # add a sample of game step into the Relay Memory
        def add(self, s, a, s_new, r,done): 
            if len(self.S)< self.N:
                self.S.append(s)
                self.A.append(a)
                self.S_new.append(s_new)
                self.R.append(r)
                self.T.append(done)
            else:
                self.S[self.i]=s
                self.A[self.i]=a
                self.S_new[self.i]=s_new
                self.R[self.i]=r
                self.T[self.i]=done
            self.i = (self.i+1)% self.N
        # sample a mini-batch from replay memory
        def sample(self, n = 100): # batch size
            idx = random.sample(range(len(self.S)),n)
            S = th.tensor([self.S[i] for i in idx])
            A = th.LongTensor([self.A[i] for i in idx])
            S_new = th.tensor([self.S_new[i] for i in idx])
            R = th.tensor([self.R[i] for i in idx])
            T = th.tensor([self.T[i] for i in idx])
            return S, A, S_new, R, T
    #-----------------------
    # define a player class of the deep Q-learning method for using the functions that you have implemented 
    class DQN:
        def __init__(self,
                          e=0.1, # epsilon: explore rate
                          gamma=0.95, # discount factor
                          lr=0.2, # learning rate
                          h= 9, # height of image 
                          w= 9, # width of image 
                          c0= 1, # color/input channels of each game state image
                          c1= 2, # number of filters in the convolutional layer 
                          s= 2, # size of the filters 
                          c= 4, # number of actions 
                          n=200, # batch size
                          n_train=100, # training Q network after every n_train game steps  
                          n_target=500, # updating Target network after every n_target game steps  
                    ):
            self.e = e
            self.gamma = gamma
            self.lr= lr 
            self.n=n
            self.n_train=n_train
            self.n_target=n_target
            self.M = ReplayMemory() # replay memory 
            h1 = (h-s+1)//2
            w1 = (w-s+1)//2
            n_flat_features = c1*h1*w1
            # parameters for Q network
            if c1 ==2 and c0==1 and s==2:
                # To speed up the testing, we initialize weights by cheating a little bit (use good filters)
                self.W1 = th.tensor([[[[1.,1.],
                                       [1.,1.]]],
                                      [[[0.,1.],
                                        [1.,1.]]]],requires_grad=True)
                self.b1 = th.tensor([-3.,-2],requires_grad=True)
            else:
                self.W1 = th.randn(c1,c0,s,s,requires_grad=True)
                self.b1 = th.zeros(c1,requires_grad=True)
            self.W2 = th.zeros(n_flat_features,c,requires_grad=True)
            self.b2 = th.zeros(c,requires_grad=True)
            self.optimizer =  th.optim.SGD([self.W1,self.b1,self.W2,self.b2], lr=lr)
            # parameters for Target Network
            self.W1_ = self.W1.clone()
            self.W2_ = self.W2.clone()
            self.b1_ = self.b1.clone()
            self.b2_ = self.b2.clone()
            # counter for training (controls frequency of training)
            self.i = 0 
        def update_target_network(self):
            self.W1_ = self.W1.clone()
            self.W2_ = self.W2.clone()
            self.b1_ = self.b1.clone()
            self.b2_ = self.b2.clone()
        def sanity_test(self):
            assert self.W1[0,0,0,0]==self.W1[0,0,0,0] # test if the weights are NaN (not a number)
            assert self.W2[0,0]==self.W2[0,0] # test if the weights are NaN (not a number)
            assert self.b1[0]==self.b1[0] # test if the biases are NaN (not a number)
            assert self.b2[0]==self.b2[0] # test if the biases are NaN (not a number)
        def choose_action(self,s):
            return sample_action(th.from_numpy(s),          # choose an action
                                    self.W1,
                                    self.b1,
                                    self.W2,
                                    self.b2,
                                    self.e)
        def update_memory(self, s,a,s_new,r,done):
            # store the data into the mini-batch
            self.M.add(s,a,s_new,r,done)
            self.i+=1 
            if self.i%self.n_train==0:
                S, A, S_new,R,T = self.M.sample(min(self.n,len(self.M.S)))
                # compute target Q value
                Qt =compute_Qt(S_new,R,T,
                                    self.W1_,
                                    self.b1_,
                                    self.W2_,
                                    self.b2_,
                                    self.gamma)
                # update Q network
                update_Q(S, A, Qt,
                               self.W1,
                               self.b1,
                               self.W2,
                               self.b2,
                               self.optimizer)
                self.sanity_test()
            if self.i%self.n_target==0:
                # update target network
                self.update_target_network()
    # create a game 
    g = FrozenLake(image_state=True)
    # create an agent (Q-learning)
    p = DQN(e=1.0) # start with random policy (epsilon = 1.) to explore the map
    # run 1000 games
    r=g.run_games(p,1000)
    sys.stdout.flush()
    assert r > 0 # winning rate of random policy (should win at least one game)
    p.e = 0.1 # now change to epsilon-greedy Q-learning
    r=g.run_games(p,500)
    print(r)
    assert r > 0.1 # winning rate of Q-learning (should win at least 50/500 games)

