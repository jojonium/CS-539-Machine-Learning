from problem2 import *
import sys
import math
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (32 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_conv2d_a():
    ''' (2 points) conv2d_a'''
    
    # an image of 6 by 8 pixels     
    # h = 6, w = 8 
    x1 = th.tensor([[0.,1.,0.,0.,0.,0.,0.,0.],
                    [1.,1.,1.,0.,0.,0.,0.,0.],
                    [0.,1.,0.,0.,0.,0.,0.,0.],
                    [0.,0.,0.,0.,0.,0.,1.,0.],
                    [0.,0.,0.,0.,0.,1.,1.,1.],
                    [0.,0.,0.,0.,0.,0.,1.,0.]])
    # a filter of shape 3 x 3, trying to match a pattern '+' in the image
    w1 = th.tensor([[0.,1.,0.],
                    [1.,1.,1.],
                    [0.,1.,0.]], requires_grad=True)
    b1 = th.tensor(-4., requires_grad=True)
    z1 = conv2d_a(x1,w1,b1)
    assert type(z1) == Tensor 
    assert np.allclose(z1.size(),(4,6))
    z1_true = [[ 1.,-2.,-3.,-4.,-4.,-4.],
               [-2.,-2.,-4.,-4.,-4.,-3.],
               [-3.,-4.,-4.,-4.,-2.,-2.],
               [-4.,-4.,-4.,-3.,-2., 1.]]
    assert np.allclose(z1.data,z1_true, atol=0.1)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in z)
    L = z1.sum()
    # back propagation
    L.backward()
    # check the gradients of w1, b1
    dL_dw1_true = [[5., 5., 2.],
                   [5., 6., 5.],
                   [2., 5., 5.]]
    assert np.allclose(w1.grad, dL_dw1_true, atol= 0.1)
    assert np.allclose(b1.grad, 24, atol= 0.1)
    # test another example:
    # an image of 4 by 4 pixels     
    x1 = th.tensor([[1.,2.,3.,0.],
                    [1.,2.,3.,0.],
                    [2.,3.,4.,0.],
                    [0.,0.,0.,0.]])
    # a filter of shape 3 x 3 
    w1 = th.tensor([[1.,2.,3.],
                    [2.,3.,4.],
                    [3.,1.,5.]], requires_grad=True)
    b1 = th.tensor(-5., requires_grad=True)
    z1 = conv2d_a(x1,w1,b1)
    assert type(z1) == Tensor 
    assert np.allclose(z1.size(),(2,2))
    z1_true = [[58., 29.],
               [38., 21.]]
    assert np.allclose(z1.data,z1_true, atol=0.1)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in z)
    L = z1.sum()
    # back propagation
    L.backward()
    # check the gradients of w1, b1
    dL_dw1_true = [[ 6., 10.,  6.],
                   [ 8., 12.,  7.],
                   [ 5.,  7.,  4.]]
    assert np.allclose(w1.grad, dL_dw1_true, atol= 0.1)
    assert np.allclose(b1.grad, 4, atol= 0.1)
    # test the function with random input sizes
    s = np.random.randint(2,5) # size of the filter 
    h = s+np.random.randint(5,20) # hight of the image 
    w = s+np.random.randint(5,20) # width of the image 
    x1  = th.randn(h,w)
    W1  = th.randn(s,s)
    b1 = th.randn(1)
    z1 = conv2d_a(x1,W1,b1) 
    assert np.allclose(z1.size(),(h-s+1,w-s+1))
#---------------------------------------------------
def test_conv2d_b():
    ''' (2 points) conv2d_b'''
    
    # an image of 6 by 8 pixels with 3 color/input channels (shape: 3 channel x 6 height x 8 width)
    # l = 3, h = 6, w = 8 
    x2 = th.tensor([
                    [[1.,0.,1.,0.,0.,0.,0.,0.],
                     [1.,0.,1.,0.,0.,0.,0.,0.],
                     [1.,0.,1.,0.,0.,0.,0.,0.],
                     [0.,0.,0.,0.,0.,1.,0.,1.],
                     [0.,0.,0.,0.,0.,1.,0.,1.],
                     [0.,0.,0.,0.,0.,1.,0.,1.]], # the first/red channel of the image 
    
                    [[0.,1.,0.,0.,0.,0.,0.,0.],
                     [1.,1.,1.,0.,0.,0.,0.,0.],
                     [0.,1.,0.,0.,0.,0.,0.,0.],
                     [0.,0.,0.,0.,0.,0.,1.,0.],
                     [0.,0.,0.,0.,0.,1.,1.,1.],
                     [0.,0.,0.,0.,0.,0.,1.,0.]],# the second/green channel of the image
    
                    [[1.,0.,1.,0.,0.,0.,0.,0.],
                     [0.,1.,0.,0.,0.,0.,0.,0.],
                     [1.,0.,1.,0.,0.,0.,0.,0.],
                     [0.,0.,0.,0.,0.,1.,0.,1.],
                     [0.,0.,0.,0.,0.,0.,1.,0.],
                     [0.,0.,0.,0.,0.,1.,0.,1.]] # the third/blue channel of the image
                    ])
    # one filter of shape 3 x 3 with 3 channels   (shape: 3 channels x  3 height x 3 width )
    w2 = th.tensor([
                    [[1.,0.,1.],
                     [1.,0.,1.],
                     [1.,0.,1.]], # the first channel of the filter, trying to match a red-colored pattern '| |' 
    
                    [[0.,1.,0.],
                     [1.,1.,1.],
                     [0.,1.,0.]], # the second channel of the filter, trying to match a green-colored pattern '+'
    
                    [[1.,0.,1.],
                     [0.,1.,0.],
                     [1.,0.,1.]]  # the third channel of the filter, trying to match a blue-colored pattern 'X'
                    ], requires_grad= True)
    b2 = th.tensor(-15., requires_grad = True)
    z2 = conv2d_b(x2,w2,b2)
    assert type(z2) == Tensor 
    assert np.allclose(z2.size(),(4,6))
    z2_true = [[  1., -13.,  -9., -15., -15., -15.],
               [ -9., -11., -13., -13., -15., -10.],
               [-10., -15., -13., -13., -11.,  -9.],
               [-15., -15., -15.,  -9., -13.,   1.]]
    assert np.allclose(z2.data,z2_true, atol=0.1)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in z)
    L = z2.sum()
    # back propagation
    L.backward()
    # check the gradients of w, b
    dL_dw2_true = [[[7., 4., 5.],
                    [6., 4., 6.],
                    [5., 4., 7.]],
    
                   [[5., 5., 2.],
                    [5., 6., 5.],
                    [2., 5., 5.]],
    
                   [[6., 4., 4.],
                    [4., 4., 4.],
                    [4., 4., 6.]]]
    assert np.allclose(w2.grad, dL_dw2_true, atol= 0.1)
    assert np.allclose(b2.grad, 24, atol= 0.1)
    # test the function with random input sizes
    s = np.random.randint(3,5) # size of the filter 
    h = s+np.random.randint(5,20) # hight of the image 
    w = s+np.random.randint(5,20) # width of the image 
    l = np.random.randint(2,10) # number of channels 
    x2 = th.randn(l,h,w)
    W2 = th.randn(l,s,s)
    b2= th.randn(1)
    z2= conv2d_b(x2,W2,b2) 
    assert np.allclose(z2.size(),(h-s+1,w-s+1))
#---------------------------------------------------
def test_conv2d_c():
    ''' (2 points) conv2d_c'''
    
    # an image of 6 by 8 pixels with 3 color/input channels  (shape: 3 channel x 6 height x 8 width )
    # l = 3, h = 6, w = 8 
    x3 = th.tensor([
                    [[1.,0.,1.,0.,0.,0.,1.,0.],
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
                     [0.,1.,0.,0.,0.,1.,0.,1.]] # the third/blue channel of the image
                    ])
    # 2 filters of shape 3 x 3 with 3 channels    (shape: 2 filters x 3 channels x 3 hight x 3 width)
    w3 = th.tensor([
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
    b3 = th.tensor([-15.,   # the bias for the first filter
                    -13.],  # the bias for the second filter
                    requires_grad=True)
    z3 = conv2d_c(x3,w3,b3)
    assert type(z3) == Tensor 
    assert np.allclose(z3.size(),(2,4,6))
    z3_true = [[[  1., -13.,  -9., -13., -10.,  -8.],  # outputs of the first filter
                [ -6.,  -9., -12., -11., -11.,  -4.],
                [ -5., -10., -11., -13.,  -8.,  -8.],
                [ -8., -10., -13.,  -9., -13.,   1.]],
    
               [[ -6.,  -8., -11.,  -9.,  -9.,   1.], # outputs of the second filter
                [ -4.,  -5., -10., -13.,  -7., -11.],
                [ -9.,  -8., -12., -10.,  -6.,  -3.],
                [  1.,  -9.,  -9., -11.,  -8.,  -6.]]]
    assert np.allclose(z3.data,z3_true, atol=0.1)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in z)
    L = z3.sum()
    # back propagation
    L.backward()
    # check the gradients of w, b
    dL_dw3_true = [[[[ 9.,  8.,  9.],
                     [10.,  8., 10.],
                     [ 9.,  8.,  9.]],
    
                    [[10., 11.,  9.],
                     [ 9., 10.,  9.],
                     [ 9., 11., 10.]],
    
                    [[ 9.,  8.,  9.],
                     [ 7.,  8.,  7.],
                     [ 8.,  8.,  8.]]],
    
    
                   [[[ 9.,  8.,  9.],
                     [10.,  8., 10.],
                     [ 9.,  8.,  9.]],
    
                    [[10., 11.,  9.],
                     [ 9., 10.,  9.],
                     [ 9., 11., 10.]],
    
                    [[ 9.,  8.,  9.],
                     [ 7.,  8.,  7.],
                     [ 8.,  8.,  8.]]]]
    assert np.allclose(w3.grad, dL_dw3_true, atol= 0.1)
    assert np.allclose(b3.grad, [24,24], atol= 0.1)
    # test the function with random input sizes
    s = np.random.randint(2,5) # size of the filter 
    h = s+np.random.randint(5,20) # hight of the image 
    w = s+np.random.randint(5,20) # width of the image 
    l = np.random.randint(2,10) # number of channels 
    n_filters = np.random.randint(2,10) # number of filters 
    x3  = th.randn(l,h,w)
    W3  = th.randn(n_filters, l,s,s)
    b3 = th.randn(n_filters)
    z3 = conv2d_c(x3,W3,b3) 
    assert np.allclose(z3.size(),(n_filters, h-s+1,w-s+1))
#---------------------------------------------------
def test_compute_z1():
    ''' (2 points) compute_z1'''
    # 2 images of 6 by 8 pixels with 3 color/input channels  (shape: 3 channel x 6 height x 8 width )
    # n = 2, c = 3, h = 6, w = 8 
    x = th.tensor([
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
    z = compute_z1(x,W,b)
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
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    s = np.random.randint(2,5) # size of the filter 
    h = s+np.random.randint(5,20) # hight of the image 
    w = s+np.random.randint(5,20) # width of the image 
    l = np.random.randint(2,10) # number of channels 
    n_filters = np.random.randint(2,10) # number of filters 
    x  = th.randn(n,l,h,w)
    W  = th.randn(n_filters, l,s,s)
    b = th.randn(n_filters)
    z = compute_z1(x,W,b) 
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
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in a)
    L = a.sum()
    # back propagation
    L.backward()
    # check the gradients dL_dz, which happens to equal to a_true in this test case.
    assert np.allclose(z.grad, a_true, atol= 0.1)
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20) # hight of the image 
    w = np.random.randint(5,20) # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    z = th.randn(n,n_filters,h,w)
    a = compute_a1(z) 
    assert np.allclose(a.size(),(n,n_filters, h,w))
#---------------------------------------------------
def test_compute_p1():
    ''' (2 points) compute_p1'''
    
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
    p = compute_p1(a)
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
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20)*2 # hight of the image 
    w = np.random.randint(5,20)*2 # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    a = th.randn(n,n_filters,h,w)
    p = compute_p1(a)
    assert np.allclose(p.size(),(n,n_filters, h/2,w/2))
    # check gradient with multiple max values
    a = th.tensor([[[[ 0., 1.],
                     [ 1., 0.]]]],requires_grad=True)
    p = compute_p1(a)
    t = p.sum()
    t.backward()
    dL_da_true = [[[[ 0., 1.],
                    [ 0., 0.]]]]
    assert np.allclose(a.grad,dL_da_true,atol=1e-2)
#---------------------------------------------------
def test_compute_z2():
    ''' (2 points) compute_z2'''
    
    # the pooled feature map of 2 images, the size of the feature map is 6 by 8 pixels with 3 input channels  (shape: 3 channel x 6 height x 8 width )
    # n= 2, c1 = 3, h = 6, w = 8 
    p1= th.tensor([
                   #---------- the feature map of the first image in the mini-batch ------
                   [[[0.,0.,1.,0.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,0.,1.,0.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,0.,1.,0.,0.],
                     [0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,0.,1.,0.,0.]], # the first channel of the feature map 
    
                    [[0.,1.,1.,1.,0.,1.,0.,0.],
                     [0.,0.,0.,0.,1.,1.,1.,0.],
                     [0.,1.,1.,1.,0.,1.,0.,0.],
                     [0.,0.,1.,0.,1.,1.,1.,0.],
                     [0.,1.,1.,1.,0.,0.,0.,0.],
                     [0.,0.,1.,0.,1.,1.,1.,0.]],# the second channel of the feature map 
    
                    [[0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,0.,1.,0.,0.,1.,0.,0.],
                     [0.,0.,1.,0.,1.,0.,1.,0.],
                     [0.,1.,0.,1.,1.,0.,1.,0.],
                     [0.,0.,1.,0.,0.,1.,0.,0.],
                     [0.,1.,0.,1.,0.,1.,0.,0.]]], # the third channel of the feature map 
    
                   #---------- the feature map of the second image in the mini-batch ------
                   [[[1.,0.,1.,0.,0.,0.,1.,0.],
                     [1.,0.,1.,0.,0.,1.,0.,1.],
                     [1.,0.,1.,0.,0.,0.,1.,0.],
                     [0.,1.,0.,0.,0.,1.,0.,1.],
                     [1.,0.,1.,0.,0.,1.,0.,1.],
                     [0.,1.,0.,0.,0.,1.,0.,1.]], # the first channel of the feature map 
    
                    [[0.,1.,0.,0.,0.,1.,1.,1.],
                     [1.,1.,1.,0.,0.,0.,0.,0.],
                     [0.,1.,0.,0.,0.,1.,1.,1.],
                     [1.,1.,1.,0.,0.,0.,1.,0.],
                     [0.,0.,0.,0.,0.,1.,1.,1.],
                     [1.,1.,1.,0.,0.,0.,1.,0.]],# the second channel of the feature map 
    
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
    z = compute_z2(p1,W,b)
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
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    s = np.random.randint(2,5) # size of the filter 
    h = s+np.random.randint(5,20) # hight of the image 
    w = s+np.random.randint(5,20) # width of the image 
    c1 = np.random.randint(2,10) # number of channels 
    c2= np.random.randint(2,10) # number of filters 
    p1  = th.randn(n,c1,h,w) # pooled feature map of the first convolutional layer
    W  = th.randn(c2,c1,s,s)
    b = th.randn(c2)
    z = compute_z2(p1,W,b) 
    assert np.allclose(z.size(),(n,c2, h-s+1,w-s+1))
#---------------------------------------------------
def test_compute_a2():
    ''' (2 points) compute_a2'''
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
    a = compute_a2(z)
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
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in a)
    L = a.sum()
    # back propagation
    L.backward()
    # check the gradients dL_dz, which happens to equal to a_true in this test case.
    assert np.allclose(z.grad, a_true, atol= 0.1)
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20) # hight of the image 
    w = np.random.randint(5,20) # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    z = th.randn(n,n_filters,h,w)
    a = compute_a2(z) 
    assert np.allclose(a.size(),(n,n_filters, h,w))
#---------------------------------------------------
def test_compute_p2():
    ''' (2 points) compute_p2'''
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
    p = compute_p2(a)
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
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20)*2 # hight of the image 
    w = np.random.randint(5,20)*2 # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    a = th.randn(n,n_filters,h,w)
    p = compute_p2(a)
    assert np.allclose(p.size(),(n,n_filters, h/2,w/2))
    # check gradient with multiple max values
    a = th.tensor([[[[ 0., 1.],
                     [ 1., 0.]]]],requires_grad=True)
    p = compute_p1(a)
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
    # test the function with random input sizes
    n = np.random.randint(2,10) # batch size 
    h = np.random.randint(5,20) # hight of the image 
    w = np.random.randint(5,20) # width of the image 
    n_filters = np.random.randint(2,10) # number of filters 
    p = th.randn(n,n_filters,h,w)
    f = flatten(p)
    assert np.allclose(f.size(),(n,n_filters*h*w))
#---------------------------------------------------
def test_compute_z3():
    ''' (2 points) compute_z3'''
    # batch_size = 4
    # number of input features  = 2
    # input feature to the second layer on one mini-batch: 4 (batch_size) by 2 (p) matrix
    f = th.tensor([[1.,1.], # the first sample in the mini-batch
                   [2.,2.], # the second sample in the mini-batch
                   [3.,3.], # the third sample in the mini-batch
                   [4.,4.]])# the fourth sample in the mini-batch
    # weights of length 2
    W3 = th.tensor([ 0.5, -0.4],requires_grad=True)
    # bias
    b3 = th.tensor(-0.3,requires_grad=True) 
    z3 = compute_z3(f,W3,b3)
    assert type(z3) == th.Tensor 
    assert np.allclose(z3.size(), (4,)) # batch_size
    z_true = [-0.2,-0.1, 0.0, 0.1]
    assert np.allclose(z3.data,z_true, atol = 1e-2)
    assert z3.requires_grad
    # check if the gradients of W is connected to z correctly
    L = th.sum(z3) # compute the sum of all elements in z
    L.backward() # back propagate gradient to W and b
    # now the gradients dL_dW should be
    dL_dW_true = [10,10]
    # here [10,10] of dL_dW is computed as the sum of the gradients in all the four samples.
    # for the 1st sample, the gradient is x = [1,1]
    # for the 2nd sample, the gradient is x = [2,2]
    # for the 3rd sample, the gradient is x = [3,3]
    # for the 4th sample, the gradient is x = [4,4]
    # so the sum of the gradients  will be [10,10]
    assert np.allclose(W3.grad,dL_dW_true, atol=0.1)
    # now the gradients of dL_db should be: 4
    # here dL_db is computed as the sum of the gradients in all the four samples: 1+1+1+1 = 4
    assert np.allclose(b3.grad,4, atol=0.1)
#---------------------------------------------------
def test_forward():
    ''' (2 points) forward'''
    # Let's use a face detector example.
    # The shape of the tensors are as follows:
    # x: 2x3x10x10  (2 images, 3 channels, 10 height, 10 width)
    # Convolutional Layer 1:  3 filters of size 3x3 
    # z1: 2x3x8x8 (2 images, 3 filter channels, 8 height, 8 width)
    # a1: 2x3x8x8 (2 images, 3 filter channels, 8 height, 8 width)
    # p1: 2x3x4x4 (2 images, 3 filter channels, 4 height, 4 width) 
    # Convolutional Layer 2:  2 filters of size 3x3 
    # z2: 2x2x2x2 (2 images, 2 filter channels, 2 height, 2 width) 
    # a2: 2x2x2x2 (2 images, 2 filter channels, 2 height, 2 width) 
    # p1: 2x2x1x1 (2 images, 2 filter channels, 1 height, 1 width)
    # f: 2x2 (2 images, 2 flattened features)
    # z3: 2 (2 images)
    x = th.tensor([
                    #---------- the first image in the mini-batch (face type 1) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
    
                        # the second/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
    
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]] 
                    ],
                    #---------- the second image in the mini-batch (face type 2) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
    
                        # the second/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
    
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]] 
                    ] 
                    #----------------------------------------------------
                ])
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 3 filters of shape 3 x 3 with 3 channels    (shape: 3 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'eye' detector ------
                        [
                         [[0.,0.,0.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
    
                         [[0.,0.,0.],
                          [0.,2.,0.],
                          [2.,0.,2.]], # the second channel (green color) of the filter
    
                         [[0.,0.,0.],
                          [0.,3.,0.],
                          [3.,0.,3.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'mouth' detector ------
                        [
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [1.,1.,1.]], # the first channel of the filter
    
                         [[0.,0.,0.],
                          [2.,0.,2.],
                          [2.,2.,2.]], # the second channel of the filter
    
                         [[0.,0.,0.],
                          [3.,0.,3.],
                          [3.,3.,3.]]  # the third channel of the filter
                         ],
                        #---------- the third filter for 'eyebrow' detector ------
                        [
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]], # the first channel of the filter
    
                         [[2.,2.,2.],
                          [0.,0.,0.],
                          [0.,0.,0.]], # the second channel of the filter
    
                         [[3.,3.,3.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ]
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-17.,  # the bias for the first filter
                   -29.,  # the bias for the second filter
                   -17.], # the bias for the third filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W2= th.tensor( [
                        #---------- the first filter for 'face type 1' detector ------
                        [
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [0.,0.,0.]], # the first channel (eye channel) of the filter 
    
                         [[0.,0.,0.],
                          [0.,0.,0.],
                          [0.,1.,0.]], # the second channel (mouth channel) of the filter
    
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [0.,0.,0.]]  # the third channel (eyebrow channel) of the filter
                         ],
                        #---------- the second filter for 'face type 2' detector ------
                        [
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [0.,0.,0.]], # the first channel (eye channel) of the filter 
    
                         [[0.,0.,0.],
                          [0.,0.,0.],
                          [0.,1.,0.]], # the second channel (mouth channel) of the filter
    
                         [[1.,0.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel (eyebrow channel) of the filter
                         ]
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b2= th.tensor([-4.,  # the bias for the first filter
                   -4.],  # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 3: Fully-connected layer
    #---------------------------
    W3 = th.tensor([1., -1.], requires_grad=True)
    b3= th.tensor(0.,requires_grad=True) 
    z3 = forward(x,W1,b1,W2,b2,W3,b3) 
    # check value 
    assert type(z3) == Tensor 
    assert np.allclose(z3.size(),(2,))
    z3_true  = [1., -1.]
    assert np.allclose(z3.data, z3_true)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in p)
    L = z3.sum()
    # back propagation
    L.backward()
    # check the gradients
    dL_dW1 = [[[[ 0.,  0., -1.],
                [ 2.,  0.,  2.],
                [ 0.,  0.,  0.]],
    
               [[ 0.,  0., -1.],
                [ 2.,  0.,  2.],
                [ 0.,  0.,  0.]],
    
               [[ 0.,  0., -1.],
                [ 2.,  0.,  2.],
                [ 0.,  0.,  0.]]],
    
    
              [[[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]],
    
    
              [[[ 0.,  0.,  0.],
                [ 2.,  0.,  1.],
                [ 0., -2.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 2.,  0.,  1.],
                [ 0., -2.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 2.,  0.,  1.],
                [ 0., -2.,  0.]]]]
    
    
    dL_dW2 = [[[[ 0.,  0.,  0.],
                [ 1.,  0.,  1.],
                [ 0.,  0.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  1.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 1.,  0.,  1.],
                [ 0.,  0.,  0.]]],
    
    
              [[[ 0.,  0.,  0.],
                [-1.,  0., -1.],
                [ 0.,  0.,  0.]],
    
               [[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0., -1.,  0.]],
    
               [[-1.,  0., -1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]]]
    assert np.allclose(W1.grad, dL_dW1, atol= 0.1)
    assert np.allclose(b1.grad, [0,0,0], atol= 0.1)
    assert np.allclose(W2.grad, dL_dW2, atol= 0.1)
    assert np.allclose(b2.grad, [1,-1], atol= 0.1)
    assert np.allclose(W3.grad, [1,1], atol= 0.1)
    assert np.allclose(b3.grad, 2, atol= 0.1)
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    s1 = np.random.randint(1,3)*2+1 # size of the filter 
    s2 = np.random.randint(1,3)*2+1 # size of the filter 
    c  = np.random.randint(2,4) # number of color channels 
    c1 = np.random.randint(2,4) # number of filters 
    c2 = np.random.randint(2,4) # number of filters 
    h2 = np.random.randint(2,4) # hight after second CONV layer 
    w2 = np.random.randint(2,4) 
    h1 = h2*2 + s2 - 1
    w1 = w2*2 + s2 - 1
    h = h1*2 + s1 - 1 # hight of the image
    w = w1*2 + s1 - 1 # width of the image
    n_flat_features = c2*h2*w2
    x  = th.randn(n,c,h,w)
    W1  = th.randn(c1,c,s1,s1)
    b1 = th.randn(c1)
    W2  = th.randn(c2,c1,s2,s2)
    b2 = th.randn(c2)
    W3  = th.randn(n_flat_features)
    b3 = th.zeros(1)
    z3 = forward(x,W1,b1,W2,b2,W3,b3) 
    assert np.allclose(z3.size(),(n,))
#---------------------------------------------------
def test_compute_L():
    ''' (2 points) compute_L'''
    
    # batch_size = 4
    # linear logits in a mini-batch
    z = th.tensor([1.,-1., -1000, 1000.], requires_grad=True) 
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.Tensor([0,1,0,1])
    L = compute_L(z,y)
    assert type(L) == th.Tensor 
    assert L.requires_grad
    assert np.allclose(L.detach().numpy(),0.6566,atol=1e-4) 
    # check if the gradients of z is connected to L correctly
    L.backward() # back propagate gradient to W and b
    dL_dz_true = [ 0.1828, -0.1828,  0.,  0.]
    assert np.allclose(z.grad,dL_dz_true, atol=0.01)
    #-----------------------------------------    
    # batch_size = 2
    # linear logits in a mini-batch
    z = th.tensor([-1000., 1000.], requires_grad=True) 
    y = th.Tensor([1,0])
    L = compute_L(z,y)
    assert L.data >100
    assert L.data < float('inf')
    L.backward() # back propagate gradient to W and b
    assert z.grad[0]<0
    assert z.grad[1]>0
#---------------------------------------------------
def test_update_parameters():
    ''' (2 points) update_parameters'''
    #---------------------------
    # Layer 1: Convolutional layer
    #---------------------------
    # 3 filters of shape 3 x 3 with 3 channels    (shape: 3 filters x 3 input channels x 3 hight x 3 width)
    W1= th.tensor( [
                        #---------- the first filter for 'eye' detector ------
                        [
                         [[0.,0.,0.],
                          [0.,1.,0.],
                          [1.,0.,1.]], # the first channel (red color) of the filter 
                         [[0.,0.,0.],
                          [0.,2.,0.],
                          [2.,0.,2.]], # the second channel (green color) of the filter
                         [[0.,0.,0.],
                          [0.,3.,0.],
                          [3.,0.,3.]]  # the third channel (blue color) of the filter
                         ],
                        #---------- the second filter for 'mouth' detector ------
                        [
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [1.,1.,1.]], # the first channel of the filter
                         [[0.,0.,0.],
                          [2.,0.,2.],
                          [2.,2.,2.]], # the second channel of the filter
                         [[0.,0.,0.],
                          [3.,0.,3.],
                          [3.,3.,3.]]  # the third channel of the filter
                         ],
                        #---------- the third filter for 'eyebrow' detector ------
                        [
                         [[1.,1.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]], # the first channel of the filter
                         [[2.,2.,2.],
                          [0.,0.,0.],
                          [0.,0.,0.]], # the second channel of the filter
                         [[3.,3.,3.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel of the filter
                         ]
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b1= th.tensor([-17.,  # the bias for the first filter
                   -29.,  # the bias for the second filter
                   -17.], # the bias for the third filter
                   requires_grad=True)
    #---------------------------
    # Layer 2: Convolutional layer
    #---------------------------
    # 2 filters of shape 3 x 3 with 3 channels    (shape: 2 filters x 3 input channels x 3 hight x 3 width)
    W2= th.tensor( [
                        #---------- the first filter for 'face type 1' detector ------
                        [
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [0.,0.,0.]], # the first channel (eye channel) of the filter 
                         [[0.,0.,0.],
                          [0.,0.,0.],
                          [0.,1.,0.]], # the second channel (mouth channel) of the filter
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [0.,0.,0.]]  # the third channel (eyebrow channel) of the filter
                         ],
                        #---------- the second filter for 'face type 2' detector ------
                        [
                         [[0.,0.,0.],
                          [1.,0.,1.],
                          [0.,0.,0.]], # the first channel (eye channel) of the filter 
                         [[0.,0.,0.],
                          [0.,0.,0.],
                          [0.,1.,0.]], # the second channel (mouth channel) of the filter
                         [[1.,0.,1.],
                          [0.,0.,0.],
                          [0.,0.,0.]]  # the third channel (eyebrow channel) of the filter
                         ]
                        #---------------------------------------------------------
                    ], requires_grad=True)
    b2= th.tensor([-4.,  # the bias for the first filter
                   -4.],  # the bias for the second filter
                   requires_grad=True)
    #---------------------------
    # Layer 3: Fully-connected layer
    #---------------------------
    W3 = th.tensor([1., -1.], requires_grad=True)
    b3= th.tensor(0.,requires_grad=True) 
    # create a toy loss function: the sum of all elements in W1, b1, W2, b2, W3 and b3
    L = W1.sum()+b1.sum() + W2.sum() + b2.sum() + W3.sum() + b3.sum()
    # back propagation to compute the gradients
    L.backward()
    # now the gradients for both W1, b1, W2, b2, W3 and b3 are all-ones
    # let's try updating the parameters with gradient descent
    # create an optimizer for the parameters with learning rate = 0.1
    optimizer = th.optim.SGD([W1,b1,W2,b2,W3,b3], lr=0.1)
    # now perform gradient descent using SGD
    update_parameters(optimizer)
    # let's check the new values of the parameters 
    W1_new = [[[[-0.1, -0.1, -0.1],
                [-0.1,  0.9, -0.1],
                [ 0.9, -0.1,  0.9]],
               [[-0.1, -0.1, -0.1],
                [-0.1,  1.9, -0.1],
                [ 1.9, -0.1,  1.9]],
               [[-0.1, -0.1, -0.1],
                [-0.1,  2.9, -0.1],
                [ 2.9, -0.1,  2.9]]],
              [[[-0.1, -0.1, -0.1],
                [ 0.9, -0.1,  0.9],
                [ 0.9,  0.9,  0.9]],
               [[-0.1, -0.1, -0.1],
                [ 1.9, -0.1,  1.9],
                [ 1.9,  1.9,  1.9]],
               [[-0.1, -0.1, -0.1],
                [ 2.9, -0.1,  2.9],
                [ 2.9,  2.9,  2.9]]],
              [[[ 0.9,  0.9,  0.9],
                [-0.1, -0.1, -0.1],
                [-0.1, -0.1, -0.1]],
               [[ 1.9,  1.9,  1.9],
                [-0.1, -0.1, -0.1],
                [-0.1, -0.1, -0.1]],
               [[ 2.9,  2.9,  2.9],
                [-0.1, -0.1, -0.1],
                [-0.1, -0.1, -0.1]]]]
    b1_new = [-17.1, -29.1, -17.1]
    W2_new = [[[[-0.1, -0.1, -0.1],
                [ 0.9, -0.1,  0.9],
                [-0.1, -0.1, -0.1]],
               [[-0.1, -0.1, -0.1],
                [-0.1, -0.1, -0.1],
                [-0.1,  0.9, -0.1]],
               [[-0.1, -0.1, -0.1],
                [ 0.9, -0.1,  0.9],
                [-0.1, -0.1, -0.1]]],
              [[[-0.1, -0.1, -0.1],
                [ 0.9, -0.1,  0.9],
                [-0.1, -0.1, -0.1]],
               [[-0.1, -0.1, -0.1],
                [-0.1, -0.1, -0.1],
                [-0.1,  0.9, -0.1]],
               [[ 0.9, -0.1,  0.9],
                [-0.1, -0.1, -0.1],
                [-0.1, -0.1, -0.1]]]]
    b2_new = [-4.1, -4.1]
    W3_new = [ 0.9000, -1.1000]
    assert np.allclose(W1.data,W1_new,atol=1e-2) 
    assert np.allclose(b1.data,b1_new,atol=1e-2) 
    assert np.allclose(W2.data,W2_new,atol=1e-2) 
    assert np.allclose(b2.data,b2_new,atol=1e-2) 
    assert np.allclose(W3.data,W3_new,atol=1e-2) 
    assert np.allclose(b3.data,-0.1,atol=1e-2) 
    assert np.allclose(W1.grad,np.zeros((3,3,3,3)),atol=1e-2) 
    assert np.allclose(b1.grad,np.zeros(3),atol=1e-2) 
    assert np.allclose(W2.grad,np.zeros((2,3,3,3)),atol=1e-2) 
    assert np.allclose(b2.grad,[0,0],atol=1e-2) 
    assert np.allclose(W3.grad,[0,0],atol=1e-2) 
    assert np.allclose(b3.grad,0,atol=1e-2) 
#---------------------------------------------------
def test_train():
    ''' (2 points) train'''
    
    X = th.tensor([
                    #---------- the first image in the mini-batch (face type 1) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]] 
                    ],
                    #---------- the second image in the mini-batch (face type 2) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]] 
                    ] 
                    #----------------------------------------------------
                ])
    Y = [1.,0.]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.tensor(Y)
        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    loader = th.utils.data.DataLoader(d, batch_size = 2)
    n_success = 0
    for _ in range(50):
        # train the model
        W1,b1,W2,b2,W3,b3 = train(loader, 
                                  c=3, 
                                  c1=32, 
                                  c2=64, 
                                  h=10, 
                                  w=10, 
                                  s1=3, 
                                  s2=3, 
                                  n_epoch=10)
        # test the label prediction
        z3= forward(X,W1,b1,W2,b2,W3,b3)
        if z3[0]>z3[1]:
            n_success +=1
    assert n_success > 35
#---------------------------------------------------
def test_predict():
    ''' (2 points) predict'''
    
    X = th.tensor([
                    #---------- the first image in the mini-batch (face type 1) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]] 
                    ],
                    #---------- the second image in the mini-batch (face type 2) ------
                    [ 
                        # the first/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the second/red channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], 
                        # the third/green channel of the image 
                        [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,1.,1.,1.,0.,1.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
                         [0.,0.,1.,0.,0.,0.,1.,0.,0.,0.],
                         [0.,1.,0.,1.,0.,1.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                         [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]] 
                    ] 
                    #----------------------------------------------------
                ])
    Y = [1.,0.]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.tensor(Y)
        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    loader = th.utils.data.DataLoader(d, batch_size = 2)
    n_success = 0
    for _ in range(50):
        # train the model
        W1,b1,W2,b2,W3,b3 = train(loader, 
                                  c=3, 
                                  c1=32, 
                                  c2=64, 
                                  h=10, 
                                  w=10, 
                                  s1=3, 
                                  s2=3, 
                                  n_epoch=10)
        # test the label prediction
        y_predict= predict(X,W1,b1,W2,b2,W3,b3)
        if np.allclose(y_predict,[1,0]):
            n_success +=1
    assert n_success > 39

