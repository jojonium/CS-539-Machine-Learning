from problem3 import *
import numpy as np
import sys
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (60 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.7 or above 
    assert sys.version_info[1]>=7 

#-------------------------------------------------------------------------
def test_linear_kernel():
    ''' (2 points) linear kernel'''
    x1 = np.array([1,2])
    x2 = np.array([3,4])
    k = linear_kernel(x1,x2)
    assert k ==11 

    x1 = np.array([1,2,3])
    x2 = np.array([3,4,5])
    k = linear_kernel(x1,x2)
    assert k ==26 

    x1 = np.array([1,2,3,4,5,6])
    x2 = np.array([3,4,5,6,7,8])
    k = linear_kernel(x1,x2)
    assert k ==133

#-------------------------------------------------------------------------
def test_linear_kernel_matrix():
    ''' (2 points) linear kernel matrix'''

    X1 = np.array([[ 1, 2],
                   [ 2, 4],
                   [ 3, 6]])
    X2 = np.array([[ 1, 1],
                   [-1,-1]])
    K = linear_kernel_matrix(X1,X2)
    assert type(K) == np.ndarray
    assert K.shape == (3,2)
    K_true=[[3,-3],
            [6,-6],
            [9,-9]]
    assert np.allclose(K, K_true, atol = 1e-3) 


    X = np.array([[0.,1.],
                  [1.,0.],
                  [1.,1.]])
    K = linear_kernel_matrix(X,X)
    assert type(K) == np.ndarray
    assert K.shape == (3,3)
    K_true=[[1,0,1],
            [0,1,1],
            [1,1,2]]
    assert np.allclose(K,K_true, atol = 1e-3) 


    X = np.array([[1.,2.],
                  [3.,4.]])
    K = linear_kernel_matrix(X,X)
    assert K.shape == (2,2)
    K_true=[[ 5,11],
            [11,25]]
    assert np.allclose(K, K_true, atol = 1e-3) 


    X1 = np.array([[0.,1.],
                   [0.,1.],
                   [0.,1.]])
    X2 = np.array([[1.,0.],
                   [1.,1.]])
    K = linear_kernel_matrix(X1,X2)
    assert K.shape == (3,2)
    K_true=[[0,1],
            [0,1],
            [0,1]]
    assert np.allclose(K, K_true, atol = 1e-3) 


    X1 = np.array([[ 1, 2, 3],
                   [ 4, 5, 6]])
    X2 = np.array([[ 1, 1, 1],
                   [-1,-1,-1]])
    K = linear_kernel_matrix(X1,X2)
    assert K.shape == (2,2)
    K_true=[[ 6, -6],
            [15,-15]]
    assert np.allclose(K, K_true, atol = 1e-3) 




#-------------------------------------------------------------------------
def test_polynomial_kernel():
    ''' (2 points) polynomial kernel'''
    x1 = np.array([1,2])
    x2 = np.array([3,4])
    k = polynomial_kernel(x1,x2,d=2)
    assert k ==144 

    x1 = np.array([1,2])
    x2 = np.array([3,4])
    k = polynomial_kernel(x1,x2,d=3)
    assert k ==1728 

    x1 = np.array([1,2,3])
    x2 = np.array([3,4,5])
    k = polynomial_kernel(x1,x2,d=2)
    assert k ==729

    x1 = np.array([1,2,3,4,5,6])
    x2 = np.array([3,4,5,6,7,8])
    k = polynomial_kernel(x1,x2,d=4)
    assert k ==322417936

#-------------------------------------------------------------------------
def test_polynomial_kernel_matrix():
    ''' (2 points) polynomial kernel matrix'''

    X1 = np.array([[ 1., 2.],
                   [ 2., 4.],
                   [ 3., 6.]])
    X2 = np.array([[ 1., 1.],
                   [-1.,-1.]])
    K = polynomial_kernel_matrix(X1,X2,d=2)
    assert type(K) == np.ndarray
    assert K.shape == (3,2)
    K_true=[[  16,  4 ],
            [  49, 25 ],
            [ 100, 64 ]]
    assert np.allclose(K, K_true, atol = 1e-3) 


    X = np.array([[0.,1.],
                  [1.,0.],
                  [1.,1.]])
    K = polynomial_kernel_matrix(X,X,d=1)
    assert type(K) == np.ndarray
    assert K.shape == (3,3)
    K_true=[[2,1,2],
            [1,2,2],
            [2,2,3]]
    assert np.allclose(K, K_true, atol = 1e-3) 


    X = np.array([[1.,2.],
                  [3.,4.]])
    K = polynomial_kernel_matrix(X,X,d=1)
    assert K.shape == (2,2)
    K_true=[[ 6,12],
            [12,26]]
    assert np.allclose(K, K_true, atol = 1e-3) 

    X = np.array([[0.,1.],
                  [1.,0.],
                  [1.,1.]])
    K = polynomial_kernel_matrix(X,X,d=2)
    assert K.shape == (3,3)
    K_true=[[4,1,4],
            [1,4,4],
            [4,4,9]]
    assert np.allclose(K, K_true, atol = 1e-3) 
    K = polynomial_kernel_matrix(X,X,d=3)
    K_true=[[8,1,8],
            [1,8,8],
            [8,8,27]]
    assert np.allclose(K, K_true, atol = 1e-3) 


    X1 = np.array([[0.,1.],
                   [0.,1.],
                   [0.,1.]])
    X2 = np.array([[1.,0.],
                   [1.,1.]])
    K = polynomial_kernel_matrix(X1,X2,d=1)
    assert K.shape == (3,2)
    K_true=[[1,2],
            [1,2],
            [1,2]]
    assert np.allclose(K, K_true, atol = 1e-3) 

#-------------------------------------------------------------------------
def test_gaussian_kernel():
    ''' (2 points) Gaussian kernel'''
    x1 = np.array([1,2])
    x2 = np.array([3,4])
    k = gaussian_kernel(x1,x2,sigma=5)
    assert np.allclose(k,0.85214378896, atol= 1e-3)

    x1 = np.array([1,2])
    x2 = np.array([3,4])
    k = gaussian_kernel(x1,x2,sigma=1)
    assert np.allclose(k,0.018315638888, atol= 1e-3)

    x1 = np.array([1,2,3])
    x2 = np.array([3,4,5])
    k = gaussian_kernel(x1,x2)
    assert np.allclose(k,0.0024787521766, atol= 1e-3)

    x1 = np.array([1,2,3,4,5,6])
    x2 = np.array([3,4,5,6,7,8])
    k = gaussian_kernel(x1,x2,sigma=10)
    assert np.allclose(k,0.88692043671, atol= 1e-3)

#-------------------------------------------------------------------------
def test_gaussian_kernel_matrix():
    ''' (1 point) Gaussian kernel matrix'''

    X1 = np.array([[ 1., 0.],
                   [ 0., 1.],
                   [ 1., 1.]])
    X2 = np.array([[ 1., 1.],
                   [-1.,-1.]])
    K = gaussian_kernel_matrix(X1,X2)
    assert type(K) == np.ndarray
    assert K.shape == (3,2)
    K_true=[[0.60653066,0.082085  ],
            [0.60653066,0.082085  ],
            [1.        ,0.01831564]]
    assert np.allclose(K, K_true, atol = 1e-3) 

    X = np.array([[1.,1.],
                  [1.,1.]])
    K = gaussian_kernel_matrix(X,X,1)
    assert K.shape == (2,2)
    assert np.allclose(K, X, atol = 1e-3) 

    X = np.array([[0.,1.],
                  [1.,0.]])
    K = gaussian_kernel_matrix(X,X,1.)
    assert type(K) == np.ndarray
    assert K.shape == (2,2)
    assert np.allclose(K, [[1,.367879],[.367879,1]], atol = 1e-3) 

    X = np.array([[0.,100.],
                  [100.,0.]])
    K = gaussian_kernel_matrix(X,X,1.)
    assert K.shape == (2,2)
    assert np.allclose(K, [[1,0],[0,1]], atol = 1e-3) 

    X = np.array([[0.,1.],
                  [1.,0.]])
    K = gaussian_kernel_matrix(X,X,0.1)
    assert K.shape == (2,2)
    assert np.allclose(K, [[1,0],[0,1]], atol = 1e-3) 


    X = np.array([[1.,1.],
                  [1.,1.],
                  [1.,1.]])
    K = gaussian_kernel_matrix(X,X,1.)
    assert K.shape == (3,3)
    assert np.allclose(K, np.ones((3,3)), atol = 1e-3) 

    X1 = np.array([[0.,1.],
                   [0.,1.],
                   [0.,1.]])
    X2 = np.array([[0.,1.],
                   [1.,1.]])
    K = gaussian_kernel_matrix(X1,X2,0.1)
    assert K.shape == (3,2)
    assert np.allclose(K, [[1,0],[1,0],[1,0]], atol = 1e-3) 


#-------------------------------------------------------------------------
def test_compute_fx():
    ''' (2 points) compute_fx'''

    K = np.array([1,2,3])
    a = np.array([0,1,1])
    y = np.array([1,-1,1])
    b = 4
    f = compute_fx(K,a,y,b)
    assert f==5 

    K = np.array([1,2,3,4])
    a = np.array([1,0,0,1])
    y = np.array([1,-1,1,-1])
    b = 0
    f = compute_fx(K,a,y,b)
    assert f==-3 



#-------------------------------------------------------------------------
def test_predict():
    ''' (3 points) predict'''

    K = np.array([[ 1, 2, 4],
                  [ 6, 3, 5]])
    a = np.array([1,-1,1])
    y = np.array([1,-1,1])
    b = 0. 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.ndarray
    assert y_test.shape == (2,)
    assert np.allclose(y_test, [1,1], atol = 1e-3) 


    K = np.array([[1.,1.],
                  [1.,1.]])
    a = np.array([1.,1.])
    y = np.array([1.,1.])
    b = 0. 
    y_test = predict(K,a,y,b)
    assert type(y_test) == np.ndarray
    assert y_test.shape == (2,)
    assert np.allclose(y_test, [1,1], atol = 1e-3) 

    K = np.array([[1.,0.],
                  [0.,1.]])
    a = np.array([1.,1.])
    y = np.array([1.,-1.])
    b = 0. 
    y_test = predict(K,a,y,b)
    assert y_test.shape == (2,)
    assert np.allclose(y_test, [1,-1], atol = 1e-3) 

    K = np.array([[1.,0.],
                  [1.,1.]])
    a = np.array([1.,1.])
    y = np.array([1.,-1.])
    b = .1 
    y_test = predict(K,a,y,b)
    assert y_test.shape == (2,)
    assert np.allclose(y_test, [1,1], atol = 1e-3) 

    K = np.array([[1.,0.],
                  [1.,1.]])
    a = np.array([1.,2.])
    y = np.array([1.,-1.])
    b = .1 
    y_test = predict(K,a,y,b)
    assert y_test.shape == (2,)
    assert np.allclose(y_test, [1,-1], atol = 1e-3) 


    K = np.array([[1.,1.],
                  [2.,3.],
                  [1.,1.]])
    a = np.array([1.,1.])
    y = np.array([1.,1.])
    b = 0. 
    y_test = predict(K,a,y,b)
    assert y_test.shape == (3,)
    assert np.allclose(y_test, [1,1,1], atol = 1e-3) 


#-------------------------------------------------------------------------
def test_compute_HL():
    ''' (5 points) compute_HL'''
    ai = 0. 
    yi = 1.
    aj = 0. 
    yj = 1.
    H,L = compute_HL(ai,yi,aj,yj,C=1.) 
    assert H == 0.
    assert L == 0. 

    H,L = compute_HL(0.,1.,0.,-1.,C=1.) 
    assert H == 1.
    assert L == 0. 

    H,L = compute_HL(0.,-1.,0.,1.,C=1.) 
    assert H == 1.
    assert L == 0. 

    H,L = compute_HL(0.,-1.,0.,1.,C=10.) 
    assert H == 10.
    assert L == 0. 

    H,L = compute_HL(0.,-1.,2.,1.,C=10.) 
    assert H == 8.
    assert L == 0. 

    H,L = compute_HL(3.,-1.,2.,1.,C=8.) 
    assert H == 8.
    assert L == 1. 

    H,L = compute_HL(3.,-1.,2.,-1.,C=8.) 
    assert H == 5.
    assert L == 0. 

#-------------------------------------------------------------------------
def test_compute_E():
    ''' (5 points) compute_E'''

    Ki = np.array([1.,1.])
    a = np.array([1.,1.])
    y = np.array([1.,1.])
    b = 0. 
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 1.

    Ki = np.array([1.,.5])
    E = compute_E(Ki,a,y,b,i=0)
    assert E == .5

    a = np.array([1.,2.])
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 1.

    b = 1.
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 2.

    y = np.array([1.,-1.])
    E = compute_E(Ki,a,y,b,i=0)
    assert E == 0. 
    


#-------------------------------------------------------------------------
def test_compute_eta():
    ''' (5 points) compute_eta'''
    e = compute_eta(1.,1.,1.5)
    assert e == 1.

    e = compute_eta(1.,1.,2.)
    assert e == 2.

    e = compute_eta(.5,1.,2.)
    assert e == 2.5

    e = compute_eta(.5,.7,2.)
    assert e == 2.8

#-------------------------------------------------------------------------
def test_update_ai():
    ''' (5 points) update_ai'''
    an = update_ai(1.,1.,1.,4.,1.,10,0.)
    assert an == 4.

    an = update_ai(1.,2.,1.,4.,1.,10,0.)
    assert an == 3.

    an = update_ai(0.,2.,1.,4.,1.,10,0.)
    assert an == 2.

    an = update_ai(0.,2.,2.,4.,1.,10,0.)
    assert an == 3.

    an = update_ai(0.,2.,1.,4.,1.,10.,3.)
    assert an == 3.

    an = update_ai(0.,2.,1.,4.,-1.,10.,3.)
    assert an == 6.

    an = update_ai(0.,2.,1.,4.,-1.,5.,3.)
    assert an == 5.

    an = update_ai(0.,2.,0.,4.,-1.,5.,3.)
    assert an == 4. # if eta =0, ai is not changed. (when i= j)


#-------------------------------------------------------------------------
def test_update_aj():
    ''' (5 points) update_aj'''
    an = update_aj(1.,1.,2.,1.,1.)
    assert an == 0.

    an = update_aj(1.,1.,2.,1.,-1.)
    assert an == 2.

    an = update_aj(1.,1.,2.,-1.,1.)
    assert an == 2.

    an = update_aj(1.,0.,2.,-1.,1.)
    assert an == 3.

    an = update_aj(1.,0.,3.,-1.,1.)
    assert an == 4.

    an = update_aj(2.,0.,3.,-1.,1.)
    assert an == 5.


#-------------------------------------------------------------------------
def test_update_b():
    ''' (5 point) update_b'''
    b = update_b(1.,2.,3.,1.,1.,1.,1.,1.,1.,1.,1.,1.,3.)
    assert b == -3.

    b = update_b(1.,3.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,3.)
    assert b == -3.

    b = update_b(1.,3.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,4.)
    assert b == -3.

    b = update_b(2.,3.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,4.)
    assert b == -2.

    b = update_b(2.,4.,2.,1.,1.,1.,1.,1.,1.,1.,1.,1.,5.)
    assert b == -3.

    b = update_b(0.,3.,0.,1.,0.,-1.,1.,0.,1.,1.,1.,1.,5.)
    assert b == 2.

    b = update_b(0.,3.,0.,2.,0.,-1.,1.,0.,1.,1.,1.,1.,5.)
    assert b == 1.

    b = update_b(0.,3.,0.,2.,0.,1.,1.,0.,1.,1.,1.,1.,5.)
    assert b == -1.

    b = update_b(0.,3.,0.,2.,0.,1.,1.,0.,1.,2.,1.,1.,5.)
    assert b == -2.

    b = update_b(0.,3.,0.,2.,0.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -3.

    b = update_b(1.,3.,0.,2.,0.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -2.

    b = update_b(1.,3.,1.,2.,0.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -3.

    b = update_b(1.,3.,2.,2.,1.,1.,1.,1.,1.,2.,1.,1.,5.)
    assert b == -3.

    b = update_b(1.,3.,2.,2.,1.,1.,-1.,1.,1.,2.,1.,1.,5.)
    assert b == -1.

    b = update_b(1.,3.,2.,2.,1.,1.,-1.,1.,1.,2.,1.,2.,5.)
    assert b == 0.

    b = update_b(0.,0.,2.,0.,1.,1.,1.,1.,0.,1.,1.,0.,5.)
    assert b == -1.

    b = update_b(0.,0.,3.,0.,1.,1.,1.,1.,0.,1.,1.,0.,5.)
    assert b == -2.

    b = update_b(0.,0.,3.,0.,1.,1.,1.,1.,1.,1.,1.,0.,5.)
    assert b == -3.

    b = update_b(0.,0.,3.,0.,1.,1.,-1.,1.,1.,1.,1.,0.,5.)
    assert b == 1.

    b = update_b(0.,5.,3.,0.,1.,1.,-1.,1.,1.,1.,1.,0.,5.)
    assert b == 1.

    b = update_b(0.,5.,3.,4.,1.,1.,-1.,1.,1.,1.,1.,1.,5.)
    assert b == 0.

    b = update_b(0.,5.,3.,4.,1.,1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == -1.

    b = update_b(0.,5.,3.,3.,1.,1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == -3.

    b = update_b(0.,5.,3.,3.,1.,-1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == 5.

    b = update_b(1.,5.,3.,3.,1.,-1.,-1.,1.,1.,1.,1.,2.,5.)
    assert b == 6.

    b = update_b(0.,5.,5.,4.,4.,1.,1.,1.,1.,0.,0.,0.,5.)
    assert b == -1.

    b = update_b(0.,5.,5.,4.,4.,1.,1.,2.,2.,0.,0.,0.,5.)
    assert b == -2.

    b = update_b(0.,5.,5.,4.,4.,1.,1.,1.,2.,0.,0.,0.,5.)
    assert b == -1.5

    b = update_b(0.,5.,5.,4.,4.,1.,1.,0.,0.,0.,0.,1.,5.)
    assert b == -1.

    b = update_b(0.,5.,5.,3.,4.,1.,1.,0.,0.,0.,0.,1.,5.)
    assert b == -1.5


#-------------------------------------------------------------------------
def test_train():
    ''' (5 points) train'''
    # linear kernel x1 = 0, x2 = 1
    K = np.array([[0.,0.],
                  [0.,1.]])
    y = np.array([-1.,1.])
    C = 1000.
    a,b = train(K,y,C,10)
    assert type(a) == np.ndarray
    assert a.shape == (2,)
    assert np.allclose(a, [2,2],atol = 1e-3)
    assert np.allclose(b , -1, atol=1e-3) 

    a,b = train(K,y,C,2)
    assert np.allclose(a, [2,2],atol = 1e-3)
    assert np.allclose(b , -1, atol=1e-3) 

    # linear kernel x1 = 0, x2 = 2
    K = np.array([[0.,0.],
                  [0.,4.]])
    a,b = train(K,y,C)
    assert np.allclose(a, [.5,.5],atol = 1e-3)
    assert np.allclose(b , -1, atol=1e-3) 

    # linear kernel x1 = -1, x2 = 1
    K = np.array([[1.,-1],
                  [-1.,1.]])
    a,b = train(K,y,C)
    assert np.allclose(a, [.5,.5],atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 


    # linear kernel x1 = -1, x2 = 1, x3 = 2
    K = np.array([[ 1.,-1.,-2.],
                  [-1., 1., 2.],
                  [-2., 2., 4.]])
    y = np.array([-1.,1.,1.])
    a,b = train(K,y,C)
    assert np.allclose(a, [.5,.5,0.],atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = -1, x2 = 1, x3 = 3
    K = np.array([[ 1.,-1.,-3.],
                  [-1., 1., 3.],
                  [-3., 3., 9.]])
    y = np.array([-1.,1.,1.])
    a,b = train(K,y,C)
    assert np.allclose(a, [.5,.5,0.],atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = -1, x2 = 1, x3 = 1.1 
    K = np.array([[ 1.,-1.,-1.1],
                  [-1., 1., 1.1],
                  [-1.1, 1.1, 1.21]])
    y = np.array([-1.,1.,1.])
    a,b = train(K,y,C)
    assert np.allclose(a, [.5,.5,0.],atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = -2, x2 = -1, x3 = 1, x4 = 2
    K = np.array([[ 4., 2.,-2.,-4.],
                  [ 2., 1.,-1.,-2.],
                  [-2.,-1., 1., 2.],
                  [-4.,-2., 2., 4.]])
    y = np.array([-1.,-1,1.,1.])
    a,b = train(K,y,C)
    assert np.allclose(a, [0.,.5,.5,0.],atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 


    # linear kernel x1 = -2, x2 = -1, x3 = 1, x4 = 2
    K = np.array([[ 4., 2.,-2.,-4.],
                  [ 2., 1.,-1.,-2.],
                  [-2.,-1., 1., 2.],
                  [-4.,-2., 2., 4.]])
    y = np.array([-1.,-1,1.,1.])
    a,b = train(K,y,0.2)
    assert np.allclose(a, [0.025,.2,.2,0.025],atol = 1e-3)
    assert np.allclose(b , 0, atol=1e-3) 

    # linear kernel x1 = (-1.1,0), x2 = (0,-1), x3 = (1,0), x4 = (0,1.1)
    K = np.array([[1.21, 0. ,-1.1, 0.  ],
                  [  0., 1. , 0. ,-1.1 ],
                  [-1.1, 0. , 1. , 0.  ],
                  [  0.,-1.1, 0. , 1.21]])
    y = np.array([-1.,-1,1.,1.])
    a,b = train(K,y,10)
    assert np.allclose(a, [0,1,1,0],atol =.01)
    assert np.allclose(b , 0, atol=1e-2) 


#-------------------------------------------------------------------------
def test_svm_linear():
    '''(3 point) SVM (linear kernel)'''
    # load a binary classification dataset
    n_samples = 200
    X=np.loadtxt('X.csv',dtype=float, delimiter=',')
    y=np.loadtxt('y.csv',dtype=int, delimiter=',')
    # split the dataset into a training set and a test set
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    # compute kernels
    K1 = linear_kernel_matrix(Xtrain,Xtrain)
    K2 = linear_kernel_matrix(Xtest,Xtrain)
    # train the model
    a,b = train(K1, Ytrain, C=1., n_epoch=1)
    n_SV = (a>0).sum() # number of support vectors
    assert n_SV < 25 
    Y = predict(K1, a, Ytrain,b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.85
    Y = predict(K2, a, Ytrain,b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.85

#-------------------------------------------------------------------------
def test_svm_poly():
    '''(3 point) SVM (polynomial kernel)'''
    # load a binary classification dataset
    n_samples = 200
    X=np.loadtxt('X.csv',dtype=float, delimiter=',')
    y=np.loadtxt('y.csv',dtype=int, delimiter=',')
    # split the dataset into a training set and a test set
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    # compute kernels
    K1 = polynomial_kernel_matrix(Xtrain,Xtrain,2)
    K2 = polynomial_kernel_matrix(Xtest,Xtrain,2)
    a,b = train(K1, Ytrain, C=1., n_epoch=2)
    n_SV = (a>0).sum() # number of support vectors
    assert n_SV < 50 
    Y = predict(K1, a, Ytrain,b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.9
    Y = predict(K2, a, Ytrain,b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.87


#-------------------------------------------------------------------------
def test_svm_RBF():
    '''(3 point) SVM (Gaussian kernel)'''
    # load a binary classification dataset
    n_samples = 200
    X=np.loadtxt('X.csv',dtype=float, delimiter=',')
    y=np.loadtxt('y.csv',dtype=int, delimiter=',')
    # split the dataset into a training set and a test set
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    # compute kernels
    K1 = gaussian_kernel_matrix(Xtrain,Xtrain)
    K2 = gaussian_kernel_matrix(Xtest,Xtrain)
    # train the model
    a,b = train(K1, Ytrain, C=1., n_epoch=1)
    n_SV = (a>0).sum() # number of support vectors
    assert n_SV < 50 
    Y = predict(K1, a, Ytrain,b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.9
    Y = predict(K2, a, Ytrain,b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.9

