from problem1 import *
import numpy as np
import sys
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_terms_and_conditions():
    ''' Read and Agree with Terms and Conditions'''
    assert Terms_and_Conditions() # require reading and agreeing with Terms and Conditions. 


#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (10 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.7 or above 
    assert sys.version_info[1]>=7

#-------------------------------------------------------------------------
def test_least_square():
    ''' (5 points) least square'''
    # a dataset of 3 instances, 2 dimensional features
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    w = least_square(X,y)
    assert type(w) == np.ndarray
    assert w.shape == (2,) 
    assert np.allclose(w, [2.5,1.], atol = 1e-2) 

    for _ in range(20):
        p = np.random.randint(2,8)
        n = np.random.randint(200,400)
        w_true = np.random.random(p)
        X = np.random.random((n,p))*10
        e = np.random.randn(n)*0.01
        y = np.dot(X,w_true) + e
        w = least_square(X,y)
        assert np.allclose(w,w_true, atol = 0.1)

#-------------------------------------------------------------------------
def test_ridge_regression():
    ''' (5 points) ridge regression'''
    # a dataset of 3 instances, 2 dimensional features
    X = np.array([[ 1.,-1.],  # the first instance,
                  [ 1., 0.],  # the second instance
                  [ 1., 1.]])
    y = np.array([1.5,2.5,3.5])
    w = ridge_regression(X,y)
    assert type(w) == np.ndarray
    assert w.shape == (2,) 
    assert np.allclose(w, [2.5,1.], atol = 1e-2) 

    w = ridge_regression(X,y,alpha = 1000)
    assert np.allclose(w, [0.,0.], atol = 1e-2) 

    for _ in range(20):
        p = np.random.randint(2,8)
        n = np.random.randint(200,400)
        w_true = np.random.random(p)
        X = np.random.random((n,p))*10
        e = np.random.randn(n)*0.01
        y = np.dot(X,w_true) + e
        w = ridge_regression(X,y)
        assert np.allclose(w,w_true, atol = 0.1)



