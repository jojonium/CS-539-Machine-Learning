from problem1 import *
import sys
import math

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (20 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_PT():
    ''' (4 points) compute_PT'''
    S = np.array([0,1,0,1])
    s = compute_PT(S)
    assert s==0.5 
    S = np.array([1,1])
    s = compute_PT(S)
    assert s==1. 
    S = np.array([0,1,0,1,1])
    s = compute_PT(S)
    assert s==0.6 
#---------------------------------------------------
def test_count_frequency():
    ''' (4 points) count_frequency'''
    W = np.array([0,1,1,2,2,2,3])
    C = count_frequency(W,c=4)
    assert type(C) == np.ndarray
    assert np.allclose(C,[1,2,3,1],atol=0.1)
    W = np.array([0,1,2,2])
    C = count_frequency(W,c=3)
    assert np.allclose(C,[1,1,2],atol=0.1)
    W = np.array([0,1,1,1,3])
    C = count_frequency(W,c=4)
    assert np.allclose(C,[1,3,0,1],atol=0.1) #-------------------------------------
#---------------------------------------------------
def test_laplace_smoothing():
    ''' (4 points) laplace_smoothing'''
    C = np.array([2,0,1,7])
    PW = laplace_smoothing(C,k=0)
    assert np.allclose(PW,[.2,0.,.1,.7],atol=0.1)
    PW = laplace_smoothing(C,k=1)
    assert np.allclose(PW,[0.21428571,0.07142857,0.14285714,0.57142857],atol=0.01)
    PW = laplace_smoothing(C,k=2)
    assert np.allclose(PW,[0.22222222,0.11111111,0.16666667,0.5       ],atol=0.01)
    PW = laplace_smoothing(C,k=5)
    assert np.allclose(PW,[0.23333333,0.16666667,0.2       ,0.4       ],atol=0.01)
    PW = laplace_smoothing(C,k=.5)
    assert np.allclose(PW,[0.20833333,0.04166667,0.125     ,0.625     ],atol=0.01)
#---------------------------------------------------
def test_compute_PW_T():
    ''' (4 points) compute_PW_T'''
    Ws = np.array([0,0,0,0,0,1,1,1,3,3])
    Wn = np.array([0,1,1,2,3])
    Ps, Pn = compute_PW_T(Ws,Wn,k=0,c=4)
    assert np.allclose(Ps,[0.5,0.3,0. ,0.2],atol=0.1)
    assert np.allclose(Pn,[0.2,0.4,0.2,0.2],atol=0.1)
    Ps, Pn = compute_PW_T(Ws,Wn,k=1,c=4)
    assert np.allclose(Ps,[0.42857143,0.28571429,0.07142857,0.21428571],atol=0.01)
    assert np.allclose(Pn,[0.22222222,0.33333333,0.22222222,0.22222222],atol=0.01)
#---------------------------------------------------
def test_likelihood_ratio():
    ''' (4 points) likelihood_ratio'''
    Ps = np.array([0.1,0.2,0.3,0.4])
    Pn = np.array([0.5,0.1,0.2,0.2])
    s = 0.2
    W = np.array([0,0])
    r=likelihood_ratio(W,Ps,Pn,s)
    assert np.allclose(r,0.01,atol=0.001)
    W = np.array([1,1])
    r=likelihood_ratio(W,Ps,Pn,s)
    assert np.allclose(r,1,atol=0.01)
    W = np.array([1,1,1])
    r=likelihood_ratio(W,Ps,Pn,s)
    assert np.allclose(r,2,atol=0.01)
    W = np.array([1,0,1,0])
    r=likelihood_ratio(W,Ps,Pn,s)
    assert np.allclose(r,0.04,atol=0.01)
    W = np.array([])
    r=likelihood_ratio(W,Ps,Pn,s)
    assert np.allclose(r,0.25,atol=0.01)
    W = np.array([1,0,1,0,3,2,2,1,3,2])
    r=likelihood_ratio(W,Ps,Pn,s)
    assert np.allclose(r,1.08,atol=0.01)

