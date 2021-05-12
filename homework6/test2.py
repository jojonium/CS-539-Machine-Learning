from problem2 import *
import sys
import math

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (50 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_sample_features():
    ''' (10 points) sample_features'''
    # a dataset of 3 features, 4 samples
    X = np.array([[1.,2.,3.,4. ],
                  [2.,4.,6.,8. ],
                  [3.,6.,9.,12.]])
    count= 0
    for i in range(5):
        X1,fid = sample_features(X,2) # sample 2 features from 3
        if i>0 and np.allclose(fid,fid_old): # each time we should have a different sample of features 
            count+=1 
        assert type(X1) == np.ndarray
        assert type(fid) == np.ndarray
        assert X1.shape == (2,4) 
        assert fid.shape == (2,)
        x = np.array([1,2,3,4])
        assert np.allclose(X1[0],x*(fid[0]+1)) 
        assert np.allclose(X1[1],x*(fid[1]+1)) 
        assert (X1[0]-X1[1]).sum()!=0 # the same feature cannot be sampled twice
        X1_old, fid_old = X1,fid 
    assert count<3
    for _ in range(5):
        n = np.random.randint(2,5)
        p1 = np.random.randint(100,200)
        p2 = np.random.randint(100,200)
        X = np.bmat([[np.zeros((p1,n))],[np.ones((p2,n))]])
        X1, fid = sample_features(X,100)
        assert X1.shape == (100,n)
        assert fid.shape == (100,)
        assert np.allclose(X1.sum()/n/100, float(p2)/(p1+p2),atol = 0.1)
#---------------------------------------------------
def test_random_forest():
    ''' (10 points) random_forest'''
    # 3 features, 2 training samples
    X = np.array([[1.,1.],
                  [1.,2.],
                  [3.,3.]])
    Y = np.array(['good','bad'])
    count =0
    for _ in range(20):
        T,Fs = random_forest(X,Y,1,2)  # create an ensemble of only one tree, each tree uses 2 features
        assert len(T) == 1
        assert len(Fs) == 1
        if Fs[0][0]==0 or Fs[0,1]==0:
            f0=True
        if Fs[0][0]==1 or Fs[0,1]==1:
            f1=True
        if Fs[0,0]==2 or Fs[0,1]==2:
            f2=True
        if T[0].root.isleaf:
            count+=1
    assert count>5
    assert count<20
    assert f0 and f1 and f2 # make sure both features have been sampled at least once
    # 3 features, 4 training samples
    X = np.array([[1,2,4,5],
                  [2,2,3,4],
                  [3,3,4,5]])
    Y = np.array(['good','good','bad','good'])
    T,Fs = random_forest(X,Y,20,2) # create an ensemble of 20 decision trees and 2 features
    assert len(T) == 20
    x = np.array([4,3,4])
    for t in T:
        p=t.predict_1(x)
        if p=='good':
            g = True
            count+=1
        if p=='bad':
            b = True
    assert g and b # different tree should be able to make different predictions
    # 3 features, 4 training samples
    X = np.array([[1,2,4,5],
                  [2,2,2,2],
                  [3,3,3,3]])
    Y = np.array(['good','good','bad','good'])
    T,Fs = random_forest(X,Y,20,2) # create an ensemble of 10 decision trees
    assert len(T) == 20
    x = np.array([4,2,3])
    for t in T:
        p=t.predict_1(x)
        if p=='good':
            g = True
        if p=='bad':
            b = True
    assert g and b # different tree should be able to make different predictions
#---------------------------------------------------
def test_predict_1():
    ''' (10 points) predict_1'''
    # 3 features, 4 training samples
    X = np.array([[1,2,4,5],
                  [2,2,2,2],
                  [3,3,3,3]])
    Y = np.array(['good','good','bad','good'])
    count =0
    for _ in range(3):
        T,F = random_forest(X,Y,11,2) # create an ensemble of 11 decision trees
        assert len(T) == 11
        x = np.array([4,2,3])
        p=predict_1(T,F,x)
        if p=='bad':
            count+=1
    assert count>0
#---------------------------------------------------
def test_predict():
    ''' (20 points) predict'''
    # load a dataset from file
    filename = 'data2.csv'
    X = np.loadtxt(filename, dtype=float, delimiter=',',skiprows=1,usecols=range(1,17), unpack=True)
    Y = np.loadtxt(filename, dtype=int, delimiter=',',skiprows=1,usecols=0, unpack=True)
    assert X.shape == (16,400)
    assert Y.shape == (400,)
    n = float(len(Y))
    for _ in range(3):
        # train a decision tree over half of the dataset
        t = DecisionTree(X[:,::2],Y[::2]) 
        # test on the other half
        Y_predict = t.predict(X[:,1::2]) 
        accuracy0 = sum(Y[1::2]==Y_predict)/float(n)*2. 
        print('test accuracy of a decision tree:', accuracy0)
        # train bagging over half of the dataset 
        T = p1.bagging(X[:,::2],Y[::2],11) 
        # test on the other half
        Y_predict = p1.predict(T,X[:,1::2]) 
        accuracy1 = sum(Y[1::2]==Y_predict)/n*2. 
        print('test accuracy of bagging:', accuracy1)
        assert accuracy1 >= .75
        # train over half of the dataset
        T,F = random_forest(X[:,::2],Y[::2],11,10) 
        # test on the other half
        Y_predict = predict(T,F,X[:,1::2]) 
        accuracy2 = sum(Y[1::2]==Y_predict)/n*2. 
        print('test accuracy of random forest:', accuracy2)
        assert accuracy2 >= accuracy0
        assert accuracy2 >= accuracy1-.05

