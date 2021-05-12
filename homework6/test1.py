from problem1 import *
import sys
import math

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (50 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_bootstrap():
    ''' (10 points) bootstrap'''
    # a dataset of 3 features, 4 samples
    X = np.array([[1.,2.,3.,4. ],
                  [2.,4.,6.,8. ],
                  [3.,6.,9.,12.]])
    Y = np.array( [1.,2.,3.,4. ])
    count= 0
    count2= 0
    count3= 0
    for i in range(5):
        X1, Y1 = bootstrap(X,Y)
        if len(np.unique(Y1))<4: # some samples may not be used
            count3+=1
        if i>0 and np.allclose(Y1,Y1_old): # each time we should have a different bootstrap sample
            count2+=1 
        assert type(X1) == np.ndarray
        assert type(Y1) == np.ndarray
        assert X1.shape == (3,4) 
        assert Y1.shape == (4,)
        assert np.allclose(X1[0]*2, X1[1])
        assert np.allclose(X1[0]*3, X1[2])
        assert np.allclose(X1[0], Y1)
        if np.allclose(Y1,Y): # the bootstrap sample should be different from the original dataset
            count+=1
        X1_old, Y1_old = X1, Y1
    assert count<2
    assert count2<2
    assert count3>2
    for _ in range(20):
        p = np.random.randint(10,20)
        n1 = np.random.randint(200,500)
        n2 = np.random.randint(200,500)
        X = np.bmat([np.zeros((p,n1)),np.ones((p,n2))])
        Y = np.bmat([np.ones(n1),np.zeros(n2)]).getA1()
        X1, Y1 = bootstrap(X,Y)
        assert X1.shape == (p,n1+n2)
        assert Y1.shape == (n1+n2,)
        assert np.allclose(Y1.sum()/(n1+n2), float(n1)/(n1+n2),atol = 0.1)
        assert np.allclose(X1.sum()/(n1+n2)/p, float(n2)/(n1+n2),atol = 0.1)
#---------------------------------------------------
def test_bagging():
    ''' (10 points) bagging'''
    # 3 features, 4 training samples
    X = np.array([[1.,1.,1.,1.],
                  [2.,2.,2.,2.],
                  [3.,3.,3.,3.]])
    Y = np.array(['good','good','good','good'])
    T = bagging(X,Y,1)  # create an ensemble of only one tree 
    assert len(T) == 1
    t = T[0]
    assert t.root.isleaf == True
    assert t.root.p == 'good' 
    for _ in range(20):
        n_tree = np.random.randint(1,10)
        T = bagging(X,Y,n_tree)  # create an ensemble of multiple trees
        assert len(T) == n_tree
        for i in range(n_tree):
            t = T[i]
            assert t.root.isleaf == True
            assert t.root.p == 'good' 
    # 3 features, 2 training samples
    X = np.array([[1.,1.],
                  [2.,2.],
                  [3.,3.]])
    Y = np.array(['good','bad'])
    T = bagging(X,Y,10) # create an ensemble of 10 decision trees
    assert len(T) == 10
    for t in T:
        assert t.root.isleaf == True
        p=t.predict(X)
        if p[0]=='good':
            g = True
        if p[0]=='bad':
            b = True
    assert g and b # different tree should be able to make different predictions
#---------------------------------------------------
def test_predict_1():
    ''' (10 points) predict_1'''
    # 3 features, 4 training samples
    X = np.array([[1.,1.,1.,1.],
                  [2.,2.,2.,2.],
                  [3.,3.,3.,3.]])
    Y = np.array(['good','good','good','bad'])
    T = bagging(X,Y,11) 
    x = np.array([1,2,3])
    y = predict_1(T,x)
    assert y =='good'
    # 3 features, 2 training samples
    X = np.array([[1.,1.],
                  [2.,2.],
                  [3.,3.]])
    Y = np.array(['good','bad'])
    for _ in range(10):
        T = bagging(X,Y,5) # create an ensemble of 5 decision trees
        y=predict_1(T,x)
        if y=='good':
            g = True
        if y=='bad':
            b = True
    assert g and b # different tree should be able to make different predictions
#---------------------------------------------------
def test_predict():
    ''' (20 points) predict'''
    # 3 features, 4 training samples
    X = np.array([[1.,1.,1.,2.],
                  [2.,2.,2.,2.],
                  [3.,3.,3.,3.]])
    Y = np.array(['good','good','good','bad'])
    T = bagging(X,Y,11) 
    Xt= np.array([[1.,2.],
                  [2.,2.],
                  [3.,3.]])
    Yt = predict(T,Xt)
    assert len(Yt) ==2
    assert Yt[0] =='good'
    assert Yt[1] =='bad'
    # load a dataset from file
    filename = 'data1.csv'
    X = np.loadtxt(filename, dtype=float, delimiter=',',skiprows=1,usecols=[1,2], unpack=True)
    Y = np.loadtxt(filename, dtype=int, delimiter=',',skiprows=1,usecols=0, unpack=True)
    n = float(len(Y))
    for _ in range(3):
        # train over half of the dataset using one decision only 
        T = bagging(X[:,::2],Y[::2],1) 
        # test on the other half
        Y_predict = predict(T,X[:,1::2]) 
        accuracy1 = sum(Y[1::2]==Y_predict)/n*2. 
        print('test accuracy of one decision tree:', accuracy1)
        assert accuracy1 >= .93
        # train over half of the dataset
        T = bagging(X[:,::2],Y[::2],11) # train with 11 trees 
        # test on the other half
        Y_predict = predict(T,X[:,1::2]) 
        accuracy2 = sum(Y[1::2]==Y_predict)/n*2. 
        print('test accuracy of a bagging ensemble of 11 trees:', accuracy2)
        assert accuracy2 >= .95

