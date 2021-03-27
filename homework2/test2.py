from problem2 import *
import sys
import math
from gradient2 import *
import warnings
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z():
    ''' (1 points) compute_z'''
    x = np.array([1., 2., 3.])
    W = np.array([[0.5,-0.6,0.3],
                  [0.6,-0.5,0.2]])
    b = np.array([0.2,0.3])
    z = compute_z(x,W,b)
    assert type(z) == np.ndarray
    assert z.shape == (2,)
    assert np.allclose(z, [0.4,0.5], atol = 1e-3) 
    x = np.array([2., 5.,2.])
    z = compute_z(x,W,b)
    assert np.allclose(z, [-1.2,-0.6], atol = 1e-3)
#---------------------------------------------------
def test_compute_dz_db():
    ''' (1 points) compute_dz_db'''
    for _ in range(20):
        c = np.random.randint(2,10)
        p = np.random.randint(2,20)
        x = np.random.random(p)
        W = np.random.random((c,p))
        b = np.random.random(c)
        dz_db = compute_dz_db(c)
        dz_db_true = check_dz_db(x,W,b)
        assert np.allclose(dz_db, dz_db_true, atol=1e-3) 
#---------------------------------------------------
def test_compute_dL_db():
    ''' (1 points) compute_dL_db'''
    dL_dz = np.array([1.2,3.5]) 
    dz_db = np.array([[1.,0.],
                      [0.,1.]])
    dL_db = compute_dL_db(dL_dz, dz_db)
    assert type(dL_db) == np.ndarray
    assert dL_db.shape == (2,) 
    assert np.allclose(dL_db, [1.2,3.5], atol=1e-2) 
    for _ in range(20):
        p = np.random.randint(2,10) # number of features
        c = np.random.randint(2,10) # number of classes
        x = np.random.random(p)
        y = np.random.randint(c) 
        W = np.random.random((c,p))
        b = np.random.random(c)
        a = forward(x,W,b)
        # analytical gradients
        dL_da = compute_dL_da(a,y)
        da_dz = compute_da_dz(a)
        dz_db = compute_dz_db(c)
        dL_dz = compute_dL_dz(dL_da, da_dz)
        dL_db = compute_dL_db(dL_dz,dz_db)
        # numerical gradients
        dL_db_true = check_dL_db(x,y,W,b) 
        assert np.allclose(dL_db.shape, dL_db_true.shape) 
        assert np.allclose(dL_db, dL_db_true, atol = 1e-3)
#---------------------------------------------------
def test_compute_dz_dW():
    ''' (1 points) compute_dz_dW'''
    for _ in range(20):
        c = np.random.randint(2,10)
        p = np.random.randint(2,20)
        x = np.random.random(p)
        W = np.random.random((c,p))
        b = np.random.random(c)
        # analytical gradients
        dz_dW = compute_dz_dW(x,c)
        # numerical gradients
        dz_dW_true = check_dz_dW(x,W,b)
        assert np.allclose(dz_dW, dz_dW_true, atol=1e-3) 
#---------------------------------------------------
def test_compute_dL_dW():
    ''' (2 points) compute_dL_dW'''
    dL_dz = np.array([0.5, -0.5])
    dz_dW = np.array([[[ 1., 2., 3.], 
                       [ 0., 0., 0.]],
                      [[ 0., 0., 0.], 
                       [ 1., 2., 3.]]])
    dL_dW = compute_dL_dW(dL_dz, dz_dW) 
    assert dL_dW.shape == (2,3) 
    dL_dW_true = [[ 0.5,  1.0,  1.5],
                  [-0.5, -1.0, -1.5]]
    assert np.allclose(dL_dW, dL_dW_true, atol = 1e-3)
    for _ in range(20):
        p = np.random.randint(2,10) # number of features
        c = np.random.randint(2,10) # number of classes
        x = np.random.random(p)
        y = np.random.randint(c) 
        W = np.random.random((c,p))
        b = np.random.random(c)
        a = forward(x,W,b)
        # analytical gradients
        dL_da = compute_dL_da(a,y)
        da_dz = compute_da_dz(a)
        dz_dW = compute_dz_dW(x,c)
        dL_dz = compute_dL_dz(dL_da, da_dz)
        dL_dW = compute_dL_dW(dL_dz,dz_dW)
        # numerical gradients
        dL_dW_true = check_dL_dW(x,y,W,b) 
        assert np.allclose(dL_dW, dL_dW_true, atol = 1e-3)
#---------------------------------------------------
def test_compute_a():
    ''' (2 points) compute_a'''
    z = np.array([1., 1.])
    a = compute_a(z)
    assert type(a) == np.ndarray
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    assert np.allclose(z, [1., 1.]) 
    a = compute_a(np.array([1., 1., 1., 1.]))
    assert np.allclose(a, [0.25, 0.25, 0.25, 0.25], atol = 1e-2) 
    a = compute_a(np.array([-1., -1., -1., -1.]))
    assert np.allclose(a, [0.25, 0.25, 0.25, 0.25], atol = 1e-2) 
    a = compute_a(np.array([-2., -1.,1., 2.]))
    assert np.allclose(a, [0.01275478,0.03467109,0.25618664,0.69638749], atol = 1e-2)
    a = compute_a(np.array([100., 100.]))
    assert np.allclose(a, [.5, .5], atol = 1e-2) 
    a = compute_a(np.array([-100., -100.]))
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    z = np.array([1000., 1000.])
    a = compute_a(z)
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    assert np.allclose(z, [1000, 1000]) 
    z = np.array([-1000., -1000.])
    a = compute_a(z)
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    assert np.allclose(z, [-1000, -1000]) 
    a = compute_a(np.array([1000., 10.]))
    assert np.allclose(a,[1., 0.], atol = 1e-2) 
    a = compute_a(np.array([-1000., -10.]))
    assert np.allclose(a, [0., 1.], atol = 1e-2) 
    z = np.array([1000., 3000.])
    a = compute_a(z)
    assert np.allclose(a, [0., 1.], atol = 1e-2) 
    assert np.allclose(z, [1000, 3000]) 
    z = np.array([1000., 3000., -100, 0, -1000, 3000, 0])
    a = compute_a(z)
    assert np.allclose(a, [0., .5, 0, 0, 0, .5, 0], atol = 1e-2) 
    assert np.allclose(z, [1000., 3000., -100, 0, -1000, 3000, 0]) 
#---------------------------------------------------
def test_compute_da_dz():
    ''' (2 points) compute_da_dz'''
    a  = np.array([0.3, 0.7])
    da_dz = compute_da_dz(a)
    assert type(da_dz) == np.ndarray
    assert da_dz.shape == (2,2)
    assert np.allclose(da_dz, [[.21,-.21],[-.21,.21]], atol= 1e-3)
    a  = np.array([0.1, 0.2, 0.7])
    da_dz = compute_da_dz(a)
    assert da_dz.shape == (3,3)
    da_dz_true = np.array( [[ 0.09, -0.02, -0.07],
                            [-0.02,  0.16, -0.14],
                            [-0.07, -0.14,  0.21]])
    assert np.allclose(da_dz,da_dz_true,atol= 1e-3)
    for _ in range(20):
        c = np.random.randint(2,10)
        z = np.random.random(c)
        a  = compute_a(z)
        # analytical gradients
        dz = compute_da_dz(a)
        # numerical gradients
        dz_true = check_da_dz(z)
        assert np.allclose(dz, dz_true, atol= 1e-3) 
#---------------------------------------------------
def test_compute_dL_dz():
    ''' (2 points) compute_dL_dz'''
    dL_da = np.array([1.,2.])
    da_dz = np.array([[0.1,0.3],
                      [0.2,0.4]])
    dL_dz = compute_dL_dz(dL_da, da_dz) 
    assert type(dL_dz) == np.ndarray
    assert dL_dz.shape == (2,) 
    dL_dz_true = [.5, 1.1]
    assert np.allclose(dL_dz, dL_dz_true, atol = 1e-3)
#---------------------------------------------------
def test_compute_L():
    ''' (2 points) compute_L'''
    
    a = np.array([0.,1.]) 
    L = compute_L(a,1)
    assert np.allclose(L, 0., atol = 1e-3) 
    assert np.allclose(a, [0.,1.]) 
    a = np.array([0., 0., 1.])
    L = compute_L(a, 2)
    assert np.allclose(L, 0., atol = 1e-3) 
    a = np.array([0.,1.]) 
    L= compute_L(a, 1)
    assert np.allclose(L, 0., atol = 1e-3) 
    a = np.array([.5,.5]) 
    L= compute_L(a, 0)
    assert np.allclose(L, 0.69314718056, atol = 1e-3) 
    L= compute_L(a, 1)
    assert np.allclose(L, 0.69314718056, atol = 1e-3) 
    a = np.array([1.,0.]) 
    L= compute_L(a, 1)
    assert L > 1000
    assert L < float("inf")
    a = np.array([0.,1.]) 
    L= compute_L(a, 0)
    assert L > 1000
    assert L < float("inf")
    a = np.array([0.,1.,0.]) 
    L= compute_L(a, 1)
    assert np.allclose(L,0)
#---------------------------------------------------
def test_compute_dL_da():
    ''' (2 points) compute_dL_da'''
    a  = np.array([0.5,0.5])
    y = 1
    dL_da = compute_dL_da(a,y)
    assert type(dL_da) == np.ndarray
    assert dL_da.shape == (2,) 
    assert np.allclose(dL_da,[0.,-2.], atol= 1e-3)
    a  = np.array([0.5,0.5])
    y = 0
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, [-2.,0.], atol= 1e-3)
    a  = np.array([0.1,0.6,0.1,0.2])
    y = 3
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, [0.,0.,0.,-5.], atol= 1e-3)
    a  = np.array([1.,0.])
    y = 1
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da[0], 0., atol= 1e-3)
    assert dL_da[1] < -1000
    assert dL_da[1] > -float("Inf")
    assert np.allclose(a, [1.,0.])
    a  = np.array([0., 1.,0.,0.])
    y = 3
    dL_da = compute_dL_da(a,y)
    assert dL_da.shape == (4,) 
    assert np.allclose(dL_da[0], 0., atol= 1e-3)
    assert np.allclose(dL_da[1], 0., atol= 1e-3)
    assert np.allclose(dL_da[2], 0., atol= 1e-3)
    assert dL_da[3] < -1000
    assert dL_da[3] > -float("Inf")
    assert np.allclose(a, [0., 1.,0.,0.])
    y = 1
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, [0,-1,0,0], atol= 1e-3)
    for _ in range(20):
        c = np.random.randint(2,10) # number of classes
        a  = np.random.random(c) # activation
        a = (a + 1)/(a+1).sum()
        y = np.random.randint(c) # label 
        # analytical gradients
        da = compute_dL_da(a,y)
        # numerical gradients
        da_true = check_dL_da(a,y)
        assert np.allclose(da, da_true, atol= 1e-2)
#---------------------------------------------------
def test_forward():
    ''' (1 points) forward'''
    x = np.array([1., 2.,3.])
    W = np.array([[0.,0.,0.],
                  [0.,0.,0.]])
    b = np.array([0.,0.])
    y = 1
    a = forward(x,W,b) 
    assert type(a) == np.ndarray
    assert a.shape == (2,) 
    a_true = [0.5,0.5]
    assert np.allclose(a, a_true, atol = 1e-3)
#---------------------------------------------------
def test_backward():
    ''' (4 points) backward'''
    x = np.array([1., 2.,3.])
    y = 1
    a = np.array([.5, .5])
    dL_dW, dL_db = backward(x,y,a) 
    assert type(dL_dW) == np.ndarray
    assert type(dL_db) == np.ndarray
    assert dL_dW.shape == (2,3) 
    assert dL_db.shape == (2,) 
    assert np.allclose(dL_dW, [[ 0.5, 1., 1.5],[-0.5,-1.,-1.5]], atol = 1e-3)
    assert np.allclose(dL_db, [0.5,-0.5], atol = 1e-3)
    for _ in range(20):
        p = np.random.randint(2,10) # number of features
        c = np.random.randint(2,10) # number of classes
        x = np.random.random(p)
        y = np.random.randint(c) 
        W = np.random.random((c,p))
        b = np.random.random(c)
        a = forward(x,W,b)
        # analytical gradients
        dL_dW, dL_db = backward(x,y,a) 
        # numerical gradients
        dL_db_true = check_dL_db(x,y,W,b) 
        dL_dW_true = check_dL_dW(x,y,W,b) 
        assert np.allclose(dL_db.shape, dL_db_true.shape) 
        assert np.allclose(dL_db, dL_db_true, atol = 1e-3)
        assert np.allclose(dL_dW, dL_dW_true, atol = 1e-3)
#---------------------------------------------------
def test_update_b():
    ''' (1 points) update_b'''
    b = np.array([0.,0.])
    dL_db = np.array([0.5, -0.5])
    b = update_b(b, dL_db, alpha=1.)
    b_true = np.array([-0.5,0.5])
    assert np.allclose(b, b_true, atol = 1e-3)
    b = np.array([0.,0.])
    b = update_b(b, dL_db, alpha=10.)
    b_true = np.array([-5.,5.])
    assert np.allclose(b, b_true, atol = 1e-3)
#---------------------------------------------------
def test_update_W():
    ''' (1 points) update_W'''
    W = np.array([[0., 0.,0.],
                  [0.,0.,0.]])
    dL_dW = np.array([[ 0.5,  1.0,  1.5],
                      [-0.5, -1.0, -1.5]])
    W = update_W(W, dL_dW, alpha=1.) 
    W_true = [[-0.5,-1.,-1.5],
               [ 0.5, 1., 1.5]]
    assert np.allclose(W, W_true, atol = 1e-3)
    W = np.array([[0., 0.,0.],
                  [0.,0.,0.]])
    W = update_W(W, dL_dW, alpha=10.)
    W_true = [[-5., -10., -15.],
               [ 5.,  10.,  15.]]
    assert np.allclose(W, W_true, atol = 1e-3)
#---------------------------------------------------
def test_train():
    ''' (1 points) train'''
    # an example feature matrix (4 instances, 2 features)
    Xtrain  = np.array( [[0., 1.],
                         [1., 0.],
                         [0., 0.],
                         [1., 1.]])
    Ytrain = [0, 1, 0, 1]
    W, b = train(Xtrain, Ytrain,alpha=0.01, n_epoch=2000)
    assert b[0] > b[1] # x3 is negative 
    assert W[1,0] + W[1,1] + b[1] > W[0,0] + W[0,1] + b[0] # x4 is positive
    assert W[1,1] + b[1] < W[0,1] + b[0] # x1 is negative 
    assert W[1,0] + b[1] > W[0,0] + b[0] # x2 is positive 
#---------------------------------------------------
def test_inference():
    ''' (1 points) inference'''
    W = np.array([[0.4, 0.],
                  [0.6, 0.]])
    b = np.array([0.1, 0.])
    x= np.array([1,1])
    y, a= inference(x, W, b )
    assert y==1
    assert np.allclose(a, [ 0.47502081, 0.52497919], atol = 1e-2)
    x= np.array([0,1])
    y, a= inference(x, W, b )
    assert y==0
    assert np.allclose(a, [ 0.52497919, 0.47502081], atol = 1e-2)
    W = np.array([[0.4, 0.],
                  [0.2,-.2],
                  [0.4, 0.]])
    b = np.array([0.1, 0.,0.1])
    x= np.array([1,1])
    y, a= inference(x, W, b )
    assert y==0
    assert np.allclose(a, [0.38365173, 0.23269654, 0.38365173], atol = 1e-2) 
#---------------------------------------------------
def test_predict():
    ''' (5 points) predict'''
    
    # an example feature matrix (4 instances, 2 features)
    Xtest  = np.array([[0., 1.],
                       [1., 0.],
                       [0., 0.],
                       [1., 1.]])
    W = np.array([[0.4, 0.],
                  [0.6, 0.]])
    b = np.array([0.1, 0.])
    Ytest, Ptest = predict(Xtest, W, b )
    assert type(Ytest) == np.ndarray
    assert Ytest.shape == (4,)
    assert type(Ptest) == np.ndarray
    assert Ptest.shape == (4,2)
    Ytest_true = [0, 1,0, 1]
    Ptest_true = [[ 0.52497919, 0.47502081],
                  [ 0.47502081, 0.52497919],
                  [ 0.52497919, 0.47502081],
                  [ 0.47502081, 0.52497919]] 
    assert np.allclose(Ytest, Ytest_true, atol = 1e-2)
    assert np.allclose(Ptest, Ptest_true, atol = 1e-2)
    n_samples = 400
    X = np.loadtxt("X2.csv",dtype=float,delimiter=",") 
    y = np.loadtxt("y2.csv",dtype=int,delimiter=",") 
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    w,b = train(Xtrain, Ytrain,alpha=.01, n_epoch=100)
    Y, P = predict(Xtrain, w, b)
    accuracy = sum(Y == Ytrain)/(n_samples/2.)
    print("Training accuracy:", accuracy)
    assert accuracy > 0.9
    Y, P = predict(Xtest, w, b)
    accuracy = sum(Y == Ytest)/(n_samples/2.)
    print("Test accuracy:", accuracy)
    assert accuracy > 0.9

