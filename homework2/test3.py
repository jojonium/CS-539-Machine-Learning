from problem3 import *
import sys
import math
from gradient3 import *
import warnings
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (40 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z1():
    ''' (1 points) compute_z1'''
    x = np.array([1., 2., 3.])
    W1 = np.array([[0.5,-0.6,0.3],
                   [0.6,-0.5,0.2]])
    b1 = np.array([0.2, 0.3])
    z1 = compute_z1(x,W1,b1)
    assert type(z1) == np.ndarray
    assert z1.shape == (2,)
    assert np.allclose(z1, [0.4,0.5], atol = 1e-3) 
    x = np.array([2., 5.,2.])
    z1 = compute_z1(x,W1,b1)
    assert np.allclose(z1, [-1.2,-0.6], atol = 1e-3) 
#---------------------------------------------------
def test_compute_dz1_db1():
    ''' (1 points) compute_dz1_db1'''
    dz_db = compute_dz1_db1(2)
    assert type(dz_db) == np.ndarray
    assert dz_db.shape == (2,2) 
    dz_db_true = np.array([[1.,0.],
                           [0.,1.]])
    assert np.allclose(dz_db, dz_db_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dL_db1():
    ''' (1 points) compute_dL_db1'''
    
    dL_dz1 = np.array([1.2,3.5]) 
    dz1_db1 = np.array([[1.,0.],
                        [0.,1.]])
    dL_db1 = compute_dL_db1(dL_dz1, dz1_db1)
    assert type(dL_db1) == np.ndarray
    assert dL_db1.shape == (2,) 
    assert np.allclose(dL_db1, [1.2,3.5], atol=1e-2) 
#---------------------------------------------------
def test_compute_dz1_dW1():
    ''' (1 points) compute_dz1_dW1'''
    x = np.array([1., 2.,3.])
    dz_dW = compute_dz1_dW1(x,2)
    assert type(dz_dW) == np.ndarray
    assert dz_dW.shape == (2,2,3) 
    dz_dW_true = np.array([[[1., 2.,3],
                            [0., 0.,0]],
                           [[0., 0.,0],
                            [1., 2.,3]]])
    assert np.allclose(dz_dW, dz_dW_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dL_dW1():
    ''' (1 points) compute_dL_dW1'''
    dL_dz1 = np.array([1,2]) 
    dz1_dW1 = np.array([[[1,2,3],
                        [0,0,0]],
                       [[0,0,0],
                        [1,2,3]]])
    dL_dW1 = compute_dL_dW1(dL_dz1, dz1_dW1)
    assert type(dL_dW1) == np.ndarray
    assert dL_dW1.shape == (2,3) 
    dL_dW1_true = np.array([[1,2,3],
                            [2,4,6]])
    assert np.allclose(dL_dW1, dL_dW1_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_a1():
    ''' (2 points) compute_a1'''
    z1 = np.array([0.,1.])
    a1 = compute_a1(z1)
    assert type(a1) == np.ndarray
    assert a1.shape == (2,)
    assert np.allclose(a1, [0.5,0.731], atol = 1e-3) 
    z1 = np.array([-1.,-100., 100])
    a1 = compute_a1(z1)
    assert a1.shape == (3,)
    assert np.allclose(a1, [0.2689, 0, 1], atol = 1e-2) 
    z1 = np.array([1000., 1000.])
    a1 = compute_a1(z1)
    assert np.allclose(a1, [1., 1.], atol = 1e-2) 
    assert np.allclose(z1, [1000, 1000]) 
    z1 = np.array([-1000., -1000.])
    a1 = compute_a1(z1)
    assert np.allclose(a1, [0., 0.], atol = 1e-2) 
    assert np.allclose(z1, [-1000, -1000]) 
    a1 = compute_a1(np.array([1000., 100.]))
    assert np.allclose(a1, [1., 1.], atol = 1e-2) 
    a = compute_a1(np.array([-1000., -10.]))
    assert np.allclose(a, [0., 0.], atol = 1e-2) 
#---------------------------------------------------
def test_compute_da1_dz1():
    ''' (2 points) compute_da1_dz1'''
    a1= np.array([.5,.5,.3,.6])
    da1_dz1 = compute_da1_dz1(a1)
    assert type(da1_dz1) == np.ndarray
    assert da1_dz1.shape == (4,4)
    da_dz_true = np.array([[.25,  0,  0,  0],
                           [  0,.25,  0,  0],
                           [  0,  0,.21,  0],
                           [  0,  0,  0,.24]])
    assert np.allclose(da1_dz1,da_dz_true , atol= 1e-3)
    # gradient-checking
    for _ in range(20):
        c = np.random.randint(2,10)
        z1 = np.random.random(c)
        a1  = compute_a1(z1)
        # analytical gradients
        dz = compute_da1_dz1(a1)
        # numerical gradients
        dz_true = check_da1_dz1(z1)
        assert np.allclose(dz, dz_true, atol= 1e-3) 
#---------------------------------------------------
def test_compute_dL_dz1():
    ''' (2 points) compute_dL_dz1'''
    dL_da1  = np.array([-0.03777044, 0.29040313,-0.42821076,-0.28597724])
    da1_dz1 = np.diag([ 0.03766515, 0.09406613, 0.06316817, 0.05718137])
    dL_dz1 = compute_dL_dz1(dL_da1, da1_dz1)
    assert type(dL_dz1) == np.ndarray
    assert dL_dz1.shape == (4,) 
    dL_dz1_true = np.array([-0.00142263, 0.0273171,  -0.02704929,-0.01635257])
    assert np.allclose(dL_dz1, dL_dz1_true, atol=1e-3) 
#---------------------------------------------------
def test_compute_z2():
    ''' (2 points) compute_z2'''
    x = np.array([1., 2., 3.])
    W2 = np.array([[0.5,-0.6,0.3],
                   [0.6,-0.5,0.2]])
    b2 = np.array([0.2, 0.3])
    z2 = compute_z2(x,W2,b2)
    assert type(z2) == np.ndarray
    assert z2.shape == (2,)
    assert np.allclose(z2, [0.4,0.5], atol = 1e-3) 
    x = np.array([2., 5.,2.])
    z2 = compute_z2(x,W2,b2)
    assert np.allclose(z2, [-1.2,-0.6], atol = 1e-3) 
#---------------------------------------------------
def test_compute_dz2_db2():
    ''' (1 points) compute_dz2_db2'''
    dz_db = compute_dz2_db2(2)
    assert type(dz_db) == np.ndarray
    assert dz_db.shape == (2,2) 
    dz_db_true = np.array([[1.,0.],
                           [0.,1.]])
    assert np.allclose(dz_db, dz_db_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dL_db2():
    ''' (1 points) compute_dL_db2'''
    dL_dz2 = np.array([1.2,3.5]) 
    dz2_db2 = np.array([[1.,0.],
                        [0.,1.]])
    dL_db2 = compute_dL_db2(dL_dz2, dz2_db2)
    assert type(dL_db2) == np.ndarray
    assert dL_db2.shape == (2,) 
    assert np.allclose(dL_db2, [1.2,3.5], atol=1e-2) 
#---------------------------------------------------
def test_compute_dz2_dW2():
    ''' (2 points) compute_dz2_dW2'''
    x = np.array([1., 2.,3.])
    dz_dW = compute_dz2_dW2(x,2)
    assert type(dz_dW) == np.ndarray
    assert dz_dW.shape == (2,2,3) 
    dz_dW_true = np.array([[[1., 2.,3],
                            [0., 0.,0]],
                           [[0., 0.,0],
                            [1., 2.,3]]])
    assert np.allclose(dz_dW, dz_dW_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dL_dW2():
    ''' (1 points) compute_dL_dW2'''
    dL_dz2 = np.array([1,2]) 
    dz2_dW2 = np.array([[[1,2,3],
                        [0,0,0]],
                       [[0,0,0],
                        [1,2,3]]])
    dL_dW2 = compute_dL_dW2(dL_dz2, dz2_dW2)
    assert type(dL_dW2) == np.ndarray
    assert dL_dW2.shape == (2,3) 
    dL_dW2_true = np.array([[1,2,3],
                            [2,4,6]])
    assert np.allclose(dL_dW2, dL_dW2_true, atol=1e-2) 
#---------------------------------------------------
def test_compute_dz2_da1():
    ''' (2 points) compute_dz2_da1'''
    W2= np.array([[1.,
                   .4,3.],
                  [8.,.5,
                  .2]])+.32
    dz2_da1 = compute_dz2_da1(W2)
    assert type(dz2_da1) == np.ndarray
    assert dz2_da1.shape == (2,3)
    assert np.allclose(dz2_da1, [[ 1.32, 0.72, 3.32], [ 8.32, 0.82, 0.52]], atol= 1e-3)
#---------------------------------------------------
def test_compute_dL_da1():
    ''' (1 points) compute_dL_da1'''
    dL_dz2 = np.array([ 0.09554921, 0.14753129, 0.47769828,-0.72077878])
    dz2_da1 = np.array([[ 0.26739761, 0.73446399, 0.24513834],
                        [ 0.80682023, 0.7841972 , 0.01415917],
                        [ 0.70592854, 0.73489433, 0.91355454],
                        [ 0.8558265 , 0.84993468, 0.24702029]]) 
    dL_da1 = compute_dL_da1(dL_dz2,dz2_da1)
    assert type(dL_da1) == np.ndarray
    assert dL_da1.shape == (3,) 
    dL_da1_true = np.array([-0.13505987,-0.07568605, 0.28386814])
    assert np.allclose(dL_da1, dL_da1_true, atol=1e-3) 
#---------------------------------------------------
def test_compute_a2():
    ''' (2 points) compute_a2'''
    z = np.array([1., 1.])
    a = compute_a2(z)
    assert type(a) == np.ndarray
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    assert np.allclose(z, [1., 1.]) 
    a = compute_a2(np.array([1., 1.,1., 1.]))
    assert np.allclose(a, [0.25, 0.25, 0.25, 0.25], atol = 1e-2) 
    a = compute_a2(np.array([-1., -1.,-1., -1.]))
    assert np.allclose(a, [0.25, 0.25, 0.25, 0.25], atol = 1e-2) 
    a = compute_a2(np.array([-2., -1.,1., 2.]))
    assert np.allclose(a, [ 0.01275478,0.03467109,0.25618664,0.69638749], atol = 1e-2)
    a = compute_a2(np.array([100., 100.]))
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    a = compute_a2(np.array([-100., -100.]))
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    z = np.array([1000., 1000.])
    a = compute_a2(z)
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    assert np.allclose(z, [1000, 1000]) 
    z = np.array([-1000., -1000.])
    a = compute_a2(z)
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 
    assert np.allclose(z, [-1000, -1000]) 
    a = compute_a2(np.array([1000., 10.]))
    assert np.allclose(a, [1., 0.], atol = 1e-2) 
    a = compute_a2(np.array([-1000., -10.]))
    assert np.allclose(a, [0., 1.], atol = 1e-2) 
#---------------------------------------------------
def test_compute_da2_dz2():
    ''' (1 points) compute_da2_dz2'''
    a  = np.array([0.3, 0.7])
    da_dz = compute_da2_dz2(a)
    assert type(da_dz) == np.ndarray
    assert da_dz.shape == (2,2)
    assert np.allclose(da_dz, [[.21,-.21],[-.21,.21]], atol= 1e-3)
    a  = np.array([0.1, 0.2, 0.7])
    da_dz = compute_da2_dz2(a)
    assert da_dz.shape == (3,3)
    da_dz_true = np.array([[ 0.09, -0.02, -0.07],
                           [-0.02,  0.16, -0.14],
                           [-0.07, -0.14,  0.21]])
    assert np.allclose(da_dz,da_dz_true,atol= 1e-3)
#---------------------------------------------------
def test_compute_dL_dz2():
    ''' (1 points) compute_dL_dz2'''
    dL_da2  = np.array([-0.03777044, 0.29040313,-0.42821076,-0.28597724])
    da2_dz2 = np.diag([ 0.03766515, 0.09406613, 0.06316817, 0.05718137])
    dL_dz2 = compute_dL_dz2(dL_da2, da2_dz2)
    assert type(dL_dz2) == np.ndarray
    assert dL_dz2.shape == (4,) 
    dL_dz2_true = np.array([-0.00142263, 0.0273171,  -0.02704929,-0.01635257])
    assert np.allclose(dL_dz2, dL_dz2_true, atol=1e-3) 
#---------------------------------------------------
def test_compute_L():
    ''' (1 points) compute_L'''
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
def test_compute_dL_da2():
    ''' (1 points) compute_dL_da2'''
    a  = np.array([0.5,0.5])
    y = 1
    dL_da = compute_dL_da2(a,y)
    assert type(dL_da) == np.ndarray
    assert dL_da.shape == (2,) 
    assert np.allclose(dL_da, [0.,-2.], atol= 1e-3)
    a  = np.array([0.5,0.5])
    y = 0
    dL_da = compute_dL_da2(a,y)
    assert np.allclose(dL_da, [-2.,0.], atol= 1e-3)
    a  = np.array([0.1,0.6,0.1,0.2])
    y = 3
    dL_da = compute_dL_da2(a,y)
    assert np.allclose(dL_da, [0.,0.,0.,-5.], atol= 1e-3)
    a  = np.array([1.,0.])
    y = 1
    dL_da = compute_dL_da2(a,y)
    assert np.allclose(dL_da[0], 0., atol= 1e-3)
    assert dL_da[1] < -1000
    assert dL_da[1] > -float("Inf")
    assert np.allclose(a.T, [1.,0.])
#---------------------------------------------------
def test_forward():
    ''' (1 points) forward'''
    x = np.array([1., 2.,3.,4])
    # first layer with 3 neurons
    W1 = np.array([[0.,0.,0.,0.],
                   [0.,0.,0.,0.],
                   [0.,0.,0.,0.]])
    b1 = np.array([0.,0.,0.])
    # second layer with 2 neurons
    W2 = np.array([[0.,0.,0.],
                   [0.,0.,0.]])
    b2 = np.array([100.,0.])
    a1, a2 = forward(x,W1,b1,W2,b2) 
    assert type(a1) == np.ndarray
    assert a1.shape == (3,)
    assert type(a2) == np.ndarray
    assert a2.shape == (2,)
    assert np.allclose(a1, [0.5,0.5,0.5], atol = 1e-3)
    assert np.allclose(a2, [1,0], atol = 1e-3)
#---------------------------------------------------
def test_backward_layer2():
    ''' (1 points) backward_layer2'''
    x = np.array([1., 2.,3.,4])
    y = 1
    # first layer with 3 hidden neurons
    W1 = np.array([[0.,0.,0.,0.],
                   [0.,0.,0.,0.],
                   [0.,0.,0.,0.]])
    b1 = np.array([0.,0.,0.])
    # second layer with 2 hidden neurons
    W2 = np.array([[0.,0.,0.],
                   [0.,0.,0.]])
    b2 = np.array([0.,0.])
    # forward pass
    a1, a2 = forward(x, W1, b1, W2, b2)
    # backward pass 
    dL_db2, dL_dW2, dL_da1 = backward_layer2(y,a1,a2,W2) 
    assert type(dL_dW2) == np.ndarray
    assert dL_dW2.shape == (2,3)
    t = [[ 0.25, 0.25, 0.25],
         [-0.25,-0.25,-0.25]]
    np.allclose(dL_dW2,t,atol=1e-3)
    assert type(dL_db2) == np.ndarray
    assert dL_db2.shape == (2,)
    t = [0.5,-0.5]
    np.allclose(dL_db2,t,atol=1e-3)
    assert type(dL_da1) == np.ndarray
    assert dL_da1.shape == (3,)
    np.allclose(dL_da1,[0,0,0],atol=1e-3)
#---------------------------------------------------
def test_backward_layer1():
    ''' (1 points) backward_layer1'''
    x = np.array([1., 2.,3.,4])
    y = 1
    # first layer with 3 hidden neurons
    W1 = np.zeros((3,4))
    b1 = np.array([0.,0.,0.])
    # second layer with 2 hidden neurons
    W2 = np.array([[0.,1.,-1.],
                   [0.,-1.,1.]])
    b2 = np.array([0,0])
    # forward pass
    a1, a2 = forward(x, W1, b1, W2, b2)
    # backward pass 
    dL_db2, dL_dW2, dL_da1 = backward_layer2(y,a1,a2,W2) 
    dL_db1, dL_dW1 = backward_layer1(x,a1,dL_da1) 
    assert type(dL_dW1) == np.ndarray
    assert dL_dW1.shape == (3,4)
    t = np.array([[ 0.  , 0. ,  0.  , 0.  ],
                  [ 0.25, 0.5,  0.75, 1.  ],
                  [-0.25,-0.5, -0.75,-1.  ]])
    np.allclose(dL_dW1,t,atol=1e-3)
    assert type(dL_db1) == np.ndarray
    assert dL_db1.shape == (3,)
    t = [0.5,-0.5]
    np.allclose(dL_db2,t,atol=1e-3)
#---------------------------------------------------
def test_backward():
    ''' (3 points) backward'''
    
    x = np.array([1., 2.,3.,4])
    y = 1
    # first layer with 3 hidden neurons
    W1 = np.zeros((3,4))
    b1 = np.array([0.,0.,0.])
    # second layer with 2 hidden neurons
    W2 = np.array([[0.,1.,-1.],
                   [0.,-1.,1.]])
    b2 = np.array([0,0])
    # forward pass
    a1, a2 = forward(x, W1, b1, W2, b2)
    # backward pass 
    dL_db2, dL_dW2, dL_db1,dL_dW1 = backward(x,y,a1,a2,W2) 
    assert type(dL_dW2) == np.ndarray
    assert dL_dW2.shape == (2,3)
    t = [[ 0.25, 0.25, 0.25],
         [-0.25,-0.25,-0.25]]
    np.allclose(dL_dW2,t,atol=1e-3)
    assert type(dL_db2) == np.ndarray
    assert dL_db2.shape == (2,)
    t = [0.5,-0.5]
    np.allclose(dL_db2,t,atol=1e-3)
    assert type(dL_dW1) == np.ndarray
    assert dL_dW1.shape == (3,4)
    t = np.array([[ 0.  , 0. ,  0.  , 0.  ],
                  [ 0.25, 0.5,  0.75, 1.  ],
                  [-0.25,-0.5, -0.75,-1.  ]])
    np.allclose(dL_dW1,t,atol=1e-3)
    assert type(dL_db1) == np.ndarray
    assert dL_db1.shape == (3,)
    t = [0.5,-0.5]
    np.allclose(dL_db2,t,atol=1e-3)
    for _ in range(20):
        p = np.random.randint(2,10) # number of features
        c = np.random.randint(2,10) # number of classes
        h = np.random.randint(2,10) # number of neurons in the 1st layer 
        x = 10*np.random.random(p)-5
        y = np.random.randint(c) 
        W1 = 2*np.random.random((h,p))-1
        b1 = np.random.random(h)
        W2 = 2*np.random.random((c,h))-1
        b2 = np.random.random(c)
        a1, a2 = forward(x, W1, b1, W2, b2)
        # analytical gradients
        dL_db2, dL_dW2, dL_db1,dL_dW1 = backward(x,y,a1,a2,W2) 
        # numerical gradients
        dL_dW2_true = check_dL_dW2(x,y, W1,b1,W2,b2)
        assert np.allclose(dL_dW2, dL_dW2_true, atol=1e-4) 
        dL_dW1_true = check_dL_dW1(x,y, W1,b1,W2,b2)
        assert np.allclose(dL_dW1, dL_dW1_true, atol=1e-4) 
#---------------------------------------------------
def test_train():
    ''' (2 points) train'''
    # an example feature matrix (4 instances, 2 features)
    Xtrain  = np.array( [[0., 1.],
                         [1., 0.],
                         [0., 0.],
                         [1., 1.]])
    Ytrain = [0, 1, 0, 1]
    # call the function
    W1, b1, W2, b2 = train(Xtrain, Ytrain,alpha=0.01, n_epoch=2000)
    # x1 is in class 0 
    a1, a2 = forward([0.,1.], W1, b1, W2, b2)
    assert a2[0]>=0.5 
    # x2 is in class 1
    a1, a2 = forward([1.,0.], W1, b1, W2, b2)
    # x3 is in class 0 
    a1, a2 = forward([0.,0.], W1, b1, W2, b2)
    assert a2[0]>=0.5 
    # x4 is in class 1
    a1, a2 = forward([1.,1.], W1, b1, W2, b2)
#---------------------------------------------------
def test_inference():
    ''' (1 points) inference'''
    W1 = np.array([[0.4, -0.1],
                  [-0.6, 0.2]])
    b1 = np.array([0.1, 0.])
    W2 = np.array([[0.4,-0.1],
                  [-0.6, 0.]])
    b2 = np.array([0.1, -0.2])
    x= np.array([1,1])
    y, a2= inference(x,W1,b1,W2,b2)
    assert y==0
    assert np.allclose(a2, [0.70235896,0.29764104], atol = 1e-2)
    x= np.array([0,1])
    y, a2= inference(x,W1,b1,W2,b2)
    assert y==0
    assert np.allclose(a2, [0.67809187,0.32190813], atol = 1e-2)
#---------------------------------------------------
def test_predict():
    ''' (4 points) predict'''
    Xtest  = np.array([[0., 1.],
                       [1., 0.],
                       [0., 0.],
                       [1., 1.]])
    W1 = np.array([[0.4, -0.1],
                  [-0.6, 0.2]])
    b1 = np.array([0.1, 0.])
    W2 = np.array([[0.4,-0.1],
                  [-0.6, 0.]])
    b2 = np.array([0.1, -0.2])
    # call the function
    Ytest, Ptest = predict(Xtest, W1, b1,W2,b2 )
    assert type(Ytest) == np.ndarray
    assert Ytest.shape == (4,)
    assert type(Ptest) == np.ndarray
    assert Ptest.shape == (4,2)
    Ytest_true = [0, 0,0,0]
    Ptest_true = [[0.67809187,0.32190813],
                  [0.70827585,0.29172415],
                  [0.68459701,0.31540299],
                  [0.70235896,0.29764104]] 
    # check the correctness of the result 
    assert np.allclose(Ytest, Ytest_true, atol = 1e-2)
    assert np.allclose(Ptest, Ptest_true, atol = 1e-2)
    n_samples = 400
    X = np.loadtxt("X3.csv",dtype=float,delimiter=",") 
    y = np.loadtxt("y3.csv",dtype=int,delimiter=",") 
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    W1,b1,W2,b2 = train(Xtrain, Ytrain,alpha=.01, n_epoch=100)
    Y, P = predict(Xtrain, W1, b1, W2, b2)
    accuracy = sum(Y == Ytrain)/(n_samples/2.)
    print("Training accuracy:", accuracy)
    assert accuracy > 0.85
    Y, P = predict(Xtest, W1, b1, W2, b2)
    accuracy = sum(Y == Ytest)/(n_samples/2.)
    print("Test accuracy:", accuracy)
    assert accuracy >= 0.8

