from problem2 import *
import numpy as np
import sys
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.7 or above 
    assert sys.version_info[1]>=7 

#-------------------------------------------------------------------------
def test_compute_fx():
    ''' (1 point) compute_fx'''

    x = np.array([1.,2.])
    w = np.array([0.1,0.2])
    b = -0.5 
    fx = compute_fx(x,w,b)
    assert np.allclose(fx,0)

    b = -0.4
    fx = compute_fx(x,w,b)
    assert np.allclose(fx,0.1)

#-------------------------------------------------------------------------
def test_compute_gx():
    ''' (1 points) compute_gx'''

    x = np.array([1.,2.])
    w = np.array([0.1,0.2])
    b = -0.4 
    gx = compute_gx(x,w,b)
    assert gx==1

    b = -0.5
    gx = compute_gx(x,w,b)
    assert gx==1

    b = -0.6
    gx = compute_gx(x,w,b)
    assert gx==-1




#-------------------------------------------------------------------------
def test_compute_gradient():
    ''' (5 points) compute_gradient'''

    x = np.array([1.,1.])
    y = -1.
    w = np.array([1.,1.])
    b = -1.
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=1.)
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,)
    assert np.allclose(dL_dw, [2,2], atol = 1e-3) 
    assert dL_db == 1.

    x = np.array([1.,2.])
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=1.)
    assert np.allclose(dL_dw, [2,3], atol = 1e-3) 
    assert dL_db == 1.

    x = np.array([2.,1.])
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=1.)
    assert np.allclose(dL_dw, [3,2], atol = 1e-3) 
    assert dL_db == 1.

    x = np.array([1.,1.])
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, [1.5,1.5], atol = 1e-3) 
    assert dL_db == 1.

    x = np.array([2.,2.])
    y = 1.
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=1.)
    assert np.allclose(dL_dw, [1.,1.], atol = 1e-3) 
    assert dL_db == 0.

    dL_dw, dL_db = compute_gradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, [.5,.5], atol = 1e-3) 
    assert dL_db == 0.

    w = np.array([2.,1.])
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, np.array([1.,.5]), atol = 1e-3) 
    assert dL_db == 0.

    x = np.array([1.,1.])
    w = np.array([1.,1.])
    dL_dw, dL_db = compute_gradient(x,y,w,b,l=.5)
    assert np.allclose(dL_dw, [.5,.5], atol = 1e-3) 
    assert dL_db == 0.


   

#-------------------------------------------------------------------------
def test_update_w():
    ''' (5 points) update_w'''
    w = np.array([1.,1.])
    dL_dw = np.array([2.,3.])
    w_new = update_w(w,dL_dw,1.)
    assert type(w_new) == np.ndarray
    assert np.allclose(w_new, [-1,-2])

    w_new = update_w(w,dL_dw,.5)
    assert np.allclose(w_new,[0,-.5])

    w = np.array([4.,6.])
    w_new = update_w(w,dL_dw,1.)
    assert np.allclose(w_new, [2,3])


#-------------------------------------------------------------------------
def test_update_b():
    ''' (5 points) update_b'''
    b = 1. 
    dL_db = 2. 
    b_new = update_b(b,dL_db,1.)
    assert np.allclose(b_new, -1)

    b_new = update_b(b,dL_db,.5)
    assert np.allclose(b_new, 0)



#-------------------------------------------------------------------------
def test_train():
    '''(5 point) train'''
    # an example feature matrix (2 instances, 2 features)
    X  = np.array([[0., 0.],
                   [1., 1.]])
    Y = np.array([-1., 1.])
    w, b = train(X, Y, 0.01,n_epoch = 1000)
    assert np.allclose(w[0]+w[1]+ b, 1.,atol = 0.1)  # x2 is a positive support vector 
    assert np.allclose(b, -1.,atol =0.1)  # x1 is a negative support vector 

    #------------------
    # another example
    X  = np.array([[0., 1.],
                   [1., 0.],
                   [2., 0.],
                   [0., 2.]])
    Y = np.array([-1., -1., 1., 1.])
    w, b = train(X, Y, 0.01, C= 10000., n_epoch = 1000)
    assert np.allclose(w[0]+b, -1, atol = 0.1)
    assert np.allclose(w[1]+b, -1, atol = 0.1)
    assert np.allclose(w[0]+w[1]+b, 1, atol = 0.1)
 
    w, b = train(X, Y, 0.01, C= 0.01, n_epoch = 1000)
    assert np.allclose(w, [0,0], atol = 0.1)

#-------------------------------------------------------------------------
def test_predict():
    ''' (3 points) predict'''

    X = np.array([[0.,1.],
                  [1.,0.],
                  [0.,0.],
                  [1.,1.]])
    w = np.array([1.,1.])
    b = -.5 
    y = predict(X,w,b)
    assert type(y) == np.ndarray
    assert y.shape == (4,)
    assert np.allclose(y, [1,1,-1,1], atol = 1e-3) 

    b = -1.5 
    y = predict(X,w,b)
    assert np.allclose(y, [-1,-1,-1,1], atol = 1e-3) 

    w = np.array([2.,1.])
    b = -1.5 
    y = predict(X,w,b)
    assert np.allclose(y, [-1,1,-1,1], atol = 1e-3) 


#-------------------------------------------------------------------------
def test_svm():
    '''(5 point) SVM '''
    # load a binary classification dataset
    n_samples = 200
    X=np.loadtxt('X.csv',dtype=float, delimiter=',')
    y=np.loadtxt('y.csv',dtype=int, delimiter=',')
    # split the dataset into a training set and a test set
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    # train SVM 
    w,b = train(Xtrain, Ytrain, .001, C=1000., n_epoch=500)
    # training accuracy
    Y = predict(Xtrain, w, b)
    accuracy = (Y == Ytrain).sum()/(n_samples/2.)
    print('Training accuracy:', accuracy)
    assert accuracy > 0.9
    # test accuracy
    Y = predict(Xtest, w, b)
    accuracy = (Y == Ytest).sum()/(n_samples/2.)
    print('Test accuracy:', accuracy)
    assert accuracy > 0.9




