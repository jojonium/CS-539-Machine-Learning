from problem3 import *
import sys
import math
import torch as th
from torch.utils.data import Dataset, DataLoader
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (20 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_zt():
    ''' (2 points) compute_zt'''
    # 2 time sequences of 3 input features at the current time step t 
    # n = 2, p = 3 
    xt = th.tensor([
                   #---------- the first time sequence in the mini-batch at time step t ------
                   [0.2,0.4,0.6],
                   #---------- the second time sequence in the mini-batch at time step t ------
                   [0.3,0.6,0.9],
                    ])
    # hidden states of 2 neurons after the previous step t-1
    # h = 2
    ht_1 = th.tensor([[ 0.5,-0.4],  # the hidden states for the first time sequence in the mini-batch
                      [-0.3, 0.6]], # the hidden states for the second time sequence in the mini-batch
                     requires_grad=True) 
    U = th.tensor([[1.,2.],
                   [3.,4.],
                   [5.,6.]],
                   requires_grad=True) 
    V = th.tensor([[1.,-2.],
                   [3.,-4.]],
                   requires_grad=True) 
    b_h = th.tensor([1.,  # bias for the first hidden state
                    -1.], # bias for the second hidden state
                    requires_grad=True)
    zt = compute_zt(xt,ht_1,U,V,b_h)
    # check if the values are correct
    assert type(zt) == th.Tensor 
    assert np.allclose(zt.size(),(2,2))
    zt_true= [[4.7, 5.2],
              [9.1, 5.6]]
    assert np.allclose(zt.data,zt_true, atol = 0.1)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in z)
    L = zt.sum()
    # back propagation
    L.backward()
    # the gradient for ht_1
    dL_dh_t_1 = [[-1., -1.],
                 [-1., -1.]] 
    dL_dU = [[0.5, 0.5],
             [1.0, 1.0],
             [1.5, 1.5]]
    dL_dV = [[0.2, 0.2],
             [0.2, 0.2]]
    assert np.allclose(ht_1.grad, dL_dh_t_1, atol= 0.1)
    assert np.allclose(U.grad, dL_dU, atol= 0.1)
    assert np.allclose(V.grad, dL_dV, atol= 0.1)
    assert np.allclose(b_h.grad, [2,2], atol= 0.1)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    p = np.random.randint(2,10) # number of input features at each time step 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    xt  = th.randn(n,p)
    U  = th.randn(p,h)
    V  = th.randn(h,h)
    b_h  = th.randn(h)
    ht_1 = th.randn(n,h)
    zt = compute_zt(xt,ht_1,U,V,b_h)
    assert np.allclose(zt.size(),(n,h))
#---------------------------------------------------
def test_compute_ht():
    ''' (2 points) compute_ht'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, h = 3 
    zt = th.tensor([
                   #---------- the hidden states for the first time sequence in the mini-batch at time step t ------
                   [0.0, 0.2, 1000.],
                   #---------- the hidden states for the second time sequence in the mini-batch at time step t ------
                   [0.5,-0.2,-1000.],
                    ], requires_grad=True)
    ht = compute_ht(zt)
    assert type(ht) == th.Tensor 
    ht_true =[[ 0.0000,  0.1974,  1.],
              [ 0.4621, -0.1974, -1.]]
    assert np.allclose(ht.data,ht_true,atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = ht.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_dz_t = [[1.0000, 0.961, 0.],
               [0.7864, 0.961, 0.]]
    assert np.allclose(zt.grad, dL_dz_t, atol= 0.01)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    zt = th.randn(n,h)
    ht = compute_ht(zt)
    assert np.allclose(ht.size(),(n,h))
#---------------------------------------------------
def test_step():
    ''' (2 points) step'''
    # 2 time sequences of 3 input features at the current time step t 
    # n = 2, p = 3 
    xt = th.tensor([
                   #---------- the first time sequence in the mini-batch at time step t ------
                   [0.2,0.4,0.6],
                   #---------- the second time sequence in the mini-batch at time step t ------
                   [0.3,0.6,0.9],
                    ])
    U = th.tensor([[ 0.1,-0.2],
                   [-0.3, 0.4],
                   [ 0.5,-0.6]],
                   requires_grad=True) 
    V = th.tensor([[0.1,-0.2],
                   [0.3,-0.4]],
                   requires_grad=True) 
    b_h = th.tensor([0.2,-0.2], requires_grad=True)
    # hidden states of 2 neurons after the previous step t-1
    # h = 2
    ht_1 = th.tensor([[ 0.5,-0.4],  # the hidden states for the first time sequence in the mini-batch
                      [-0.3, 0.6]], # the hidden states for the second time sequence in the mini-batch
                     requires_grad=True) 
    ht = step(xt,ht_1,U,V,b_h) 
    # check if the values are correct
    assert type(ht) == th.Tensor 
    assert np.allclose(ht.size(),(2,2))
    ht_true= [[ 0.3185, -0.3627],
              [ 0.5717, -0.6291]]
    assert np.allclose(ht.data,ht_true, atol = 0.1)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in ht)
    L = ht.sum()
    # back propagation
    L.backward()
    # the gradient for ht_1
    dL_dh_t_1 = [[-0.0838, -0.0778],
                 [-0.0535, -0.0397]] 
    dL_dU = [[0.3817, 0.3549],
             [0.7633, 0.7099],
             [1.1450, 1.0648]]
    dL_dV = [[0.2473, 0.2530],
             [0.0445, 0.0151]]
    dL_db_h = [1.5717, 1.4726]
    assert np.allclose(ht_1.grad, dL_dh_t_1, atol= 0.01)
    assert np.allclose(U.grad, dL_dU, atol= 0.01)
    assert np.allclose(V.grad, dL_dV, atol= 0.01)
    assert np.allclose(b_h.grad, dL_db_h, atol= 0.01)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    p = np.random.randint(2,10) # number of input features at each time step 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    xt  = th.randn(n,p)
    U  = th.randn(p,h)
    V  = th.randn(h,h)
    b_h  = th.randn(h)
    ht_1 = th.randn(n,h)
    zt = compute_zt(xt,ht_1,U,V,b_h)
    assert np.allclose(zt.size(),(n,h)) 
#---------------------------------------------------
def test_compute_z():
    ''' (2 points) compute_z'''
    # 2 time sequences in a mini-batch at the current time step t, with 3 hidden states (neurons)
    # n = 2, c = 3 
    ht = th.tensor([
                   #---------- the hidden states for the first time sequence in the mini-batch at the last time step t ------
                   [0.0, 0.2, 1.],
                   #---------- the hidden states for the second time sequence in the mini-batch at the last time step t ------
                   [0.5,-0.2,-1.],
                    ], requires_grad=True)
    W = th.tensor([1., 2., -3.], requires_grad=True)
    b = th.tensor(1., requires_grad=True)
    z = compute_z(ht,W,b)
    assert type(z) == th.Tensor 
    assert np.allclose(z.size(),(2,))
    assert np.allclose(z.data,[-1.6,  4.1],atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = z.sum()
    # back propagation
    L.backward()
    # the gradient for zt
    dL_dh_t = [[ 1.,  2., -3.],
               [ 1.,  2., -3.]]
    assert np.allclose(ht.grad, dL_dh_t, atol= 0.01)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    ht = th.randn(n,h)
    W = th.randn(h) 
    b = th.randn(1) 
    z = compute_z(ht,W,b)
    assert np.allclose(z.size(),(n,))
#---------------------------------------------------
def test_forward():
    ''' (2 points) forward'''
    
    # 2 time sequences of 3 time steps with 2 input features at each time step 
    # n = 2, l=3 p = 2
    x = th.tensor([
                     #---------- the first time sequence in the mini-batch ------
                     [
                       [1.,0.], # the first time step of the time sequence
                       [0.,1.], # the second time step of the time sequence
                       [1.,0.]  # the third time step of the time sequence
                     ],
                     #---------- the second time sequence in the mini-batch ------
                     [
                       [1.,0.], # the first time step of the time sequence
                       [1.,0.], # the second time step of the time sequence
                       [0.,1.]  # the third time step of the time sequence
                     ]
                     #------------------------------------------------------------
                   ])
    #---------------------------
    # Layer 1: Recurrent layer
    #---------------------------
    # 4 hidden states 
    # h = 4 (p=2)
    U = th.tensor([[ 2.1, 2.2, 2.3, 2.4],
                   [-1.1,-1.2,-2.3,-2.4]],
                   requires_grad=True) 
    V = th.tensor([[0.0,-1.0, 0.0,  0.0],
                   [0.0, 0.0,-1.0,  0.0],
                   [0.0, 0.0, 0.0,  1.0],
                   [0.0, 0.0, 0.0,  0.0]],
                   requires_grad=True) 
    b_h = th.tensor([-0.1,0.1,-0.1,0.1], requires_grad=True)
    # initial hidden states of 4 neurons on 2 time sequences 
    ht = th.zeros(2,4, requires_grad=True) 
    #---------------------------
    # Layer 2: Fully-connected layer
    #---------------------------
    W = th.tensor([-1., 1., -1., 1.], requires_grad=True)
    b = th.tensor(0., requires_grad=True)
    z = forward(x,ht,U,V,b_h,W,b)
    assert type(z) == th.Tensor 
    assert np.allclose(z.size(),(2,))
    assert np.allclose(z.data,[-0.0587, -0.0352], atol=1e-2)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in h)
    L = z.sum()
    # back propagation
    L.backward()
    # the gradient for the parameters
    dL_dW = [ 0.1304,  0.0279, -0.0007,  0.0078]
    dL_db = 2.
    dL_dU = [[-0.0752,  0.0067,  0.0502,  0.1800],
             [-0.3073,  0.0629, -0.0049,  0.1941]]
    dL_dV = [[-0.2416,  0.0556,  0.0563,  0.0371],
             [-0.2038,  0.0488,  0.0588, -0.0052],
             [-0.1922,  0.0467,  0.0589, -0.0166],
             [-0.2497,  0.0576,  0.0577,  0.0375]]
    dL_dbh = [-0.3825,  0.0695,  0.0453,  0.3740]
    assert np.allclose(W.grad, dL_dW, atol= 0.01)
    assert np.allclose(b.grad, dL_db, atol= 0.01)
    assert np.allclose(U.grad, dL_dU, atol= 0.01)
    assert np.allclose(V.grad, dL_dV, atol= 0.01)
    assert np.allclose(b_h.grad, dL_dbh, atol= 0.01)
    # test the function with random input sizes
    h = np.random.randint(2,10) # number of hidden states 
    l = np.random.randint(2,10) # number of time steps in a sequence 
    p = np.random.randint(2,10) # number of input features at each time step 
    n = np.random.randint(2,10) # number of sequences in a mini-batch
    x  = th.randn(n,l,p)
    ht = th.randn(n,h)
    U  = th.randn(p,h)
    V  = th.randn(h,h)
    b_h  = th.randn(h)
    W = th.randn(h) 
    b = th.randn(1) 
    z = forward(x,ht,U,V,b_h,W,b)
    assert np.allclose(z.size(),(n,))
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
    # Layer 1: Recurrent layer
    #---------------------------
    # 4 hidden states 
    # h = 4 (p=2)
    U = th.tensor([[ 2.1, 2.2, 2.3, 2.4],
                   [-1.1,-1.2,-2.3,-2.4]],
                   requires_grad=True) 
    V = th.tensor([[1.0,-1.0, 0.0,  0.0],
                   [1.0, 0.0,-1.0,  0.0],
                   [1.0, 0.0, 0.0,  1.0],
                   [1.0, 0.0, 0.0,  0.0]],
                   requires_grad=True) 
    b_h = th.tensor([-0.1,0.1,-0.1,0.1], requires_grad=True)
    #---------------------------
    # Layer 2: Fully-connected layer
    #---------------------------
    W = th.tensor([-1., 1., -1., 1.], requires_grad=True)
    b = th.tensor(0., requires_grad=True)
    # create a toy loss function: the sum of all elements in all parameters 
    L = W.sum()+ b + U.sum() + V.sum() + b_h.sum() 
    # back propagation to compute the gradients
    L.backward()
    # now the gradients for all parameters should be all-ones
    # let's try updating the parameters with gradient descent
    # create an optimizer for the parameters with learning rate = 0.1
    optimizer = th.optim.SGD([U,V,b_h,W,b], lr=0.1)
    # now perform gradient descent using SGD
    update_parameters(optimizer)
    # let's check the new values of the parameters 
    U_new = [[ 2.0,  2.1,  2.2,  2.3],
             [-1.2, -1.3, -2.4, -2.5]]
    V_new = [[ 0.9, -1.1, -0.1, -0.1],
             [ 0.9, -0.1, -1.1, -0.1],
             [ 0.9, -0.1, -0.1,  0.9],
             [ 0.9, -0.1, -0.1, -0.1]]
    b_h_new = [-0.2, 0.0, -0.2,  0.0]
    W_new = [-1.1,  0.9, -1.1,  0.9]
    assert np.allclose(U.data,U_new,atol=1e-2) 
    assert np.allclose(V.data,V_new,atol=1e-2) 
    assert np.allclose(b_h.data,b_h_new,atol=1e-2) 
    assert np.allclose(W.data,W_new,atol=1e-2) 
    assert np.allclose(b.data,-0.1,atol=1e-2) 
    assert np.allclose(U.grad,np.zeros((2,4)),atol=1e-2) 
    assert np.allclose(V.grad,np.zeros((4,4)),atol=1e-2) 
    assert np.allclose(b_h.grad,np.zeros(4),atol=1e-2) 
    assert np.allclose(W.grad,np.zeros(4),atol=1e-2) 
    assert np.allclose(b.grad,0,atol=1e-2) 
#---------------------------------------------------
def test_train():
    ''' (4 points) train'''
    # n = 4, l=3, p = 2 
    X  = [
          [ # instance 0
            [0.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.]  # time step 2
          ], 
          [ # instance 1
            [0.,0.], 
            [0.,0.], 
            [0.,1.]
          ],
          [ # instance 2
            [0.,0.], 
            [1.,0.], 
            [0.,0.]
          ],
          [ # instance 3
            [0.,1.], 
            [0.,0.], 
            [0.,0.]
          ] 
         ]
    Y = [0,0,1,1]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.Tensor(Y)
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    h=32
    n=2
    loader = th.utils.data.DataLoader(d, batch_size = n,shuffle=True)
    U,V,b_h,W,b = train(loader,p=2,h=h,n = n,n_epoch=100)
    ht = th.zeros(4,h) # initialize the hidden states as all zero
    z = forward(th.Tensor(X),ht,U,V,b_h,W,b)
    assert z[0] < z[2]
    assert z[1] < z[2]
    assert z[0] < z[3]
    assert z[1] < z[3]
#---------------------------------------------------
def test_predict():
    ''' (2 points) predict'''
    
    # n = 4, l=3, p = 2 
    X  = [
          [ # instance 0
            [0.,0.], # time step 0 
            [0.,0.], # time step 1
            [0.,0.]  # time step 2
          ], 
          [ # instance 1
            [0.,0.], 
            [0.,0.], 
            [0.,1.]
          ],
          [ # instance 2
            [0.,0.], 
            [1.,0.], 
            [0.,0.]
          ],
          [ # instance 3
            [0.,1.], 
            [0.,0.], 
            [0.,0.]
          ] 
         ]
    Y = [0,0,1,1]
    class toy(Dataset):
        def __init__(self):
            self.X  = th.Tensor(X)            
            self.Y = th.Tensor(Y)
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    h=32
    n=2
    loader = th.utils.data.DataLoader(d, batch_size = n,shuffle=True)
    U,V,b_h,W,b = train(loader,p=2,h=h,n = n,n_epoch=300)
    y_predict = predict(th.Tensor(X),U,V,b_h,W,b)
    assert np.allclose(y_predict, Y)

