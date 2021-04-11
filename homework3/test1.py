from problem1 import *
import sys
import math
import torch as th
from torch.utils.data import Dataset, DataLoader
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (12 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_z():
    ''' (2 points) compute_z'''
    # batch_size = 4
    # number of classes c = 3
    # number of input features p = 2
    # feature matrix of one mini-batch: 4 (batch_size) by 2 (p) matrix
    x = th.tensor([[1.,1.], # the first sample in the mini-batch
                   [2.,2.], # the second sample in the mini-batch
                   [3.,3.], # the third sample in the mini-batch
                   [4.,4.]])# the fourth sample in the mini-batch
    W = th.tensor([[ 0.5, 0.1,-0.2],
                   [-0.6, 0.0, 0.3]],requires_grad=True)
    b = th.tensor([0.2,-0.3,-0.1],requires_grad=True) 
    z = compute_z(x,W,b)
    assert type(z) == th.Tensor 
    assert np.allclose(z.size(), (4,3)) # batch_size x c 
    z_true = [[ 0.1,-0.2, 0.0], # linear logits for the first sample in the mini-batch
              [ 0.0,-0.1, 0.1], # linear logits for the second sample in the mini-batch
              [-0.1, 0.0, 0.2], # linear logits for the third sample in the mini-batch
              [-0.2, 0.1, 0.3]] # linear logits for the fourth sample in the mini-batch
    assert np.allclose(z.data,z_true, atol = 1e-2)
    assert z.requires_grad
    # check if the gradients of W is connected to z correctly
    L = th.sum(z) # compute the sum of all elements in z
    L.backward() # back propagate gradient to W and b
    # now the gradients dL_dW should be
    dL_dW_true = [[10,10,10],
                  [10,10,10]]
    # here [10,10] in each column of dL_dW is computed as the sum of the gradients in all the four samples.
    # for the 1st sample, the gradient is x = [1,1]
    # for the 2nd sample, the gradient is x = [2,2]
    # for the 3rd sample, the gradient is x = [3,3]
    # for the 4th sample, the gradient is x = [4,4]
    # so the sum of the gradients  will be [10,10] for each class (column of dL_dW matrix)
    assert np.allclose(W.grad,dL_dW_true, atol=0.1)
    # now the gradients of dL_db should be
    dL_db_true = [4,4,4]
    # here each element (4) of dL_db is computed as the sum of the gradients in all the four samples: 1+1+1+1 = 4
    assert np.allclose(b.grad,dL_db_true, atol=0.1)
    n = np.random.randint(2,5) # batch_size 
    c = np.random.randint(2,5) # the number of classes 
    p = np.random.randint(2,5) # the number of input features 
    x  = th.randn(n,p)
    W  = th.randn(p,c)
    b = th.randn(c)
    z = compute_z(x,W,b) 
    assert np.allclose(z.size(),(n,c))
#---------------------------------------------------
def test_compute_L():
    ''' (2 points) compute_L'''
    # batch_size = 4
    # number of classes c = 3
    # linear logits in a mini-batch:  shape (4 x 3) or (batch_size x c)
    z = th.tensor([[ 0.1,-0.2, 0.0], # linear logits for the first sample in the mini-batch
                   [ 0.0,-0.1, 0.1], # linear logits for the second sample in the mini-batch
                   [-0.1, 0.0, 0.2], # linear logits for the third sample in the mini-batch
                   [-0.2, 0.1, 0.3]], requires_grad=True) # linear logits for the last sample in the mini-batch
    # the labels of the mini-batch: vector of length 4 (batch_size)
    y = th.LongTensor([1,2,1,0])
    L = compute_L(z,y)
    assert type(L) == th.Tensor 
    assert L.requires_grad
    assert np.allclose(L.detach().numpy(),1.2002,atol=1e-4) 
    # check if the gradients of z is connected to L correctly
    L.backward() # back propagate gradient to W and b
    dL_dz_true = [[ 0.0945, -0.1800,  0.0855],
                  [ 0.0831,  0.0752, -0.1582],
                  [ 0.0724, -0.1700,  0.0977],
                  [-0.1875,  0.0844,  0.1031]]
    assert np.allclose(z.grad,dL_dz_true, atol=0.01)
    #-----------------------------------------    
    # batch_size = 3
    # number of classes c = 3
    # linear logits in a mini-batch:  shape (3 x 3) or (batch_size x c)
    z = th.tensor([[  0.1,-1000, 1000], # linear logits for the first sample in the mini-batch
                   [  0.0, 1100, 1000], # linear logits for the second sample in the mini-batch
                   [-2000,-1900,-5000]], requires_grad=True) # linear logits for the last sample in the mini-batch
    y = th.LongTensor([2,1,1])
    L = compute_L(z,y)
    assert np.allclose(L.data,0,atol=1e-4) 
    #-----------------------------------------    
    # batch_size = 2
    # number of classes c = 3
    # linear logits in a mini-batch:  shape (2 x 3) or (batch_size x c)
    z = th.tensor([[  0.1,-1000, 1000], # linear logits for the first sample in the mini-batch
                   [-2000,-1900,-5000]], requires_grad=True) # linear logits for the last sample in the mini-batch
    y = th.LongTensor([0,2])
    L = compute_L(z,y)
    assert L.data >100
    assert L.data < float('inf')
    # test the function with random input sizes
    n = np.random.randint(2,5) # batch_size 
    c = np.random.randint(2,5) # the number of classes 
    y = th.randint(0,c,(n,)) # the number of classes 
    z  = th.rand(n,c)
    L = compute_L(z,y) 
    assert np.allclose(L.size(),n)
#---------------------------------------------------
def test_update_parameters():
    ''' (2 points) update_parameters'''
    # weight matrix of shape (2 x 3) or (p x c)
    W = th.tensor([[ 0.5, 0.1,-0.2],
                   [-0.6, 0.0, 0.3]],requires_grad=True)
    # bias vector of length 3 (c)
    b = th.tensor([0.2,-0.3,-0.1],requires_grad=True) 
    # create a toy loss function: the sum of all elements in W and b
    L = W.sum()+b.sum()
    # back propagation to compute the gradients
    L.backward()
    # now the gradients for both W and b are all-ones: the global gradient of every element in W or b is 1
    # let's try updating W and b with stochastic gradient descent
    # create an optimizer for W and b with learning rate = 0.1
    optimizer = th.optim.SGD([W,b], lr=0.1)
    # now perform gradient descent using SGD
    update_parameters(optimizer)
    # let's check the new values of the W and b
    W_new_true = [[ 0.4,  0.0, -0.3],
                  [-0.7, -0.1,  0.2]]
    b_new_true = [0.1, -0.4, -0.2]
    assert np.allclose(W.data,W_new_true,atol=1e-2) 
    assert np.allclose(b.data,b_new_true,atol=1e-2) 
    assert np.allclose(W.grad,np.zeros((2,3)),atol=1e-2) 
    assert np.allclose(b.grad,np.zeros(3),atol=1e-2) 
    # now let's try another optimizer (ADAM)
    optimizer = th.optim.Adam([W, b], lr=1.,betas=(0.5, 0.5))
    L1 = W.sum()+b.sum()
    L1.backward()
    # now perform gradient descent using ADAM for one step
    update_parameters(optimizer)
    # let's construct the gradients as the opposite direction of the previous step (to see the effect of momentum in the optimizer)
    L2 = -W.sum()-b.sum()
    L2.backward()
    # now perform gradient descent using ADAM for another step
    update_parameters(optimizer)
    # let's check the new values of the W and b
    W_new_true = [[-0.2667, -0.6667, -0.9667],
                  [-1.3667, -0.7667, -0.4667]]
    b_new_true = [-0.5667, -1.0667, -0.8667]
    assert np.allclose(W.data,W_new_true,atol=1e-2) 
    assert np.allclose(b.data,b_new_true,atol=1e-2) 
    assert np.allclose(W.grad,np.zeros((2,3)),atol=1e-2) 
    assert np.allclose(b.grad,np.zeros(3),atol=1e-2) 
#---------------------------------------------------
def test_train():
    ''' (2 points) train'''
    # create a toy dataset 
    # p = 2, c = 2, batch_size = 2
    class toy1(Dataset):
        def __init__(self):
            self.X  = th.tensor([[0., 0.], 
                                 [1., 1.]])
            self.Y = th.LongTensor([0, 1])
        def __len__(self):
            return 2 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    # create a toy dataset
    d = toy1()
    # create a dataset loader
    data_loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
    # train the model
    W, b = train(data_loader, c=2, p=2, alpha=0.1,n_epoch = 100)
    # test the data
    x=th.tensor([[0., 0.], [1., 1.]])
    z = compute_z(x,W,b)
    assert z[0,0]>z[0,1] # the class label for the first sample should be 0
    assert z[1,0]<z[1,1] # the class label for the second sample should be 1
    # create another toy dataset 
    # p = 2, c = 4, batch_size = 2
    class toy2(Dataset):
        def __init__(self):
            self.X  = th.tensor([[0., 0.],
                                 [1., 0.],
                                 [0., 1.],
                                 [1., 1.]])            
            self.Y = th.LongTensor([0, 1, 2, 3])
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy2()
    data_loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
    W, b = train(data_loader, c=4, p=2, alpha=0.1,n_epoch = 100)
    # test the data
    x=th.tensor([[0., 0.],
                 [1., 0.],
                 [0., 1.],
                 [1., 1.]])
    z = compute_z(x,W,b)
    assert th.max(z[0])==z[0,0] # the class label for the first sample should be 0
    assert th.max(z[1])==z[1,1] # the class label for the second sample should be 1
    assert th.max(z[2])==z[2,2] # the class label for the third sample should be 2
    assert th.max(z[3])==z[3,3] # the class label for the fourth sample should be 3
#---------------------------------------------------
def test_predict():
    ''' (4 points) predict'''
    
    # create a toy dataset for testing mini-batch training
    class toy(Dataset):
        def __init__(self):
            self.X  = th.tensor([[1., 1.],
                                 [1., 2.],
                                 [1., 3.],
                                 [1., 4.]])            
            self.Y = th.LongTensor([0, 1, 2, 0])
        def __len__(self):
            return 4 
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    d = toy()
    loader = DataLoader(d, batch_size=4, shuffle=False, num_workers=0)
    # weight matrix of shape (2 x 3) or (p x c)
    W = th.tensor([[ 0.2, 0.2, 0.2],
                   [-0.5,-0.1, 0.1]],requires_grad=True)
    # bias vector of length 3 (c)
    b = th.tensor([1.0, 0.4,-0.4],requires_grad=True) 
    for x, y in loader:
        y_predict = predict(x,W,b)
        assert np.allclose(y_predict, [0,1,1,2])
    # create a multi-class classification dataset
    n_samples = 400
    X = np.loadtxt('X1.csv',dtype=float,delimiter=',') 
    y = np.loadtxt('y1.csv',dtype=int,delimiter=',') 
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    class data_train(Dataset):
        def __init__(self):
            self.X  = th.Tensor(Xtrain)            
            self.Y = th.LongTensor(Ytrain)
        def __len__(self):
            return int(n_samples/2)
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    class data_test(Dataset):
        def __init__(self):
            self.X  = th.Tensor(Xtest)            
            self.Y = th.LongTensor(Ytest)
        def __len__(self):
            return int(n_samples/2)
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    dtr = data_train()
    loader_train = DataLoader(dtr, batch_size=10, shuffle=True, num_workers=0)
    dte = data_test()
    loader_test = DataLoader(dte, batch_size=200, shuffle=False, num_workers=0)
    W,b = train(loader_train, c=3, p=5, alpha=.01, n_epoch=100)
    for x, y in loader_test:
        y_predict = predict(x, W, b)
        accuracy = th.sum(y == y_predict)/(n_samples/2.)
        print('Test accuracy:', accuracy.data)
        assert accuracy > 0.9

