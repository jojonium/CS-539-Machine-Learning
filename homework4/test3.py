from problem3 import *
import sys
import math
from game import *
from torch import Tensor
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (35 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_Q():
    ''' (2 points) compute_Q'''
    # 3 samples in a mini-batch, each sample is a 2 dimensional feature vector
    S = th.tensor([[1.,2.], # the current game state in the first sample of the mini-batch
                   [3.,1.], # the current game state in the second sample of the mini-batch
                   [1.,1.]])# the current game state in the second sample of the mini-batch
    # 4 possible actions
    W= th.tensor([[.8,.6,.5,.3],  # weights of the Q network (2 features x 4 actions)
                  [.1,.2,.5,.4]],
                 requires_grad=True) 
    b= th.tensor([-0.2,-0.1, 0.1,0.2], # biases of the Q network on 4 actions
                requires_grad=True) 
    Q = compute_Q(S,W,b) 
    # check value 
    assert type(Q) == Tensor 
    assert np.allclose(Q.size(),(3,4))
    assert Q.requires_grad
    Q_true  = [[0.8, 0.9, 1.6, 1.3],
               [2.3, 1.9, 2.1, 1.5],
               [0.7, 0.7, 1.1, 0.9]]
    assert np.allclose(Q.data, Q_true)
    # check if the gradients are connected correctly
    # create a simple loss function (sum of all elements in p)
    L = Q.sum()
    # back propagation
    L.backward()
    # check the gradients
    dL_dW = [[5., 5., 5., 5.],
             [4., 4., 4., 4.]]
    dL_db = [3., 3., 3., 3.]
    assert np.allclose(W.grad, dL_dW, atol= 0.1)
    assert np.allclose(b.grad, dL_db, atol= 0.1)
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    c = np.random.randint(2,4) # number of actions 
    p = np.random.randint(2,4) # number of features 
    S  = th.randn(n,p)
    W  = th.randn(p,c)
    b = th.randn(c)
    Q= compute_Q(S,W,b) 
    assert np.allclose(Q.size(),(n,c))
#---------------------------------------------------
def test_compute_Qt():
    ''' (5 points) compute_Qt'''
    # 3 samples in a mini-batch, each sample is a 2 dimensional feature vector
    S_new = th.tensor([[1.,2.], # the next game state in the first sample of the mini-batch
                       [3.,1.], # the next game state in the second sample of the mini-batch
                       [1.,1.]])# the next game state in the second sample of the mini-batch
    R = th.tensor([ 1.,2.,3.]) # immediate reward in the 3 samples
    T = th.tensor([False,False,True]) # whether S_new is a terminal state on the 3 samples
    # 4 possible actions
    W= th.tensor([[.8,.6,.5,.3],  # weights of the Q network (2 features x 4 actions)
                  [.1,.2,.5,.4]],
                 requires_grad=True) 
    b= th.tensor([-0.2,-0.1, 0.1,0.2], # biases of the Q network on 4 actions
                requires_grad=True) 
    Qt = compute_Qt(S_new,R,T,W,b,gamma=0.5) 
    # check value 
    assert type(Qt) == Tensor 
    assert np.allclose(Qt.size(),(3,))
    # check if the gradients are disconnected correctly
    assert Qt.requires_grad == False 
    Qt_true  = [1.8, 3.15, 3.]
    assert np.allclose(Qt.data, Qt_true,atol=0.01)
    #------------------------------
    # check with another gamma value
    Qt = compute_Qt(S_new, R, T, W,b, gamma=0.9) 
    # check value 
    Qt_true  = [2.44, 4.07, 3.]
    assert np.allclose(Qt.data, Qt_true,atol=0.01)
    #------------------------------
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    c = np.random.randint(2,4) # number of actions 
    p = np.random.randint(2,4) # number of features 
    S_new  = th.randn(n,p)
    R = th.randn(n)
    T = th.randn(n)>0
    W  = th.randn(p,c)
    b = th.randn(c)
    Qt= compute_Qt(S_new,R,T,W,b) 
    assert np.allclose(Qt.size(),n)
#---------------------------------------------------
def test_compute_L():
    ''' (5 points) compute_L'''
    # 2 samples in the mini-batch, 3 actions
    Q = th.tensor([[1.,2.,3.],
                   [6.,5.,4.]], requires_grad=True)  # the estimated Q values on all actions
    A = th.LongTensor([2,1]) # the actions chosen
    Qt = th.tensor([4.,7.]) # the target Q values
    L = compute_L(Q,A,Qt)
    assert type(L) == th.Tensor 
    assert np.allclose(L.data,2.5,atol=0.1)
    # check gradient
    L.backward()
    assert np.allclose(Q.grad.data,[[0,0,-1],[0,-2,0]],atol=0.1)
    # 4 samples in the mini-batch, 2 actions
    Q = th.tensor([[1.,2.],
                   [3.,4.],
                   [5.,6.],
                   [7.,8.]], requires_grad=True)  # the estimated Q values on all actions
    A = th.LongTensor([0,1,0,1]) # the actions chosen
    Qt = th.tensor([0.,2.,8.,4.]) # the target Q values
    L = compute_L(Q,A,Qt)
    assert np.allclose(L.data,7.5,atol=0.1)
    # check gradient
    L.backward()
    Q_grad_true= [[ 0.5,0.],
                  [ 0. ,1.],
                  [-1.5,0.],
                  [ 0. ,2.]]
    assert np.allclose(Q.grad.data,Q_grad_true,atol=0.1)
    #------------------------------
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    c = np.random.randint(2,4) # number of actions 
    Q = th.randn(n,c,requires_grad=True)
    A = th.randint(0,c,size=(n,))
    Qt = th.randn(n)
    L = compute_L(Q,A,Qt) 
    assert L.requires_grad 
#---------------------------------------------------
def test_update_parameters():
    ''' (2 points) update_parameters'''
    W = th.tensor([[ 1., -2., 3.],
                   [-1.,  2.,-3.]], requires_grad=True)
    b= th.tensor([0.,1.,2.],requires_grad=True) 
    # create a toy loss function: the sum of all elements in W and b
    L = W.sum()+b.sum()
    # back propagation to compute the gradients
    L.backward()
    # now the gradients for both W and b are all-ones
    # let's try updating the parameters with gradient descent
    # create an optimizer for the parameters with learning rate = 0.1
    optimizer = th.optim.SGD([W,b], lr=0.1)
    # now perform gradient descent using SGD
    update_parameters(optimizer)
    # let's check the new values of the parameters 
    W_new = [[ 0.9, -2.1,  2.9],
              [-1.1,  1.9, -3.1]]
    b_new = [-0.1,  0.9,  1.9]
    assert np.allclose(W.data,W_new,atol=1e-2) 
    assert np.allclose(b.data,b_new,atol=1e-2) 
    assert np.allclose(W.grad,np.zeros((2,3)),atol=1e-2) 
    assert np.allclose(b.grad,[0,0,0],atol=1e-2) 
#---------------------------------------------------
def test_update_Q():
    ''' (5 points) update_Q'''
    # 3 samples in a mini-batch, each sample is a 2 dimensional feature vector
    S = th.tensor([[1.,2.], # the current game state in the first sample of the mini-batch
                   [3.,1.], # the current game state in the second sample of the mini-batch
                   [1.,1.]])# the current game state in the second sample of the mini-batch
    A = th.LongTensor([0,1,2]) # the actions chosen
    # 4 possible actions
    W= th.tensor([[.8,.6,.5,.3],  # weights of the Q network (2 features x 4 actions)
                  [.1,.2,.5,.4]],
                 requires_grad=True) 
    b= th.tensor([-0.2,-0.1, 0.1,0.2], # biases of the Q network on 4 actions
                requires_grad=True) 
    # create an optimizer for the parameters with learning rate = 0.1
    optimizer = th.optim.SGD([W,b], lr=0.5)
    Qt = th.tensor([2.44, 4.07, 3.])
    # update Q
    update_Q(S,A,Qt,W,b,optimizer)
    W_true = [[1.3467, 2.7700, 1.1333, 0.3],
              [1.1933, 0.9233, 1.1333, 0.4]]
    b_true = [0.3467, 0.6233, 0.7333, 0.2]
    assert np.allclose(W.data,W_true,atol=1e-3) 
    assert np.allclose(b.data,b_true,atol=1e-3) 
    assert np.allclose(W.grad,np.zeros((2,4)),atol=1e-2) 
    assert np.allclose(b.grad,[0,0,0,0],atol=1e-2) 
    #------------------------------
    # test with another gamma and learning rate value
    W= th.tensor([[.8,.6,.5,.3],  # weights of the Q network (2 features x 4 actions)
                  [.1,.2,.5,.4]],
                 requires_grad=True) 
    b= th.tensor([-0.2,-0.1, 0.1,0.2], # biases of the Q network on 4 actions
                requires_grad=True) 
    optimizer = th.optim.SGD([W,b], lr=0.4)
    Qt = th.tensor([1.32, 2.46, 3.])
    update_Q(S,A,Qt,W,b,optimizer)
    W_true = [[0.9387, 1.0480, 1.0067, 0.3],
              [0.3773, 0.3493, 1.0067, 0.4]]
    b_true = [-0.0613, 0.0493, 0.6067, 0.2]
    assert np.allclose(W.data,W_true,atol=1e-2) 
    assert np.allclose(b.data,b_true,atol=1e-2) 
    assert np.allclose(W.grad,np.zeros((2,4)),atol=1e-2) 
    assert np.allclose(b.grad,[0,0,0,0],atol=1e-2) 
    #------------------------------
    # test the function with random input sizes
    n  = np.random.randint(2,4) # batch size 
    c = np.random.randint(2,4) # number of actions 
    p = np.random.randint(2,4) # number of features 
    S  = th.randn(n,p)
    A = th.randint(0,c,size=(n,))
    Qt  = th.randn(n)
    W  = th.randn(p,c,requires_grad=True)
    b = th.randn(c,requires_grad=True)
    optimizer = th.optim.SGD([W,b], lr=0.1)
    update_Q(S,A,Qt, W,b,optimizer)
    assert True
#---------------------------------------------------
def test_predict_q():
    ''' (3 points) predict_q'''
    # the current game state: a 2 dimensional feature vector
    s = th.tensor([1.,-1.])
    # 4 possible actions
    W= th.tensor([[.8,.6,.5,.3],  # weights of the Q network (2 features x 4 actions)
                  [.1,.2,.5,.4]],
                 requires_grad=True) 
    b= th.tensor([-0.2,-0.1, 0.1,0.2], # biases of the Q network on 4 actions
                requires_grad=True) 
    q = predict_q(s,W,b) 
    # check value 
    assert type(q) == Tensor 
    assert np.allclose(q.size(),(4,)) # 4 actions
    assert np.allclose(q.data,[.5,.3,.1,.1],atol=0.1)
    #------------------------------
    # test the function with random input sizes
    p = np.random.randint(2,4) # number of features 
    c = np.random.randint(2,4) # number of actions 
    s  = th.randn(p)
    W  = th.randn(p,c)
    b = th.randn(c)
    q= predict_q(s,W,b) 
    assert np.allclose(q.size(),(c))
#---------------------------------------------------
def test_greedy_policy():
    ''' (2 points) greedy_policy'''
    # check with 3 possible actions
    q=th.tensor([0.,1.,5.],requires_grad=True)
    a= greedy_policy(q)
    assert a == 2
    q=th.tensor([0.,2.,1.],requires_grad=True)
    a= greedy_policy(q)
    assert a == 1 
    # check with any number of actions
    for _ in range(10):
        c = np.random.randint(2,10) # number of possible actions in the game 
        h = np.random.randint(10,20) 
        q = th.rand(c)*h
        a_true = np.random.randint(c) # the index of the best action
        q[a_true] = h*1.246
        # test the function
        a = greedy_policy(q)
        assert a == a_true
#---------------------------------------------------
def test_egreedy_policy():
    ''' (3 points) egreedy_policy'''
    # check with 3 possible actions
    q=th.tensor([0.,1.,5.],requires_grad=True)
    count = np.zeros(3)
    N =1000
    for _ in range(N):
        a= egreedy_policy(q,e=0.75)
        count[a]+=1
    assert np.allclose(count/N,[.25,.25,.5], atol = 0.05)
    # check with any number of actions
    N =1000
    for _ in range(3):
        c = np.random.randint(2,10) # number of possible actions in the game 
        h = np.random.randint(10,20)
        q = th.rand(c)*h
        a_true = np.random.randint(c) # the index of the best action
        q[a_true] = h*1.522
        # test the function
        count = np.zeros(c)
        for _ in range(N):
            a= egreedy_policy(q,e=0.7)
            count[a]+=1
        true_rate = .7*np.ones(c)/c
        true_rate[a_true]+=.3
#---------------------------------------------------
def test_sample_action():
    ''' (8 points) sample_action'''
    
    # the current game state: a 2 dimensional feature vector
    s = th.tensor([1.,-1.])
    # 4 possible actions
    W= th.tensor([[.8,.6,.5,.3],  # weights of the Q network (2 features x 4 actions)
                  [.1,.2,.5,.4]],
                 requires_grad=True) 
    b= th.tensor([-0.2,-0.1, 0.1,0.2], # biases of the Q network on 4 actions
                requires_grad=True) 
    count = np.zeros(4)
    N =400
    for _ in range(N):
        a= sample_action(s,W,b,e=0.8)
        count[a]+=1
    assert np.allclose(count/N,[.4,.2,.2,.2], atol = 0.1)
    # define a player class of the Q-learning method for using the functions that you have implemented 
    class QNet:
        def __init__(self,
                          e=0.1, # epsilon: explore rate
                          gamma=0.95, # discount factor
                          lr=0.2, # learning rate
                          p= 16, # number of features  
                          c= 4, # number of actions 
                          n=10 # batch size
                    ):
            self.e = e
            self.gamma = gamma
            self.lr= lr 
            self.n=n 
            self.S = [] # memory to store a mini-batch of game-step samples
            self.A = [] # memory to store a mini-batch of game-step samples
            self.S_new = [] # memory to store a mini-batch of game-step samples
            self.R = [] # memory to store a mini-batch of game-step samples
            self.T = [] # memory to store a mini-batch of game-step samples
            self.W = th.zeros(p,c,requires_grad=True)
            self.b = th.zeros(c,requires_grad=True)
            self.optimizer =  th.optim.SGD([self.W,self.b], lr=lr)
            self.i = 0 # counter for mini-batch of samples
        def sanity_test(self):
            assert self.W[0,0]==self.W[0,0] # test if the weights are NaN (not a number)
            assert self.b[0]==self.b[0] # test if the weights are NaN (not a number)
        def choose_action(self,s):
            return sample_action(th.Tensor(s),self.W,self.b, self.e)
        def update_memory(self, s,a,s_new,r,done):
            # store the data into the mini-batch
            self.S.append(s)
            self.A.append(a)
            self.S_new.append(s_new)
            self.R.append(r)
            self.T.append(done)
            # update mini-batch counter
            self.i= (self.i + 1) % self.n 
            if self.i==0:
                Qt = compute_Qt(th.tensor(self.S_new),
                                th.tensor(self.R),
                                th.tensor(self.T),
                                self.W,
                                self.b,
                                self.gamma)
                # update Q network
                update_Q(th.tensor(self.S),
                         th.LongTensor(self.A),
                         Qt,
                         self.W,
                         self.b,
                         self.optimizer)# update Q network (gradient descent) 
                self.sanity_test()
                # reset mini-batch memory
                self.S = [] 
                self.A = [] 
                self.S_new = [] 
                self.R = [] 
                self.T = [] 
    # create a game 
    g = FrozenLake(vector_state=True)
    # create an agent (Q-learning)
    p = QNet(e=1.0) # start with random policy (epsilon = 1.) to explore the map
    # run 1000 games
    r=g.run_games(p,1000)
    assert r > 0 # winning rate of random policy (should win at least one game)
    p.e = 0.1 # now change to epsilon-greedy Q-learning
    r=g.run_games(p,1000)
    assert r > 0.85 # winning rate of Q-learning (should win at least 850/1000 games)

