from problem2 import *
import sys
import math
from game import *
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (20 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_random_policy():
    ''' (2 points) random_policy'''
    # check with 3 possible actions
    count=np.zeros(3)
    for i in range(300):
        a= random_policy(3)
        count[a]+=1
    # check if all actions have the same probability
    assert np.allclose(count/300,np.ones(3)/3, atol =0.1)
    # check with any number of actions
    c = np.random.randint(2,10) # number of possible actions in the game 
    count=np.zeros(c)
    for i in range(100*c):
        a= random_policy(c)
        count[a]+=1
    # check if all actions have the same probability
    assert np.allclose(count/100/c,np.ones(c)/c, atol =0.1)
#---------------------------------------------------
def test_greedy_policy():
    ''' (3 points) greedy_policy'''
    # check with 3 possible actions
    Q=np.array([[ 0,1,5],
                [-1,2,1]])
    a= greedy_policy(0,Q)
    assert a == 2
    a= greedy_policy(1,Q)
    assert a == 1 
    assert np.allclose(Q,[[0,1,5],[-1,2,1]]) 
    # check with any number of actions
    for _ in range(10):
        c = np.random.randint(2,10) # number of possible actions in the game 
        n = np.random.randint(2,10) # number of states in the game
        h = np.random.randint(10,20) 
        Q = np.random.randint(0,h,size=(n,c))
        s = np.random.randint(0,n) # the current state  
        a_true = np.random.randint(c) # the index of the best action
        Q[s,a_true] = h*1.246
        # test the function
        a = greedy_policy(s,Q)
        assert a == a_true
#---------------------------------------------------
def test_choose_action_e_greedy():
    ''' (3 points) choose_action_e_greedy'''
    # check with 3 possible actions
    Q=np.array([[ 0,1,5],
                [-1,2,1]])
    count = np.zeros((2,3))
    N =1000
    for _ in range(N):
        a= choose_action_e_greedy(0,Q,e=0.75)
        count[0,a]+=1
        a= choose_action_e_greedy(1,Q,e=0.75)
        count[1,a]+=1
    assert np.allclose(Q,[[0,1,5],[-1,2,1]]) 
    assert np.allclose(count/N,[[.25,.25,.5],
                                [0.25,.5,.25]], atol = 0.05)
    # check with any number of actions
    N =1000
    for _ in range(3):
        c = np.random.randint(2,10) # number of possible actions in the game 
        n = np.random.randint(2,10) # number of states in the game
        h = np.random.randint(10,20)
        Q = np.random.randint(0,h,size=(n,c))
        s = np.random.randint(0,n) # the current state  
        a_true = np.random.randint(c) # the index of the best action
        Q[s,a_true] = h*1.522
        # test the function
        count = np.zeros(c)
        for _ in range(N):
            a= choose_action_e_greedy(s,Q,e=0.7)
            count[a]+=1
        true_rate = .7*np.ones(c)/c
        true_rate[a_true]+=.3
        assert np.allclose(count/N, true_rate,atol = 0.05)
#---------------------------------------------------
def test_running_average():
    ''' (2 points) running_average'''
    C = 0.
    lr = 0.5
    C = running_average(C,1.,lr)
    assert C == .5
    C = running_average(C,1.,lr)
    assert C == .75
    C = running_average(C,1.,lr)
    assert C == .875
    C = running_average(C,1.,lr)
    assert C == .9375
    C = 0.
    lr = 0.2
    C = running_average(C,1.,lr)
    assert C == .2
    C = running_average(C,2.,lr)
    assert C == .56
    C = running_average(C,3.,lr)
    assert C == 1.048
    C = running_average(C,4.,lr)
    assert C == 1.6384
#---------------------------------------------------
def test_update_Q():
    ''' (10 points) update_Q'''
    # 3 actions, 2 states
    Q= np.array([[1.,2.,3.],
                 [6.,5.,4.]]) 
    update_Q(s=0,s_new=1,a=0,r=1.,Q=Q,gamma = .9, lr=0.2)
    Q_true=[[2.08,2.,3.],
            [6.  ,5.,4.]] 
    assert np.allclose(Q,Q_true,atol=0.01)
    Q= np.array([[1.,2.,3.],
                 [6.,5.,4.]]) 
    update_Q(s=1,s_new=1,a=1,r=2.,Q=Q,gamma = .7, lr=0.1)
    Q_true=[[1.,2.  ,3.],
            [6.,5.12,4.]] 
    assert np.allclose(Q,Q_true,0.01)
    Q= np.array([[1.,2.,3.],
                 [6.,5.,4.]]) 
    update_Q(s=1,s_new=0,a=2,r=4.,Q=Q,gamma = .6, lr=0.3)
    Q_true=[[1.,2.,3.  ],
            [6.,5.,4.54]] 
    assert np.allclose(Q,Q_true,0.01)
    # test the performance of the Q-learning method that you have implemented on the frozen lake game
    # define a player class of the Q-learning method for using the functions that you have implemented 
    class QLearner:
        def __init__(self,e=0.1,gamma=0.99,lr=0.01):
            self.e = e
            self.gamma = gamma
            self.lr= lr 
            self.Q= np.zeros((16,4)) 
        def choose_action(self,s):
            return choose_action_e_greedy(s,self.Q,self.e) # choose an action
        def update_memory(self, s,a,s_new,r,done):
            update_Q(s,a,s_new,r,self.Q,self.gamma,self.lr)# update Q value
    # create a game 
    g = FrozenLake()
    # create an agent (Q-learning)
    p = QLearner(e=1.0) # start with random policy (epsilon = 1.) to explore the map
    # run 1000 games
    r=g.run_games(p,1000)
    assert r > 0 # win at least one game using random policy
    assert np.allclose(p.Q[5],np.zeros(4))
    assert np.allclose(p.Q[7],np.zeros(4))
    assert np.allclose(p.Q[11],np.zeros(4))
    assert np.allclose(p.Q[12],np.zeros(4))
    assert ((p.Q[-2]-p.Q[0])>0).all()
    assert ((p.Q[-2]-p.Q[6])>0).all()
    p.e = 0.1 # now change to epsilon-greedy Q-learning
    r=g.run_games(p,1000)
    assert r > 0.85 # winning rate of Q-learning (should win at least 850/1000 games)

