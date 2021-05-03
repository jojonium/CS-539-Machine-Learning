from problem1 import *
import sys
import math
from game import *
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (15 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_update_memory():
    ''' (3 points) update_memory'''
    Rt=np.array([0.,2.,3.])
    Ct=np.array([0,1,5])
    a = 2
    r = 4.
    update_memory(a,r,Rt,Ct)
    assert np.allclose(Rt,[0.,2.,7.])
    assert np.allclose(Ct,[0,1,6])
    Rt=np.array([1.,0.,2.,3.])
    Ct=np.array([3,0,1,5])
    a = 1
    r = 3.
    update_memory(a,r,Rt,Ct)
    assert np.allclose(Rt,[1.,3.,2.,3.])
    assert np.allclose(Ct,[3,1,1,5])
#---------------------------------------------------
def test_choose_action_explore():
    ''' (3 points) choose_action_explore'''
    # check with 3 possible actions
    count=np.zeros(3)
    for i in range(300):
        a= choose_action_explore(3)
        assert np.isscalar(a)
        count[a]+=1
    # check if all actions have the same probability
    assert np.allclose(count/300,np.ones(3)/3, atol =0.1)
    # check with any number of actions
    c = np.random.randint(2,10) # number of possible actions in the game 
    count=np.zeros(c)
    for i in range(100*c):
        a= choose_action_explore(c)
        count[a]+=1
    # check if all actions have the same probability
    assert np.allclose(count/100/c,np.ones(c)/c, atol =0.1)
#---------------------------------------------------
def test_choose_action_exploit():
    ''' (3 points) choose_action_exploit'''
    
    # check with 3 possible actions
    Rt=np.array([0.,2.,3.])
    Ct=np.array([0,1,5])
    a= choose_action_exploit(Rt,Ct)
    assert a == 1 
    assert np.allclose(Rt,[0.,2.,3.])
    assert np.allclose(Ct,[0,1,5])
    Rt=np.array([0.,-2.,-3.])
    Ct=np.array([0,1,5])
    a= choose_action_exploit(Rt,Ct)
    assert a == 0 
    assert np.allclose(Rt,[0.,-2.,-3.])
    assert np.allclose(Ct,[0,1.,5.])
    # check with any number of actions
    for _ in range(10):
        c = np.random.randint(2,10) # number of possible actions in the game 
        h = np.random.randint(10,20)
        Rt = np.random.randint(0,h,size=c)
        Ct = np.random.randint(0,h,size=c)
        Rt[Ct==0] = 0
        a_true = np.random.randint(c) # the index of the best action
        if Ct[a_true] == 0:
            Ct[a_true] = 1
        Rt[a_true] = h*Ct[a_true]
        # test the function
        a = choose_action_exploit(Rt,Ct)
        assert a == a_true
#---------------------------------------------------
def test_choose_action():
    ''' (6 points) choose_action'''
    # check with 3 possible actions
    Rt=np.array([0.,2.,3.])
    Ct=np.array([0,1,5])
    count = np.zeros(3)
    N =1000
    for _ in range(N):
        a= choose_action(Rt,Ct,e=0.75)
        count[a]+=1
    assert np.allclose(Rt,[0.,2.,3.])
    assert np.allclose(Ct,[0,1,5]) 
    assert np.allclose(count/N,[.25,.5,.25], atol = 0.05)
    Rt=np.array([6.,2.,3.])
    Ct=np.array([1 ,1 ,5 ])
    count = np.zeros(3)
    N =1000
    for _ in range(N):
        a= choose_action(Rt,Ct,e=0.9)
        count[a]+=1
    assert np.allclose(Rt,[6.,2.,3.])
    assert np.allclose(Ct,[1,1,5]) 
    assert np.allclose(count/N,[.4,.3,.3], atol = 0.05)
    # check with any number of actions
    N =1000
    for _ in range(3):
        c = np.random.randint(2,10) # number of possible actions in the game 
        h = np.random.randint(10,20)
        Rt = np.random.randint(0,h,size=c)
        Ct = np.random.randint(0,h,size=c)
        Rt[Ct==0] = 0
        a_true = np.random.randint(c) # the index of the best action
        if Ct[a_true] == 0:
            Ct[a_true] = 1
        Rt[a_true] = h*Ct[a_true]
        # test the function
        count = np.zeros(c)
        for _ in range(N):
            a= choose_action(Rt,Ct,e=0.7)
            count[a]+=1
        true_rate = .7*np.ones(c)/c
        true_rate[a_true]+=.3
        assert np.allclose(count/N, true_rate,atol = 0.05)
    # test the performance of the epsilon-greedy method that you have implemented on Multi-Armed Bandit game
    # define a player class of the epsilon-greedy method using the methods that you implemented
    class Epsilon_greedy:
        def __init__(self,e=0.05):
            self.e = e
        def initialize_game_statistics(self,c):
            self.Rt= np.zeros(c) 
            self.Ct= np.zeros(c) 
        def choose_action(self): # choose an action 
            return choose_action(self.Rt,self.Ct,self.e)
        def update_memory(self, a,r):
            update_memory(a,r,self.Rt, self.Ct)
    # create an agent/player 
    p = Epsilon_greedy(0.5)
    # create a multi-armed bandit game with 4 arms
    b1 = SlotMachine([.3,.7],[0,1]) # the first action has 70% chance to win R=1
    b2 = SlotMachine([.5,.5],[0,2]) # the second action has 50% chance to win R=2
    b3 = SlotMachine([.8,.2],[0,3]) # the third action has 20% chance to win R=3
    b4 = SlotMachine([.97,.02,.01],[0,10,300]) # a jackpot,  with a small chance (1%) to win a large reward R=300, and 2% chance to win a smaller reward R=10
    b = CategoricalMAB([b1,b2,b3,b4]) # put the four arms together into a Multi-Armed Bandit game
    # let the epsilon-greedy player play this game
    N = 3000
    e = b.run_games(p,N) # play 3000 games
    print('average reward per game:',e)
    assert e>0.7 # need to explore in order to escape the first action (exploit-only will achieve 0.7)
    assert e>0.77 # should be better than random action
    assert e>1 # need to explore more to win the jackpot

