import random

import numpy as np
from problem1 import choose_action_explore
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Reinforcement Learning Problem and Q-Learning Method (20 points)
    In this problem, you will implement an AI player for the frozen lake game.  The main goal of this problem is to get familiar with reinforcement learning problem, and how to use Q learning method to find optimal policy in a game 

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    In a game step, choose an action using random policy. Randomly pick an action with uniform distribution: equal probabilities for all actions. 
    ---- Inputs: --------
        * c: the number of possible actions in the game, an integer scalar.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could re-use the choose_action_explore() function in the previous problem. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def random_policy(c):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    a = choose_action_explore(c)
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_random_policy
        --- OR ---- 
        python3 -m nose -v test2.py:test_random_policy
        --- OR ---- 
        python -m nose -v test2.py:test_random_policy
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given the current game state s and a Q function,  choose an action at the current  step using greedy policy on the Q function. Choose the action with the largest Q value for state s. 
    ---- Inputs: --------
        * s: the current state of the game, an integer scalar between 0 and n-1.
        * Q: the current Q function/table, a float matrix of shape n by c, Q[s,a] represents the Q value for (state s and action a).
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could us the argmax() function in numpy to return the index of the largest value in a vector. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def greedy_policy(s, Q):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    a = np.argmax(Q[s])
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_greedy_policy
        --- OR ---- 
        python3 -m nose -v test2.py:test_greedy_policy
        --- OR ---- 
        python -m nose -v test2.py:test_greedy_policy
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
     Given the current Q values of a game,  choose an action at the current step using epsilon-greedy method on the Q function. We have epsilon probability to follow the random policy (randomly pick an action with uniform distribution) and  (1-epsilon) probability to follow the greedy policy on Q function (pick the action according to the largest Q value for the current game state s). 
    ---- Inputs: --------
        * s: the current state of the game, an integer scalar between 0 and n-1.
        * Q: the current Q function/table, a float matrix of shape n by c, Q[s,a] represents the Q value for (state s and action a).
        * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could use the random.rand() function in numpy to sample a number randomly using uniform distribution between 0 and 1. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def choose_action_e_greedy(s, Q, e):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    a = random_policy(Q[s].size) if random.random() < e else greedy_policy(s, Q)
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_choose_action_e_greedy
        --- OR ---- 
        python3 -m nose -v test2.py:test_choose_action_e_greedy
        --- OR ---- 
        python -m nose -v test2.py:test_choose_action_e_greedy
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given the current running average (C) of sample values and a new sampled value (v_sample), please compute the updated running average (C_new) of the samples using running average method. 
    ---- Inputs: --------
        * C: the current running average of sampled values, a float scalar.
        * v_sample: a new sampled value, a float scalar.
        * lr: learning rate, a float scalar, between 0 and 1.
    ---- Outputs: --------
        * C_new: updated running average of sample values, a float scalar.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def running_average(C, v_sample, lr=0.1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    C_new = C * (1 - lr) + v_sample * lr
    #########################################
    return C_new
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_running_average
        --- OR ---- 
        python3 -m nose -v test2.py:test_running_average
        --- OR ---- 
        python -m nose -v test2.py:test_running_average
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given the player's current Q table (value function) and a sample of one step of a game episode (i.e., the current state s, the action a chosen by the player, the next game state s_new, and reward received r), update the Q table/function using Bellman Optimality Equation (with discount factor gamma) using running average method. 
    ---- Inputs: --------
        * s: the current state of the game, an integer scalar between 0 and n-1.
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
        * s_new: the next state of the game after the transition, an integer scalar between 0 and n-1.
        * r: the reward returned after the transition, a float scalar.
        * Q: the current Q function/table, a float matrix of shape n by c, Q[s,a] represents the Q value for (state s and action a).
        * gamma: the discount factor, a float scalar between 0 and 1.
        * lr: learning rate, a float scalar, between 0 and 1.
    ---- Hints: --------
        * (Step 1) compute the target Q value using Bellman Optimality Equation. 
        * (Step 2) update the element of Q table using running average method. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def update_Q(s, a, s_new, r, Q, gamma=0.95, lr=0.1):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    sample = r + gamma * np.max(Q[s_new])
    Q[s, a] = running_average(Q[s, a], sample, lr)
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_update_Q
        --- OR ---- 
        python3 -m nose -v test2.py:test_update_Q
        --- OR ---- 
        python -m nose -v test2.py:test_update_Q
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 2: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py
        --- OR ---- 
        python3 -m nose -v test2.py
        --- OR ---- 
        python -m nose -v test2.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 2 (20 points in total)--------------------- ... ok
        * (2 points) random_policy ... ok
        * (3 points) greedy_policy ... ok
        * (3 points) choose_action_e_greedy ... ok
        * (2 points) running_average ... ok
        * (10 points) update_Q ... ok
        ----------------------------------------------------------------------
        Ran 5 tests in 1.823s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* c:  the number of possible actions in the game, an integer scalar. 
* n:  the number of states in the game, an integer scalar. 
* a:  the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1. 
* s:  the current state of the game, an integer scalar between 0 and n-1. 
* s_new:  the next state of the game after the transition, an integer scalar between 0 and n-1. 
* r:  the reward returned after the transition, a float scalar. 
* gamma:  the discount factor, a float scalar between 0 and 1. 
* lr:  learning rate, a float scalar, between 0 and 1. 
* Q:  the current Q function/table, a float matrix of shape n by c, Q[s,a] represents the Q value for (state s and action a). 
* e:  (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values. 
* C:  the current running average of sampled values, a float scalar. 
* v_sample:  a new sampled value, a float scalar. 
* C_new:  updated running average of sample values, a float scalar. 

'''
#--------------------------------------------