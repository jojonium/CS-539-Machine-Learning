import random

import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Multi-Armed Bandit Problem (15 points)
    In this problem, you will implement the epsilon-greedy method for Multi-armed bandit problem.

    A list of all variables being used in this problem is provided at the end of this file.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your Dropbox automatically copied your solution from your desktop computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other students about this homework, only discuss high-level ideas or use pseudo-code. Don't discuss about the solution at the code level. For example, two students discuss about the solution of a function (which needs 5 lines of code to solve) and they then work on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences (variable names are different). In this case, the two students violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Note: we may use the Stanford Moss system to check your code for code similarity. https://theory.stanford.edu/~aiken/moss/
      Historical Data: in one year, we ended up finding 25% of the students in that class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #*******************************************
    # CHANGE HERE: if you have read and agree with the term above, change "False" to "True".
    Read_and_Agree = True
    #*******************************************
    return Read_and_Agree

#----------------------------------------------------
'''
    Given the player's memory about the previous results in the game and the action chosen and reward received at the current time step, update the player's memory. 
    ---- Inputs: --------
        * a: the index of the action being chosen by the player, an integer scalar between 0 and c-1.
        * r: the reward received at the current time step, a float scalar.
        * Rt: (player's memory) the total rewards (i.e., sum of rewards) collected for each action, a numpy float vector of length c. Rt_1[i] represents the sum of total rewards collected on the i-th action.
        * Ct: (player's memory) the counts on how many times each action has been tried, a numpy integer vector of length c. Ct_1[i] represents the total number of samples collected on the i-th action, i.e., how many times the i-th action has been tried before".
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def update_memory(a, r, Rt, Ct):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    Rt[a] = Rt[a] + r
    Ct[a] = Ct[a] + 1
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_update_memory
        --- OR ---- 
        python3 -m nose -v test1.py:test_update_memory
        --- OR ---- 
        python -m nose -v test1.py:test_update_memory
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Explore-only) Given a multi-armed bandit game,  choose an action at the current time step using explore-only strategy. Randomly pick an action with uniform distribution: equal probability for all actions. 
    ---- Inputs: --------
        * c: the number of possible actions in a multi-armed bandit problem, an integer scalar.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def choose_action_explore(c):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    a = np.random.randint(0, c)
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_choose_action_explore
        --- OR ---- 
        python3 -m nose -v test1.py:test_choose_action_explore
        --- OR ---- 
        python -m nose -v test1.py:test_choose_action_explore
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Exploit-only) Given a multi-armed bandit game and the player's memory about the previous results,  choose an action at the current time step using exploit-only strategy: choose the action with the highest average reward. 
    ---- Inputs: --------
        * Rt: (player's memory) the total rewards (i.e., sum of rewards) collected for each action, a numpy float vector of length c. Rt_1[i] represents the sum of total rewards collected on the i-th action.
        * Ct: (player's memory) the counts on how many times each action has been tried, a numpy integer vector of length c. Ct_1[i] represents the total number of samples collected on the i-th action, i.e., how many times the i-th action has been tried before".
    ---- Outputs: --------
        * a: the index of the action being chosen by the player, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * If the count in Ct[i] for the i-th action is 0, we can assume the average reward for the i-th action is 0. For example, if the count Ct for 3 actions are [0,1,1], we can assume the average reward for the first action is 0. 
        * You could us the argmax() function in numpy to return the index of the largest value in a vector. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def choose_action_exploit(Rt, Ct):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    a = np.argmax([0 if Ct[i] == 0 else Rt[i] / Ct[i] for i in range(Rt.size)])
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_choose_action_exploit
        --- OR ---- 
        python3 -m nose -v test1.py:test_choose_action_exploit
        --- OR ---- 
        python -m nose -v test1.py:test_choose_action_exploit
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given a multi-armed bandit game and the player's memory about the previous results, choose an action at the current step of the game using epsilon-greedy method: with a small probability (epsilon) to follow explore-only method (randomly choose an action) and with a large probability (1-epsilon) to follow exploit-only method (choose the action with the highest average reward). 
    ---- Inputs: --------
        * Rt: (player's memory) the total rewards (i.e., sum of rewards) collected for each action, a numpy float vector of length c. Rt_1[i] represents the sum of total rewards collected on the i-th action.
        * Ct: (player's memory) the counts on how many times each action has been tried, a numpy integer vector of length c. Ct_1[i] represents the total number of samples collected on the i-th action, i.e., how many times the i-th action has been tried before".
        * e: (epsilon) the probability of the player to follow the exploration-only strategy. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the exploitation-only strategy.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could use the random.rand() function in numpy to sample a number randomly using uniform distribution between 0 and 1. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def choose_action(Rt, Ct, e=0.05):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    a = choose_action_explore(Ct.size) if random.random() < e else choose_action_exploit(Rt, Ct)
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_choose_action
        --- OR ---- 
        python3 -m nose -v test1.py:test_choose_action
        --- OR ---- 
        python -m nose -v test1.py:test_choose_action
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 1: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py
        --- OR ---- 
        python3 -m nose -v test1.py
        --- OR ---- 
        python -m nose -v test1.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 1 (15 points in total)--------------------- ... ok
        * (3 points) update_memory ... ok
        * (3 points) choose_action_explore ... ok
        * (3 points) choose_action_exploit ... ok
        * (6 points) choose_action ... ok
        ----------------------------------------------------------------------
        Ran 4 tests in 0.586s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* c:  the number of possible actions in a multi-armed bandit problem, an integer scalar. 
* e:  (epsilon) the probability of the player to follow the exploration-only strategy. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the exploitation-only strategy. 
* Rt:  (player's memory) the total rewards (i.e., sum of rewards) collected for each action, a numpy float vector of length c. Rt_1[i] represents the sum of total rewards collected on the i-th action. 
* Ct:  (player's memory) the counts on how many times each action has been tried, a numpy integer vector of length c. Ct_1[i] represents the total number of samples collected on the i-th action, i.e., how many times the i-th action has been tried before". 
* a:  the index of the action being chosen by the player, an integer scalar between 0 and c-1. 
* r:  the reward received at the current time step, a float scalar. 

'''
#--------------------------------------------