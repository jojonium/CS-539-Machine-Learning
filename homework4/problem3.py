import numpy as np
import torch as th
from problem2 import random_policy
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: Q Network (35 points)
    In this problem, you will implement a neural network (with one fully-connected layer only) to estimate Q values in a game  

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Training: estimate Q values using Q network) Given a Q network with parameters (W, b) and we have a mini-batch of sampled game states S. Please compute the predicted Q values on the mini-batch of samples. 
    ---- Inputs: --------
        * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,p), where S[i] is the current game state in the i-th sample in the mini-batch.
        * W: the weights of fully connected layer of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (p,c).
        * b: the biases of fully connected layer of Q network, a float torch vector of length c.
    ---- Outputs: --------
        * Q: the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_Q(S, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    Q = S@W + b
    #########################################
    return Q
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_Q
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_Q
        --- OR ---- 
        python -m nose -v test3.py:test_compute_Q
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: compute Target Q values using  Bellman Optimality Equation) Suppose we have a mini-batch of training samples: including the new/next games states S_new and immediate rewards R in the sampled game steps in the mini-batch. Please compute the target Q values (Qt) for the mini-batch of samples using Bellman Optimality Equation. Note the gradients cannot flow through Qt, i.e., the gradients of Qt tensor should not connect with the parameters W and b. 
    ---- Inputs: --------
        * S_new: the new/next game states for a mini-batch of sampled games steps after state transition, a torch tensor of shape (n,p). S_new[i] is the next/new game state in the i-th sample of the mini-batch.
        * R: a mini-batch of the immediate rewards returned after the transition, a float vector of length (n). R[i] is the received immediate reward of the i-th sampled game step in the mini-batch.
        * T: whether or not the new/next game state is a terminal state in a mini-batch of sampled games steps, a boolean torch tensor of length n. T[i]= True if S_new[i] is a terminal state in the game (where the game ends).
        * W: the weights of fully connected layer of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (p,c).
        * b: the biases of fully connected layer of Q network, a float torch vector of length c.
        * gamma: the discount factor, a float scalar between 0 and 1.
    ---- Outputs: --------
        * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch.
    ---- Hints: --------
        * (Step 1) compute Q values on the new/next game states. 
        * (Step 2.1) If S_new[i] is a terminal state (i.e., T[i] = True), use the immediate reward R[i] as the target reward. 
        * (Step 2.2) Otherwise, use Bellman Optimality Equation to estimate the target Q value. 
        * You could re-use compute_Q() function. 
        * To detach the gradients of a torch tensor x, you could use x.detach(), so that gradient will not flow through x. 
        * To negate the boolean values in a tensor x, you could use ~x. 
        * To convert a boolean-valued tensor x into an integer tensor, you could use x.int(). 
        * To compute the max value of a tensor, you could use th.max() function. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_Qt(S_new, R, T, W, b, gamma=0.95):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    new_Q = compute_Q(S_new, W, b)
    #########################################
    return Qt
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_Qt
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_Qt
        --- OR ---- 
        python -m nose -v test3.py:test_compute_Qt
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: Loss function) Given estimated Q values by the Q network, the action chosen and the target Q values on a mini-batch of sampled game steps, please compute the mean-squared-error loss on the mini-batch of samples. 
    ---- Inputs: --------
        * Q: the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch.
        * A: a mini-batch of the actions chosen by the player, an integer vector of length (n).
        * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch.
    ---- Outputs: --------
        * L: the average of the least square losses on a mini-batch of training images, a torch float scalar.
    ---- Hints: --------
        * You could use arange(n) function in Pytorch to create an index list of [0,1,2,...,n-1]. 
        * You could use y = X[list1,list2] to select elements of matrix X into a vector. For example if list1=[1,3,5], list2=[2,4,6], then y will be a list of [  X[1,2], X[3,4], X[5,6] ]. 
        * You could use MSELoss in Pytorch to compute the mean squared error. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_L(Q, A, Qt):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)

    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_L
        --- OR ---- 
        python -m nose -v test3.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: Gradient Descent) Suppose we are given a Q neural network with parameters (W, b) and we have a mini-batch of training samples (S,A,S_new,R).  Suppose we have already computed the global gradients of the loss L w.r.t. the weights W and biases b on the mini-batch of samples. Assume that we have already created an optimizer for the parameter W and b. Please update the weights W and biases b using gradient descent. After the update, the global gradients of W and b should be set to all zeros. 
    ---- Inputs: --------
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W and b).
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def update_parameters(optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    optimizer.step()
    optimizer.zero_grad()
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_update_parameters
        --- OR ---- 
        python3 -m nose -v test3.py:test_update_parameters
        --- OR ---- 
        python -m nose -v test3.py:test_update_parameters
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: Train Q Network on a mini-batch of samples) Given a mini-batch of training samples: S (current game states), A (actions chosen), S_new (new/next game states) and R (immediate rewards), suppose the target Q values are already computed (Qt), please train the Q network using gradient descent: update the weights W and biases b using the gradients on the mini-batch of data samples. 
    ---- Inputs: --------
        * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,p), where S[i] is the current game state in the i-th sample in the mini-batch.
        * A: a mini-batch of the actions chosen by the player, an integer vector of length (n).
        * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch.
        * W: the weights of fully connected layer of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (p,c).
        * b: the biases of fully connected layer of Q network, a float torch vector of length c.
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W and b).
    ---- Hints: --------
        * Step 1 Forward pass: compute estimated Q values, target Q values and the loss L. 
        * Step 2 Back propagation: compute the gradients of W and b. 
        * Step 3 Gradient descent: update the parameters W and b using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def update_Q(S, A, Qt, W, b, optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_update_Q
        --- OR ---- 
        python3 -m nose -v test3.py:test_update_Q
        --- OR ---- 
        python -m nose -v test3.py:test_update_Q
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Sampling: using Q network for playing the game) Given the Q network with parameters W and b and we have only the current states s in the game. Please compute the estimated Q values on the current game state. 
    ---- Inputs: --------
        * s: the current state of the game, a torch vector of length p.
        * W: the weights of fully connected layer of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (p,c).
        * b: the biases of fully connected layer of Q network, a float torch vector of length c.
    ---- Outputs: --------
        * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action.
    ---- Hints: --------
        * You could re-use the compute_Q() function above by creating a mini-batch of only one sample. 
        * To add a dimension to a torch tensor, you could use unsqueeze() function in torch tensor. 
        * To delete a dimension to a torch tensor, you could use squeeze() function in torch tensor. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict_q(s, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    
    #########################################
    return q
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_predict_q
        --- OR ---- 
        python3 -m nose -v test3.py:test_predict_q
        --- OR ---- 
        python -m nose -v test3.py:test_predict_q
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Sampling: Policy 1: greedy on Q) Given the Q values estimated by the Q network on the current game state s, choose an action using greedy policy on the Q values. Choose the action with the largest Q value for state s. 
    ---- Inputs: --------
        * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could us the argmax() function in torch to return the index of the largest value in a vector. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def greedy_policy(q):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    a = np.argmax(q.detach())
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_greedy_policy
        --- OR ---- 
        python3 -m nose -v test3.py:test_greedy_policy
        --- OR ---- 
        python -m nose -v test3.py:test_greedy_policy
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Sampling: Policy 2: epsilon-greedy on Q) Given the Q values estimated by the Q network on the current game state s, choose an action using epsilon-greedy policy on the Q values. 
    ---- Inputs: --------
        * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action.
        * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could re-use the random_policy() implemented in problem 2. 
        * You could use the random.rand() function in numpy to sample a number randomly using uniform distribution between 0 and 1. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def egreedy_policy(q, e):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    a = random_policy(q.shape[0]) if np.random.random() < e else np.argmax(q.detach())
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_egreedy_policy
        --- OR ---- 
        python3 -m nose -v test3.py:test_egreedy_policy
        --- OR ---- 
        python -m nose -v test3.py:test_egreedy_policy
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Sampling: Sample an action) Given the current game state s, sample an action using epsilon-greedy method on the Q values estimated by the Q network. We have epsilon probability to follow the random policy (randomly pick an action with uniform distribution) and  (1-epsilon) probability to follow the greedy policy on Q values (pick the action according to the largest Q value for the current game state s). 
    ---- Inputs: --------
        * s: the current state of the game, a torch vector of length p.
        * W: the weights of fully connected layer of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (p,c).
        * b: the biases of fully connected layer of Q network, a float torch vector of length c.
        * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * (Step 1) use the Q network to predict the Q values for the current game state. 
        * (Step 2) use epsilon-greedy policy on the Q values to sample an action. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def sample_action(s, W, b, e):
    #########################################
    ## INSERT YOUR CODE HERE (8 points)
    
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_sample_action
        --- OR ---- 
        python3 -m nose -v test3.py:test_sample_action
        --- OR ---- 
        python -m nose -v test3.py:test_sample_action
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 3: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py
        --- OR ---- 
        python3 -m nose -v test3.py
        --- OR ---- 
        python -m nose -v test3.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 3 (35 points in total)--------------------- ... ok
        * (2 points) compute_Q ... ok
        * (5 points) compute_Qt ... ok
        * (5 points) compute_L ... ok
        * (2 points) update_parameters ... ok
        * (5 points) update_Q ... ok
        * (3 points) predict_q ... ok
        * (2 points) greedy_policy ... ok
        * (3 points) egreedy_policy ... ok
        * (8 points) sample_action ... ok
        ----------------------------------------------------------------------
        Ran 9 tests in 4.785s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of game-step samples in a mini-batch, an integer scalar. 
* p:  the number of features in a game state, an integer scalar. 
* c:  the number of possible actions in the game, an integer scalar. 
* W:  the weights of fully connected layer of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (p,c). 
* b:  the biases of fully connected layer of Q network, a float torch vector of length c. 
* L:  the average of the least square losses on a mini-batch of training images, a torch float scalar. 
* lr:  learning rate for gradient descent, a float scalar, between 0 and 1. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W and b). 
* Q:  the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch. 
* Q_new:  the Q values (estimated by the target Q network) on the new game states for a mini-batch of sampled game steps, a pytorch matrix of shape (n, c). Q_new[i,j] represents the Q value on the j-th action for the new game state in the i-th sample of the mini-batch. 
* Qt:  the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch. 
* q:  the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action. 
* a:  the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1. 
* gamma:  the discount factor, a float scalar between 0 and 1. 
* e:  (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values. 
* S:  the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,p), where S[i] is the current game state in the i-th sample in the mini-batch. 
* S_new:  the new/next game states for a mini-batch of sampled games steps after state transition, a torch tensor of shape (n,p). S_new[i] is the next/new game state in the i-th sample of the mini-batch. 
* R:  a mini-batch of the immediate rewards returned after the transition, a float vector of length (n). R[i] is the received immediate reward of the i-th sampled game step in the mini-batch. 
* A:  a mini-batch of the actions chosen by the player, an integer vector of length (n). 
* T:  whether or not the new/next game state is a terminal state in a mini-batch of sampled games steps, a boolean torch tensor of length n. T[i]= True if S_new[i] is a terminal state in the game (where the game ends). 
* s:  the current state of the game, a torch vector of length p. 

'''
#--------------------------------------------