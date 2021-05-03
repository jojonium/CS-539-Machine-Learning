import torch as th
from problem3 import egreedy_policy, compute_L, update_parameters
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 4: Deep Q-Learning (30 points)
    In this problem, you will implement Deep Q Networks (DQN). We will use a convolutional layer and a fully-connected layer to approximate the Q values. Each (game state) is represented as an image. The outputs of the neural network are the Q values for different actions at the current game state. The structure of the DQN is ( CONV layer -> Fully-connected layer )

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Q network, Convolutional Layer: Linear Logits) Suppose we have a mini-batch of n game step samples,s1, s2, ... s_n, where s_i represents the game state of the i-th sample in the mini-batch. Each game state s is an image with c0 input/color channels. Let's compute the convolutional layer (1st layer) of the Q network. Here we have c1 filters in the convolutional layer (with weights W1 and biases b1), please compute the 2D convolution on the mini-batch of game state images. Here we assume that stride=1 and padding = 0. 
    ---- Inputs: --------
        * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,c0,h,w).
        * W1: the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1).
        * b1: the biases of filters in the convolutional layer of Q network, a float torch vector of length c1.
    ---- Outputs: --------
        * z1: the linear logits of the convolutional layer of Q network on a mini-batch of samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1).
    ---- Hints: --------
        * You could use the conv2d() function in PyTorch.. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z1(S, W1, b1):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    
    #########################################
    return z1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_z1
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z1
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Q network, Convolutional Layer: ReLU activation) Given the linear logits (z1) of the first layer, please compute the ReLU activations. 
    ---- Inputs: --------
        * z1: the linear logits of the convolutional layer of Q network on a mini-batch of samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1).
    ---- Outputs: --------
        * a1: the ReLU activations of the convolutional layer of Q network on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_a1(z1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return a1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_a1
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_a1
        --- OR ---- 
        python -m nose -v test4.py:test_compute_a1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Q network, Convolutional Layer:  Maxpooling) Given the activations (a1) of convolutional layer, please compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2. 
    ---- Inputs: --------
        * a1: the ReLU activations of the convolutional layer of Q network on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1).
    ---- Outputs: --------
        * p: the pooled activations (using max pooling) of the convolutional layer of the Q network on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_p(a1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return p
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_p
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_p
        --- OR ---- 
        python -m nose -v test4.py:test_compute_p
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Q network, flatten) Given the pooling results (p) of the convolutional layer of shape n x c2 x h2 x w2, please flatten the pooling results into a vector, so that it can be used as the input to the fully-connected layer. The flattened features will be a 2D matrix of shape (n x n_flat_features), where n_flat_features is computed as c2 x h2 x w2. 
    ---- Inputs: --------
        * p: the pooled activations (using max pooling) of the convolutional layer of the Q network on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1).
    ---- Outputs: --------
        * f: the input features to the fully connected layer after flattening the outputs of the convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features ).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def flatten(p):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return f
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_flatten
        --- OR ---- 
        python3 -m nose -v test4.py:test_flatten
        --- OR ---- 
        python -m nose -v test4.py:test_flatten
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Q network, Fully Connected Layer) Given flattened features on a mini-batch of sample game states, please compute the linear logits z3 (also called Q) of the fully-connected layer (which are the predicted Q values for different actions) on the mini-batch of samples. 
    ---- Inputs: --------
        * f: the input features to the fully connected layer after flattening the outputs of the convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features ).
        * W2: the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c).
        * b2: the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c.
    ---- Outputs: --------
        * Q: the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z2(f, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return Q
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_z2
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z2
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Q network, compute predicted Q values for training) Given a convolutional Q network with parameters W1, b1, W2, and b2 and we have a mini-batch of sample game states S. Please compute the predicted Q values on the mini-batch of samples. 
    ---- Inputs: --------
        * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,c0,h,w).
        * W1: the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1).
        * b1: the biases of filters in the convolutional layer of Q network, a float torch vector of length c1.
        * W2: the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c).
        * b2: the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c.
    ---- Outputs: --------
        * Q: the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch.
    ---- Hints: --------
        * It's easier to follow a certain order to compute all the values: z1, a1, p, f, .... 
        * This problem can be solved using 5 line(s) of code.
'''
#---------------------
def compute_Q(S, W1, b1, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    
    #########################################
    return Q
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_Q
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_Q
        --- OR ---- 
        python -m nose -v test4.py:test_compute_Q
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Bellman Optimality Equation: compute target Q values) Suppose we have a copy of parameters for the Q network (called target network) and a mini-batch of training samples in the replay memory: including the new/next games states S_new and immediate rewards R in the sampled game steps in the mini-batch. Please compute the target Q values (Qt) for the mini-batch of samples using Bellman Optimality Equation. Note the gradients cannot flow through Qt, i.e., Qt tensor should not connect the gradient dL/dQt to the parameters of the Q network. 
    ---- Inputs: --------
        * S_new: the new/next game states for a mini-batch of sampled games steps after state transition, a torch tensor of shape (n,c0,h,w).
        * R: a mini-batch of the immediate rewards returned after the transition, a float vector of length (n).
        * T: a mini-batch of terminal state indicator, a boolean torch vector of length (n). T[i] represents whether or not S_new[i] is a terminal state (True: terminal state, False: otherwise).
        * W1: the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1).
        * b1: the biases of filters in the convolutional layer of Q network, a float torch vector of length c1.
        * W2: the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c).
        * b2: the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c.
        * gamma: the discount factor, a float scalar between 0 and 1.
    ---- Outputs: --------
        * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch.
    ---- Hints: --------
        * (Step 1) compute Q values on the new/next game states. 
        * (Step 2.1) If S_new[i] is the terminal state (i.e., T[i] = True), use the immediate reward R[i] as the target reward. 
        * (Step 2.2) Otherwise, use Bellman Optimality Equation to estimate the target Q values. 
        * You could re-use compute_Q() function. 
        * To negate the boolean values in a tensor x, you could use ~x. 
        * To convert a boolean-valued tensor x into an integer tensor, you could use x.int(). 
        * To detach the gradients from flowing through a torch tensor x, you could use x.detach() to detach the gradients from x so that gradient will not flow through x. 
        * To compute the max value of a tensor, you could use th.max() function. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_Qt(S_new, R, T, W1, b1, W2, b2, gamma=0.95):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    
    #########################################
    return Qt
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_Qt
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_Qt
        --- OR ---- 
        python -m nose -v test4.py:test_compute_Qt
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Train Q Network on a mini-batch of samples using gradient descent) Given a mini-batch of training samples: S (current game states), A (actions chosen), S_new (new/next game states) and R (immediate rewards), suppose the target Q values are already computed (Qt), please train the Q network using gradient descent: update the weights W1, W2 and biases b1, b2 using the gradients on the mini-batch of data samples. 
    ---- Inputs: --------
        * S: the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,c0,h,w).
        * A: a mini-batch of the actions chosen by the player, an integer vector of length (n).
        * Qt: the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch.
        * W1: the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1).
        * b1: the biases of filters in the convolutional layer of Q network, a float torch vector of length c1.
        * W2: the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c).
        * b2: the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c.
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W1, b1, W2, and b2).
    ---- Hints: --------
        * You could re-use compute_L() and update_parameters() functions from the previous problem. 
        * Step 1 Forward pass: compute Q values and the loss L. 
        * Step 2 Back propagation: compute the gradients of W1, b1, W2, and b2. 
        * Step 3 Gradient descent: update the parameters W1, b1, W2, and b2 using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def update_Q(S, A, Qt, W1, b1, W2, b2, optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_update_Q
        --- OR ---- 
        python3 -m nose -v test4.py:test_update_Q
        --- OR ---- 
        python -m nose -v test4.py:test_update_Q
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (using Q network for playing the game) Given the convolutional Q network with parameters W1, b1, W2, and b2 and we have only the current states s in the game. Please compute the predicted Q values on the current game state. 
    ---- Inputs: --------
        * s: the current state of the game, a torch tensor of shape (c0,h,w).
        * W1: the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1).
        * b1: the biases of filters in the convolutional layer of Q network, a float torch vector of length c1.
        * W2: the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c).
        * b2: the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c.
    ---- Outputs: --------
        * q: the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action.
    ---- Hints: --------
        * You could re-use the compute_Q() function above by creating a mini-batch of only one sample. 
        * To add a dimension to a torch tensor, you could use unsqueeze() function in torch tensor. 
        * To delete a dimension to a torch tensor, you could use squeeze() function in torch tensor. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict_q(s, W1, b1, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return q
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_predict_q
        --- OR ---- 
        python3 -m nose -v test4.py:test_predict_q
        --- OR ---- 
        python -m nose -v test4.py:test_predict_q
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Sample an action for a game step sample) Given the current game state s, choose an action using epsilon-greedy method on the Q values estimated by the Q network. We have epsilon probability to follow the random policy (randomly pick an action with uniform distribution) and  (1-epsilon) probability to follow the greedy policy on Q values (pick the action according to the largest Q value for the current game state s). 
    ---- Inputs: --------
        * s: the current state of the game, a torch tensor of shape (c0,h,w).
        * W1: the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1).
        * b1: the biases of filters in the convolutional layer of Q network, a float torch vector of length c1.
        * W2: the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c).
        * b2: the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c.
        * e: (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values.
    ---- Outputs: --------
        * a: the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1.
    ---- Hints: --------
        * You could re-use egreedy_policy() function from the previous problem. 
        * (Step 1) use the Q network to predict the Q values for the current game state. 
        * (Step 2) use epsilon-greedy policy on the Q values to sample an action for the current game state. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def sample_action(s, W1, b1, W2, b2, e):
    #########################################
    ## INSERT YOUR CODE HERE (8 points)
    
    #########################################
    return a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_sample_action
        --- OR ---- 
        python3 -m nose -v test4.py:test_sample_action
        --- OR ---- 
        python -m nose -v test4.py:test_sample_action
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 4: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py
        --- OR ---- 
        python3 -m nose -v test4.py
        --- OR ---- 
        python -m nose -v test4.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 4 (30 points in total)--------------------- ... ok
        * (3 points) compute_z1 ... ok
        * (2 points) compute_a1 ... ok
        * (2 points) compute_p ... ok
        * (2 points) flatten ... ok
        * (2 points) compute_z2 ... ok
        * (3 points) compute_Q ... ok
        * (3 points) compute_Qt ... ok
        * (3 points) update_Q ... ok
        * (2 points) predict_q ... ok
        * (8 points) sample_action ... ok
        ----------------------------------------------------------------------
        Ran 10 tests in 10.828s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of samples in a mini-batch, an integer scalar. 
* c:  the number of possible actions in the game, an integer scalar. 
* h:  the height of each input image of the game state, an integer scalar. 
* w:  the width of each input image of the game state, an integer scalar. 
* h1:  the height of the feature map after using the convolutional layer and maxpooling, h1 = (h - s1 +1)/2, an integer scalar. 
* w1:  the width of the feature map after using the convolutional layer and maxpooling, w1 = (w - s1 +1)/2, an integer scalar. 
* c0:  the number of input/color channels of the image in each game state, an integer scalar. 
* c1:  the number of filters in the convolutional layer, an integer scalar. 
* s1:  the size of filters (height = width = s1) in the convolutional layer of Q network, an integer scalar. 
* W1:  the weights of the filters in the convolutional layer of Q network, a float torch Tensor of shape (c0, s1, s1). 
* W2:  the weights of fully connected layer (2nd layer) of Q network, which is used to predict the Q values of each game state, a float torch matrix of shape (n_flat_features,c). 
* b1:  the biases of filters in the convolutional layer of Q network, a float torch vector of length c1. 
* b2:  the biases of fully connected layer (2nd layer) of Q network, a float torch vector of length c. 
* z1:  the linear logits of the convolutional layer of Q network on a mini-batch of samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1). 
* a1:  the ReLU activations of the convolutional layer of Q network on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1). 
* p:  the pooled activations (using max pooling) of the convolutional layer of the Q network on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1). 
* n_flat_features:  the number of input features to the fully connected layer after flattening the outputs of the convolutional layer on a mini-batch of images,  an integer scalar. n_flat_features = c1 * h1 * w1. 
* f:  the input features to the fully connected layer after flattening the outputs of the convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features ). 
* L:  the average of the least square losses on a mini-batch of training images, a torch float scalar. 
* data_loader:  the PyTorch loader of a dataset. 
* lr:  learning rate for gradient descent, a float scalar, between 0 and 1. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W1, b1, W2, and b2). 
* S:  the current states for a mini-batch of sampled game steps, a torch tensor of shape (n,c0,h,w). 
* S_new:  the new/next game states for a mini-batch of sampled games steps after state transition, a torch tensor of shape (n,c0,h,w). 
* R:  a mini-batch of the immediate rewards returned after the transition, a float vector of length (n). 
* A:  a mini-batch of the actions chosen by the player, an integer vector of length (n). 
* T:  a mini-batch of terminal state indicator, a boolean torch vector of length (n). T[i] represents whether or not S_new[i] is a terminal state (True: terminal state, False: otherwise). 
* Q:  the predicted Q values by the Q network on all actions for a mini-batch of game state samples, a pytorch matrix of shape (n, c). Q[i,j] represents the Q value on the j-th action for the i-th sample in the mini-batch. 
* Q_new:  the Q values (estimated by the target Q network) on the new game states for a mini-batch of sampled game steps, a pytorch matrix of shape (n, c). Q_new[i,j] represents the Q value on the j-th action for the new game state in the i-th sample of the mini-batch. 
* Qt:  the target Q values (estimated by Bellman Optimality Equation with the target Q network) for a mini-batch of samples, a pytorch vector of length (n). Qt[i] represents the target Q value for the i-th sample in the mini-batch. 
* s:  the current state of the game, a torch tensor of shape (c0,h,w). 
* q:  the Q values estimated by the Q-network on all actions for the current step of the game, a torch vector of length c. q[i] represents the estimated Q value for the i-th action. 
* a:  the index of the action being chosen by the player at the current step, an integer scalar between 0 and c-1. 
* gamma:  the discount factor, a float scalar between 0 and 1. 
* e:  (epsilon) the probability of the player to follow the random policy in epsilon-greedy method. e is a float scalar between 0 and 1. The player has 1-e probability in each time step to follow the greedy policy on the Q values. 

'''
#--------------------------------------------