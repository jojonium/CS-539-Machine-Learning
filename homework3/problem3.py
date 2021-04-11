import torch as th
import numpy as np
import problem1 as sr
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: Recurrent Neural Network for Binary Time Sequence Classification (with PyTorch) (20 points)
    In this problem, you will implement the recurrent neural network for binary sequence classification problems.  Here we assume that each time sequence is assigned with one binary label.  For example, in audio classification, each time sequence is a short clip of audio recording, and the label of the sequence is either 0 (non-wake word) or 1 (wake word).  The goal of this problem is to learn the details of recurrent neural network by building RNN from scratch.  The structure of the RNN includes one recurrent layer repeating itself for l time steps and a fully-connected layer attached to the last time step of the recurrent layer to predict the label of a time sequence.  (Recurrent layer for time step 1)-> (Recurrent layer for time step 2) -> ...(Recurrent layer for time step t) -> (Fully connected layer) -> predicted label

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Recurrent Layer: Linear Logits) Given a recurrent neural network layer with parameters weights U, V and biases b_h. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Please compute the linear logits zt in the recurrent layer at the t-th time step on a mini-batch of data samples. 
    ---- Inputs: --------
        * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step.
        * ht_1: the hidden states (i.e., the activations of the recurrent layer) at the end of the (t-1)th time step, a float torch tensor of shape (n, h).
        * U: the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h).
        * V: the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output).
        * b_h: the biases of the recurrent layer, a float torch vector of length c. b_h[k] is the bias on the k-th neuron of the hidden states.
    ---- Outputs: --------
        * zt: the linear logits of the recurrent layer at the t-th time step on a mini-batch of time sequences,  a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_zt(xt, ht_1, U, V, b_h):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return zt
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_zt
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_zt
        --- OR ---- 
        python -m nose -v test3.py:test_compute_zt
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Tanh Activation) Given the linear logits zt of a recurrent layer at time step t, please use the element-wise hyperbolic tangent function to compute the activations h_(t) (also called hidden states) at time step t. Each element ht[i] is computed as tanh(zt[i]). 
    ---- Inputs: --------
        * zt: the linear logits of the recurrent layer at the t-th time step on a mini-batch of time sequences,  a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_ht(zt):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return ht
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_ht
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_ht
        --- OR ---- 
        python -m nose -v test3.py:test_compute_ht
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Step Forward) Given a recurrent neural network with parameters U, V and b_h and we have a mini-batch of data samples x_t at time step t. Suppose we have already computed the hidden state h_(t-1) at the previous time step t-1. Please compute the activations (also called hidden state) h_(t) of the recurrent layer for time step t. 
    ---- Inputs: --------
        * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step.
        * ht_1: the hidden states (i.e., the activations of the recurrent layer) at the end of the (t-1)th time step, a float torch tensor of shape (n, h).
        * U: the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h).
        * V: the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output).
        * b_h: the biases of the recurrent layer, a float torch vector of length c. b_h[k] is the bias on the k-th neuron of the hidden states.
    ---- Outputs: --------
        * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def step(xt, ht_1, U, V, b_h):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return ht
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_step
        --- OR ---- 
        python3 -m nose -v test3.py:test_step
        --- OR ---- 
        python -m nose -v test3.py:test_step
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Fully-Connected Layer: Linear Logit) Given the hidden state h_(t) of the recurrent neural network layer at time step t on a mini-batch of time sequences. Suppose the current time step t is the last time step (t=l) of the time sequences, please compute the linear logit z in the second layer (fully-connected layer) on the mini-batch of time sequences. 
    ---- Inputs: --------
        * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h).
        * W: the weights of the fully connected layer (2nd layer) of RNN, a float torch vector of length h.
        * b: the bias of the fully connected layer (2nd layer) of RNN, a float torch scalar.
    ---- Outputs: --------
        * z: the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n.
    ---- Hints: --------
        * Here we are assuming that the classification task is binary classification. So the linear logit z is a scalar on each time sequence in the mini-batch. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z(ht, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_z
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_z
        --- OR ---- 
        python -m nose -v test3.py:test_compute_z
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Forward Pass) Given a recurrent neural network with parameters U, V, b_h, W and b, and a mini-batch of time sequences x, where each time sequence has l time steps. Suppose the initial hidden states of the RNN before seeing any data are given as h_(t=0). Please compute the linear logits z of the RNN on the mini-batch of time sequences. 
    ---- Inputs: --------
        * x: a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step.
        * ht: the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h).
        * U: the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h).
        * V: the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output).
        * b_h: the biases of the recurrent layer, a float torch vector of length c. b_h[k] is the bias on the k-th neuron of the hidden states.
        * W: the weights of the fully connected layer (2nd layer) of RNN, a float torch vector of length h.
        * b: the bias of the fully connected layer (2nd layer) of RNN, a float torch scalar.
    ---- Outputs: --------
        * z: the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n.
    ---- Hints: --------
        * Step 1 Recurrent Layer: apply the recurrent layer to each time step of the time sequences in the mini-batch. 
        * Step 2 Fully-connected Layer: compute the linear logit z each time sequence in the mini-batch. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def forward(x, ht, U, V, b_h, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_forward
        --- OR ---- 
        python3 -m nose -v test3.py:test_forward
        --- OR ---- 
        python -m nose -v test3.py:test_forward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given a recurrent neural network and suppose we have already computed the linear logits z in the second layer (fully-connected layer) in the last time step t on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average binary cross-entropy loss on the mini-batch of training samples. 
    ---- Inputs: --------
        * z: the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n.
        * y: the binary labels of the time sequences in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1.
    ---- Outputs: --------
        * L: the average binary cross entropy loss on a mini-batch of training samples, a torch float scalar.
    ---- Hints: --------
        * In our problem setting, the classification task is assumed to be binary classification (e.g., predicting 'wake word' or not) instead of multi-class classification (e.g., predicting different types of commands). So the loss function should be binary cross entropy loss instead of multi-class cross entropy loss. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(z, y):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
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
    (Gradient Descent) Suppose we are given a recurrent neural network with parameters (U, V, bh, W and b) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the parameters on the mini-batch of data samples. Assume that we have already created an optimizer for the parameters. Please update the parameter values using gradient descent. After the update, the global gradients of all the parameters should be set to zero. 
    ---- Inputs: --------
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (U, V, b_h, W and b).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_parameters(optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
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
    (Training Recurrent Neural Network) Given a training dataset X (time sequences), Y (labels) in a data loader, train the recurrent neural network using mini-batch stochastic gradient descent: iteratively update the parameters using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples. 
    ---- Inputs: --------
        * data_loader: the PyTorch loader of a dataset.
        * p: the number of input features at each time step of a time sequence, an integer scalar.
        * h: the number of neurons in the hidden states (or the activations of the recurrent layer), an integer scalar.
        * n: batch size, the number of time sequences in a mini-batch, an integer scalar.
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * U: the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h).
        * V: the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output).
        * b_h: the biases of the recurrent layer, a float torch vector of length c. b_h[k] is the bias on the k-th neuron of the hidden states.
        * W: the weights of the fully connected layer (2nd layer) of RNN, a float torch vector of length h.
        * b: the bias of the fully connected layer (2nd layer) of RNN, a float torch scalar.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits in the last layer z and the loss L. 
        * Step 2 Back propagation: compute the gradients of all parameters. 
        * Step 3 Gradient descent: update the parameters using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(data_loader, p, h, n, alpha=0.001, n_epoch=1000):
    U = th.randn(p, h, requires_grad=True) # initialize randomly using standard normal distribution
    V = th.randn(h, h, requires_grad=True) # initialize randomly using standard normal distribution
    b_h = th.zeros(h, requires_grad=True) # initialize b as all zeros
    W = th.randn(h, requires_grad=True) # initialize randomly using standard normal distribution
    b = th.zeros(1, requires_grad=True) # initialize b as zero
    ht = th.zeros(n, h) # initialize the hidden states as all zero
    optimizer = th.optim.SGD([U,V,b_h,W,b], lr=alpha) # SGD optimizer
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        for mini_batch in data_loader: # iterate through the dataset with one mini-batch of random training samples (x,y) at a time
            x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
            y=mini_batch[1] # the labels of the samples in a mini-batch
            #########################################
            ## INSERT YOUR CODE HERE (4 points)
    
            #########################################
    return U, V, b_h, W, b
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_train
        --- OR ---- 
        python3 -m nose -v test3.py:test_train
        --- OR ---- 
        python -m nose -v test3.py:test_train
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using RNN model)  Given a trained RNN model, suppose we have a mini-batch of test time sequences. Please use the RNN model to predict the labels. 
    ---- Inputs: --------
        * x: a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step.
        * U: the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h).
        * V: the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output).
        * b_h: the biases of the recurrent layer, a float torch vector of length c. b_h[k] is the bias on the k-th neuron of the hidden states.
        * W: the weights of the fully connected layer (2nd layer) of RNN, a float torch vector of length h.
        * b: the bias of the fully connected layer (2nd layer) of RNN, a float torch scalar.
    ---- Outputs: --------
        * y_predict: the predicted labels of a mini-batch of time sequences, a torch integer vector of length n. y_predict[i] represents the predicted label on the i-th time sequence in the mini-batch.
    ---- Hints: --------
        * This is a binary classification task. When a linear logit in z is >0, then the label should be predicted as 1, otherwise 0. 
        * You could use the x>0 in PyTorch to convert a float tensor into a binary/boolean tensor using the element-wise operation (x[i]>0 returns True, otherwise return False). 
        * You could use the x.int() in PyTorch to convert a boolean tensor into an integer tensor. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def predict(x, U, V, b_h, W, b):
    ht = th.zeros(x.size()[0], V.size()[0]) # initialize the hidden states as all zeros
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return y_predict
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_predict
        --- OR ---- 
        python3 -m nose -v test3.py:test_predict
        --- OR ---- 
        python -m nose -v test3.py:test_predict
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
        ----------- Problem 3 (20 points in total)--------------------- ... ok
        * (2 points) compute_zt ... ok
        * (2 points) compute_ht ... ok
        * (2 points) step ... ok
        * (2 points) compute_z ... ok
        * (2 points) forward ... ok
        * (2 points) compute_L ... ok
        * (2 points) update_parameters ... ok
        * (4 points) train ... ok
        * (2 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 9 tests in 0.789s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  batch size, the number of time sequences in a mini-batch, an integer scalar. 
* l:  the length / the (maximum) number of time steps in each time sequence, an integer scalar. 
* p:  the number of input features at each time step of a time sequence, an integer scalar. 
* h:  the number of neurons in the hidden states (or the activations of the recurrent layer), an integer scalar. 
* x:  a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step. 
* y:  the binary labels of the time sequences in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1. 
* y_predict:  the predicted labels of a mini-batch of time sequences, a torch integer vector of length n. y_predict[i] represents the predicted label on the i-th time sequence in the mini-batch. 
* xt:  a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step. 
* zt:  the linear logits of the recurrent layer at the t-th time step on a mini-batch of time sequences,  a float torch tensor of shape (n, h). 
* ht:  the hidden states (i.e., the activations of the recurrent layer) at the end of the t-th time step, a float torch tensor of shape (n, h). 
* ht_1:  the hidden states (i.e., the activations of the recurrent layer) at the end of the (t-1)th time step, a float torch tensor of shape (n, h). 
* U:  the weights of the recurrent layers on the input features of the time sequence at the current time step, a float torch Tensor of shape (p, h). 
* V:  the weights of the recurrent layers on the old memory ht_1 (the hidden states at the previous time step (t-1), a float torch Tensor of shape (h, h). Here V[j,k] is the weight connecting the j-th neuron in ht_1 (input) to the k-th neuron in ht (output). 
* b_h:  the biases of the recurrent layer, a float torch vector of length c. b_h[k] is the bias on the k-th neuron of the hidden states. 
* W:  the weights of the fully connected layer (2nd layer) of RNN, a float torch vector of length h. 
* b:  the bias of the fully connected layer (2nd layer) of RNN, a float torch scalar. 
* z:  the linear logits of the fully connected layer (2nd layer) of RNN on a mini-batch of data samples, a float torch vector of length n. 
* L:  the average binary cross entropy loss on a mini-batch of training samples, a torch float scalar. 
* data_loader:  the PyTorch loader of a dataset. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (U, V, b_h, W and b). 

'''
#--------------------------------------------