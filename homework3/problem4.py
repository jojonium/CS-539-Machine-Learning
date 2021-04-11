import torch as th
import problem1 as sr
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 4: Long-Short Term Memory (LSTM) Method for Multi-class Time Sequence Classification (with PyTorch) (36 points)
    In this problem, you will implement another kind of recurrent neural network (LSTM, Long-Short Term Memory) for multi-class sequence classification problems.  Here we assume that each long time sequence is assigned with one multi-class label.  For example, in long audio classification, each time sequence is a long clip of audio recording, and the label of the sequence is one out of the multiple possible categories (0 check time, 1 check email, 2 add calendar event, 3 turn on the light, etc).  The goal of this problem is to learn the details of LSTM by building LSTM from scratch.  The structure of the LSTM includes one recurrent layer repeating itself for l time steps and a fully-connected layer attached to the last time step of the recurrent layer to predict the label of a time sequence.  (Recurrent layer for time step 1)-> (Recurrent layer for time step 2) -> ...(Recurrent layer for time step t) -> (Fully connected layer) -> predicted label

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Recurrent Layer: concatenating xt with ht_1) Given a mini-batch of time sequences at the t-th time step (xt). Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Please concatenate the two vectors for each sample in a mini-batch. 
    ---- Inputs: --------
        * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step.
        * ht_1: the output hidden states at the end of the (t-1)th time step, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ].
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_xh(xt, ht_1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return xh
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_xh
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_xh
        --- OR ---- 
        python -m nose -v test4.py:test_compute_xh
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Forget Gate: Linear Logits) Given the forget gates in an LSTM with parameters weights W_f and biases b_f. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Suppose we have already concatenated xt and h_(t-1) into xh. Please compute the linear logits z_f for the forget gates at the t-th time step on the mini-batch of data samples. 
    ---- Inputs: --------
        * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ].
        * W_f: (forget gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_f: the biases of the forget gates, a float torch vector of length h.
    ---- Outputs: --------
        * z_f: the linear logits of the forget gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z_f(xh, W_f, b_f):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z_f
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_z_f
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z_f
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z_f
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Forget Gate: Activation) Given the linear logits z_f of the forget gates at time step t on a mini-batch of training samples, please use the element-wise sigmoid function to compute the forget gates (activations) f_(t) at time step t. Each element f_t[i] is computed as sigmoid(z_f[i]). 
    ---- Inputs: --------
        * z_f: the linear logits of the forget gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * f_t: the forget gates (i.e., the activations of the forget gates) at the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_f_t(z_f):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return f_t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_f_t
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_f_t
        --- OR ---- 
        python -m nose -v test4.py:test_compute_f_t
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Input Gate: Linear Logits) Given the input gates in an LSTM with parameters weights W_i and biases b_i. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Suppose we have already concatenated xt and h_(t-1) into xh. Please compute the linear logits z_i for the input gates at the t-th time step on the mini-batch of data samples. 
    ---- Inputs: --------
        * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ].
        * W_i: (input gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_i: the biases of the input gates, a float torch vector of length h.
    ---- Outputs: --------
        * z_i: the linear logits of the input gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z_i(xh, W_i, b_i):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z_i
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_z_i
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z_i
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z_i
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Input Gate: Activation) Given the linear logits z_i of the input gates at time step t on a mini-batch of training samples, please use the element-wise sigmoid function to compute the input gates (activations) i_t at time step t. Each element i_t[i] is computed as sigmoid(z_i[i]). 
    ---- Inputs: --------
        * z_i: the linear logits of the input gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * i_t: the input gates (i.e., the activations of the input gates) at the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_i_t(z_i):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return i_t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_i_t
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_i_t
        --- OR ---- 
        python -m nose -v test4.py:test_compute_i_t
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Candidate Output: Linear Logits) Given an LSTM with parameters weights W_c and biases b_c for generating candidate outputs. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Suppose we have already concatenated xt and h_(t-1) into xh. Please compute the linear logits z_i for the candidate outputs at the t-th time step on the mini-batch of data samples. 
    ---- Inputs: --------
        * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ].
        * W_c: (candidate outputs) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_c: the biases of the candidate outputs, a float torch vector of length h.
    ---- Outputs: --------
        * z_c: the linear logits of the candidate cell states at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z_c(xh, W_c, b_c):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z_c
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_z_c
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z_c
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z_c
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Candidate Cell State: Activation) Given the linear logits z_c for candidate cell states at time step t on a mini-batch of training samples, please use the element-wise hyperbolic tangent function to compute the candidate cell states (activations) C_c at time step t. Each element C_c[i] is computed as tanh(z_c[i]). 
    ---- Inputs: --------
        * z_c: the linear logits of the candidate cell states at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * C_c: the candidate cell states (i.e., the activations in the candidate cell states) at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_C_c(z_c):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return C_c
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_C_c
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_C_c
        --- OR ---- 
        python -m nose -v test4.py:test_compute_C_c
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Update Cell State) Suppose we have the forget gates (f_t), input gates (i_t) and candidate cell states (C_c) at time step t.  We also have the old cell states (Ct_1) at the end of time step t-1, please compute the new cell state (Ct) for the time step t. 
    ---- Inputs: --------
        * f_t: the forget gates (i.e., the activations of the forget gates) at the t-th time step, a float torch tensor of shape (n, h).
        * i_t: the input gates (i.e., the activations of the input gates) at the t-th time step, a float torch tensor of shape (n, h).
        * C_c: the candidate cell states (i.e., the activations in the candidate cell states) at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
        * Ct_1: the old cell states of the LSTM cells at the end of (t-1)-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_Ct(f_t, i_t, C_c, Ct_1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return Ct
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_Ct
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_Ct
        --- OR ---- 
        python -m nose -v test4.py:test_compute_Ct
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Output Gate: Linear Logits) Given the output gates in an LSTM with parameters weights W_o and biases b_o. We have a mini-batch of data samples xt at the t-th time step. Suppose we have already computed the hidden states h_(t-1) at the previous time step (t-1). Suppose we have already concatenated xt and h_(t-1) into xh. Please compute the linear logits z_i for the output gates at the t-th time step on the mini-batch of data samples. 
    ---- Inputs: --------
        * xh: the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ].
        * W_o: (output gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_o: the biases of the output gates, a float torch vector of length h.
    ---- Outputs: --------
        * z_o: the linear logits of the output gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z_o(xh, W_o, b_o):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z_o
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_z_o
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z_o
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z_o
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Output Gate: Activation) Given the linear logits z_o of the output gates at time step t on a mini-batch of training samples, please use the element-wise sigmoid function to compute the output gates (activations) i_t at time step t. Each element o_t[i] is computed as sigmoid(z_o[i]). 
    ---- Inputs: --------
        * z_o: the linear logits of the output gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * o_t: the output gates (i.e., the activations of the output gates) at the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_o_t(z_o):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return o_t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_o_t
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_o_t
        --- OR ---- 
        python -m nose -v test4.py:test_compute_o_t
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Output Hidden States) Given the new cell states Ct of the LSTM recurrent layer at time step t. Suppose we have also computed the output gates for time step t. Please compute the output hidden states h_(t) at time step t. 
    ---- Inputs: --------
        * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
        * o_t: the output gates (i.e., the activations of the output gates) at the t-th time step, a float torch tensor of shape (n, h).
    ---- Outputs: --------
        * ht: the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_ht(Ct, o_t):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return ht
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_ht
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_ht
        --- OR ---- 
        python -m nose -v test4.py:test_compute_ht
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Recurrent Layer: Step Forward) Given an LSTM recurrent layer and we have a mini-batch of data samples x_t at time step t. Suppose we have already computed the old cell states C_(t-1) and the hidden states h_(t-1) at the previous time step t-1. Please compute the new cell states Ct and hidden states h_(t) of the recurrent layer on the mini-batch of samples for time step t. 
    ---- Inputs: --------
        * xt: a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step.
        * ht_1: the output hidden states at the end of the (t-1)th time step, a float torch tensor of shape (n, h).
        * Ct_1: the old cell states of the LSTM cells at the end of (t-1)-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
        * W_f: (forget gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_f: the biases of the forget gates, a float torch vector of length h.
        * W_i: (input gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_i: the biases of the input gates, a float torch vector of length h.
        * W_c: (candidate outputs) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_c: the biases of the candidate outputs, a float torch vector of length h.
        * W_o: (output gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_o: the biases of the output gates, a float torch vector of length h.
    ---- Outputs: --------
        * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
        * ht: the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h).
    ---- Hints: --------
        * It's easier to follow a certain order to compute all the values: xh, z_f, f_t, z_i ... 
        * This problem can be solved using 11 line(s) of code.
'''
#---------------------
def step(xt, ht_1, Ct_1, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return Ct, ht
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_step
        --- OR ---- 
        python3 -m nose -v test4.py:test_step
        --- OR ---- 
        python -m nose -v test4.py:test_step
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Fully-Connected Layer: Linear Logit) Given the hidden state h_(t) of the recurrent layer at time step t on a mini-batch of time sequences. Suppose the current time step t is the last time step (t=l) of the time sequences, please compute the linear logit z in the second layer (fully-connected layer) on the mini-batch of time sequences. 
    ---- Inputs: --------
        * ht: the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h).
        * W: the weights of the fully connected layer (2nd layer), a float torch matrix of shape (h, c).
        * b: the bias of the fully connected layer (2nd layer), a float torch vector of length c.
    ---- Outputs: --------
        * z: the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c).
    ---- Hints: --------
        * Here we are assuming that the classification task is multi-class classification. So the linear logit z is a vector on each time sequence in the mini-batch. 
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
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
        nosetests -v test4.py:test_compute_z
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_z
        --- OR ---- 
        python -m nose -v test4.py:test_compute_z
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Forward Pass) Given an LSTM model, and a mini-batch of time sequences x, where each time sequence has l time steps. Suppose the initial hidden states before seeing any data are given as h_(t=0). Similarly the initial cell states before seeing any data are given as C_(t=0)  Please compute the linear logits z of the LSTM on the mini-batch of time sequences. 
    ---- Inputs: --------
        * x: a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step.
        * ht: the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h).
        * Ct: the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h).
        * W_f: (forget gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_f: the biases of the forget gates, a float torch vector of length h.
        * W_i: (input gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_i: the biases of the input gates, a float torch vector of length h.
        * W_c: (candidate outputs) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_c: the biases of the candidate outputs, a float torch vector of length h.
        * W_o: (output gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_o: the biases of the output gates, a float torch vector of length h.
        * W: the weights of the fully connected layer (2nd layer), a float torch matrix of shape (h, c).
        * b: the bias of the fully connected layer (2nd layer), a float torch vector of length c.
    ---- Outputs: --------
        * z: the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c).
    ---- Hints: --------
        * Step 1 Recurrent Layer: apply the recurrent layer to each time step of the time sequences in the mini-batch. 
        * Step 2 Fully-connected Layer: compute the linear logit z each time sequence in the mini-batch. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def forward(x, ht, Ct, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W, b):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_forward
        --- OR ---- 
        python3 -m nose -v test4.py:test_forward
        --- OR ---- 
        python -m nose -v test4.py:test_forward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given an LSTM model and suppose we have already computed the linear logits z in the second layer (fully-connected layer) in the last time step t on a mini-batch of training samples. Suppose the labels of the training samples are in y. Please compute the average multi-class cross-entropy loss on the mini-batch of training samples. 
    ---- Inputs: --------
        * z: the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c).
        * y: the binary labels of the time sequences in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1.
    ---- Outputs: --------
        * L: the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar.
    ---- Hints: --------
        * In this problem setting, the classification task is assumed to be multi-class classification (e.g., predicting different types of commands). So the loss function should be multi-class cross entropy loss. 
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
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
        nosetests -v test4.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test4.py:test_compute_L
        --- OR ---- 
        python -m nose -v test4.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Gradient Descent) Suppose we are given an LSTM model with parameters (W_f, b_f, W_i, b_i, W_o, b_o, W_c, b_c, W and b) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the parameters on the mini-batch of data samples. Assume that we have already created an optimizer for the parameters. Please update the parameter values using gradient descent. After the update, the global gradients of all the parameters should be set to zero. 
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
        nosetests -v test4.py:test_update_parameters
        --- OR ---- 
        python3 -m nose -v test4.py:test_update_parameters
        --- OR ---- 
        python -m nose -v test4.py:test_update_parameters
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training LSTM) Given a training dataset X (time sequences), Y (labels) in a data loader, train an LSTM model using mini-batch stochastic gradient descent: iteratively update the parameters using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples. 
    ---- Inputs: --------
        * data_loader: the PyTorch loader of a dataset.
        * p: the number of input features at each time step of a time sequence, an integer scalar.
        * h: the number of memory cells (which is also the number of hidden states), an integer scalar.
        * n: batch size, the number of time sequences in a mini-batch, an integer scalar.
        * c: the number of classes in the classification task, an integer scalar.
        * alpha: the step-size parameter of gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * W_f: (forget gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_f: the biases of the forget gates, a float torch vector of length h.
        * W_i: (input gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_i: the biases of the input gates, a float torch vector of length h.
        * W_c: (candidate outputs) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_c: the biases of the candidate outputs, a float torch vector of length h.
        * W_o: (output gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_o: the biases of the output gates, a float torch vector of length h.
        * W: the weights of the fully connected layer (2nd layer), a float torch matrix of shape (h, c).
        * b: the bias of the fully connected layer (2nd layer), a float torch vector of length c.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits in the last layer z and the loss L. 
        * Step 2 Back propagation: compute the gradients of all parameters. 
        * Step 3 Gradient descent: update the parameters using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(data_loader, p, h, n, c, alpha=0.001, n_epoch=1000):
    W_f = th.randn(p+h, h, requires_grad=True) # initialize randomly using standard normal distribution
    b_f = th.zeros(h, requires_grad=True) # initialize b as all zeros
    W_i = th.randn(p+h, h, requires_grad=True) # initialize randomly using standard normal distribution
    b_i = th.zeros(h, requires_grad=True) # initialize b as all zeros
    W_o = th.randn(p+h, h, requires_grad=True) # initialize randomly using standard normal distribution
    b_o = th.zeros(h, requires_grad=True) # initialize b as all zeros
    W_c = th.randn(p+h, h, requires_grad=True) # initialize randomly using standard normal distribution
    b_c = th.zeros(h, requires_grad=True) # initialize b as all zeros
    W = th.randn(h,c, requires_grad=True) # initialize randomly using standard normal distribution
    b = th.zeros(c, requires_grad=True) # initialize b as all zeros
    ht = th.zeros(n, h) # initialize the hidden states as all zero
    Ct = th.zeros(n, h) # initialize the cell states as all zero
    optimizer = th.optim.SGD([W_f,b_f,W_i,b_i,W_o,b_o,W_c,b_c,W,b], lr=alpha) # SGD optimizer
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        for mini_batch in data_loader: # iterate through the dataset with one mini-batch of random training samples (x,y) at a time
            x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
            y=mini_batch[1] # the labels of the samples in a mini-batch
            #########################################
            ## INSERT YOUR CODE HERE (2 points)
    
            #########################################
    return W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W, b
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_train
        --- OR ---- 
        python3 -m nose -v test4.py:test_train
        --- OR ---- 
        python -m nose -v test4.py:test_train
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using LSTM model)  Given a trained LSTM model, suppose we have a mini-batch of test time sequences. Please use the LSTM model to predict the labels. 
    ---- Inputs: --------
        * x: a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step.
        * W_f: (forget gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_f: the biases of the forget gates, a float torch vector of length h.
        * W_i: (input gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_i: the biases of the input gates, a float torch vector of length h.
        * W_c: (candidate outputs) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_c: the biases of the candidate outputs, a float torch vector of length h.
        * W_o: (output gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h).
        * b_o: the biases of the output gates, a float torch vector of length h.
        * W: the weights of the fully connected layer (2nd layer), a float torch matrix of shape (h, c).
        * b: the bias of the fully connected layer (2nd layer), a float torch vector of length c.
    ---- Outputs: --------
        * y_predict: the predicted labels of a mini-batch of time sequences, a torch integer vector of length n. y_predict[i] represents the predicted label on the i-th time sequence in the mini-batch.
    ---- Hints: --------
        * This is a multi-class classification task, for each sample, the label should be predicted as the index of the largest value of each row of the linear logit z. 
        * You could use the argmax() function in PyTorch to return the indices of the largest values in a tensor. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def predict(x, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W, b):
    ht = th.zeros(x.size()[0], W.size()[0]) # initialize the hidden states as all zeros
    Ct = th.zeros(x.size()[0], W.size()[0]) # initialize the cell states as all zeros
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    
    #########################################
    return y_predict
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_predict
        --- OR ---- 
        python3 -m nose -v test4.py:test_predict
        --- OR ---- 
        python -m nose -v test4.py:test_predict
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
        ----------- Problem 4 (36 points in total)--------------------- ... ok
        * (2 points) compute_xh ... ok
        * (2 points) compute_z_f ... ok
        * (2 points) compute_f_t ... ok
        * (2 points) compute_z_i ... ok
        * (2 points) compute_i_t ... ok
        * (2 points) compute_z_c ... ok
        * (2 points) compute_C_c ... ok
        * (2 points) compute_Ct ... ok
        * (2 points) compute_z_o ... ok
        * (2 points) compute_o_t ... ok
        * (2 points) compute_ht ... ok
        * (2 points) step ... ok
        * (2 points) compute_z ... ok
        * (2 points) forward ... ok
        * (2 points) compute_L ... ok
        * (2 points) update_parameters ... ok
        * (2 points) train ... ok
        * (2 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 18 tests in 2.189s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  batch size, the number of time sequences in a mini-batch, an integer scalar. 
* l:  the length / the (maximum) number of time steps in each time sequence, an integer scalar. 
* p:  the number of input features at each time step of a time sequence, an integer scalar. 
* h:  the number of memory cells (which is also the number of hidden states), an integer scalar. 
* c:  the number of classes in the classification task, an integer scalar. 
* x:  a mini-batch of time sequences, a float torch tensor of shape (n, l, p). x[k,t] represents the k-th time sequence in the mini-batch at the t-th time step. 
* y:  the binary labels of the time sequences in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1. 
* y_predict:  the predicted labels of a mini-batch of time sequences, a torch integer vector of length n. y_predict[i] represents the predicted label on the i-th time sequence in the mini-batch. 
* xt:  a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p). xt[k] represents the k-th time sequence in the mini-batch at the t-th time step. 
* xh:  the concatenation of xt with ht_1 on a mini-batch of time sequences at the t-th time step, a float torch tensor of shape (n, p+h). xh[i] represents the vector concatenating the i-th time sequence input feature xt[i], with the hidden states on the i-th time sequence after the (t-1)-th time step: ht_1. For example, if we have two samples in a mini-batch at time step t, xt= [ [1,2], [3,4] ]. Suppose at the end of step t-1, the hidden states ht_1 on the two samples are [ [0.1,0.2,0.3], [0.4,0.5,0.6] ]. Then after concatenation, xh = [ [1,2,0.1,0.2,0.3], [3,4,0.4,0.5,0.6] ]. 
* W_f:  (forget gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h). 
* b_f:  the biases of the forget gates, a float torch vector of length h. 
* z_f:  the linear logits of the forget gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* f_t:  the forget gates (i.e., the activations of the forget gates) at the t-th time step, a float torch tensor of shape (n, h). 
* W_i:  (input gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h). 
* b_i:  the biases of the input gates, a float torch vector of length h. 
* z_i:  the linear logits of the input gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* i_t:  the input gates (i.e., the activations of the input gates) at the t-th time step, a float torch tensor of shape (n, h). 
* W_o:  (output gates) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h). 
* b_o:  the biases of the output gates, a float torch vector of length h. 
* z_o:  the linear logits of the output gates at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* o_t:  the output gates (i.e., the activations of the output gates) at the t-th time step, a float torch tensor of shape (n, h). 
* W_c:  (candidate outputs) the weights on the current input xt and past memory ht_1, a float torch matrix of shape (p+h, h). 
* b_c:  the biases of the candidate outputs, a float torch vector of length h. 
* z_c:  the linear logits of the candidate cell states at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* C_c:  the candidate cell states (i.e., the activations in the candidate cell states) at the t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* Ct:  the new cell states of the LSTM cells at the end of t-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* Ct_1:  the old cell states of the LSTM cells at the end of (t-1)-th time step on a mini-batch of data samples, a float torch tensor of shape (n, h). 
* ht:  the output hidden states at the end of the t-th time step, a float torch tensor of shape (n, h). 
* ht_1:  the output hidden states at the end of the (t-1)th time step, a float torch tensor of shape (n, h). 
* W:  the weights of the fully connected layer (2nd layer), a float torch matrix of shape (h, c). 
* b:  the bias of the fully connected layer (2nd layer), a float torch vector of length c. 
* z:  the linear logits of the fully connected layer (2nd layer) on a mini-batch of data samples, a float torch matrix of shape (n,c). 
* L:  the average multi-class cross entropy loss on a mini-batch of training samples, a torch float scalar. 
* data_loader:  the PyTorch loader of a dataset. 
* alpha:  the step-size parameter of gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for parameters in the model (U, V, b_h, W and b). 

'''
#--------------------------------------------