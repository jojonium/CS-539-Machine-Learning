import torch as th
import numpy as np
import problem1 as sr
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Convolutional Neural Network for Binary Image Classification (using PyTorch) (32 points)
    In this problem, you will implement a convolutional neural network (CNN) with two convolution layers with max pooling and ReLU activations, and one fully-connected layer at the end to predict the label. The classification task is that given an input RGB color image, predict the binary label of the image (e.g., whether the image is the owner of the smartphone or not).  The goal of this problem is to learn the details of convolutional neural network by building CNN from scratch. The structure of the CNN is (Conv layer 1) -> ReLU -> maxpooling -> (Conv layer 2) -> ReLU -> maxpooling -> (Fully-connected layer)

    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Example A: Convolutional Layer with 1 filter and 1 input/color channel on 1 image) Let's first get familiar with the 2D Convolution. Let's start with one filter and one image with one input channel. Given a convolutional filter (with weights W_a and bias  b_a) and an image x_a with one color channel, height h and width w, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0. 
    ---- Inputs: --------
        * x_a: one input image with 1 input/color channel with hight h and width w of pixels, a float torch tensor of shape (h, w).
        * W_a: the weights of 1 convolutional filter with 1 color/input channel,  a float torch matrix of shape (s, s).
        * b_a: the bias of 1 convolutional filter with 1 color/input channel,  a float torch scalar.
    ---- Outputs: --------
        * z_a: the linear logits of the convolutional layer with 1 filter and 1 channel on one image,  a float torch matrix of shape (h-s+1, w-s+1).
    ---- Hints: --------
        * You could use A.size() to get the shape of a torch tensor A. 
        * You could use th.empty() to create an empty torch tensor. 
        * In order to connect the global gradients of z_a (dL_dz) with the global gradients of W_a (dL_dW) and b_a (dL_db), please use operators/functions in PyTorch to build the computational graph. 
        * You could use A*B to compute the element-wise product of two torch tensors A and B. 
        * You could use A.sum() to compute the sum of all elements in a torch tensor A. 
        * You could use A+B to compute the element-wise sum of two torch tensors A and B. 
        * You could use A[i:j,k:l] to get a sub-matrix of a torch matrix A. 
        * This problem can be solved using 6 line(s) of code.
'''
#---------------------
def conv2d_a(x_a, W_a, b_a):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    h, w = x_a.size()
    s = W_a.size()[0]
    z_a = th.empty((h - s + 1, w - s + 1))
    for i in range(h-s+1):
        for j in range(w-s+1):
            z_a[i, j] = (W_a * x_a[i:i+s, j:j+s]).sum() + b_a
    #########################################
    return z_a
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_conv2d_a
        --- OR ---- 
        python3 -m nose -v test2.py:test_conv2d_a
        --- OR ---- 
        python -m nose -v test2.py:test_conv2d_a
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Example B: Convolutional Layer with 1 filter, c input/color channels on 1 image) Let's continue with one filter and one image with multiple (c) input channels. Given a convolutional filter (with weights W_b and bias b_b) and an image x_b with c color channels, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0. 
    ---- Inputs: --------
        * x_b: one input image with c color/input channels with hight h and width w of pixels, a float torch tensor of shape (c, h, w). Here x_b[i] represents i-th color/input channel of the color image.
        * W_b: the weights of 1 convolutional filter with c color/input channels,  a float torch tensor of shape (c, s, s). Here W_b[i] represents the weights of the filter on the i-th input/color channel.
        * b_b: the bias of 1 convolutional filter with c color/input channels,  a float torch scalar.
    ---- Outputs: --------
        * z_b: the linear logits of the convolutional layer with 1 filter and c channels on one image,  a float torch matrix of shape (h-s+1, w-s+1).
    ---- Hints: --------
        * You could use A[:, i:j,k:l] to get all the indices in the first dimension of a 3D torch tensor A, while only using sub-sets of the indices in the 2nd and 3rd dimension of the tensor A. 
        * This problem can be solved using 6 line(s) of code.
'''
#---------------------
def conv2d_b(x_b, W_b, b_b):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    c, h, w = x_b.size()
    s = W_b.size()[1]
    z_b = th.empty((h - s + 1, w - s + 1))
    for i in range(h-s+1):
        for j in range(w-s+1):
            z_b[i, j] = (W_b * x_b[:, i:i+s, j:j+s]).sum() + b_b
    #########################################
    return z_b
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_conv2d_b
        --- OR ---- 
        python3 -m nose -v test2.py:test_conv2d_b
        --- OR ---- 
        python -m nose -v test2.py:test_conv2d_b
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Example C: Convolutional Layer with c1 filters, c input/color channels on 1 image) Let's continue with multiple (c1) filters and one image with multiple (c) input channels. Given c1 convolutional filters (with weights W_c and biases b_c) and an image x_c with c color channels, please compute the 2D convolution with the filter on the image. Here we assume that stride=1 and padding = 0. 
    ---- Inputs: --------
        * x_c: one input image with c color channels with hight h and width w of pixels, a float torch tensor of shape (c, h, w).
        * W_c: the weights of c1 convolutional filters, where each filter has c color/input channels,  a float torch tensor of shape (c1, c, s, s). Here W_c[i] represents the weights of i-th convolutional filter.
        * b_c: the biases of c1 convolutional filters,  a float torch vector of length c1. Here b_c[i] represents the bias of the i-th convolutional filter.
    ---- Outputs: --------
        * z_c: the linear logits of the c1 convolutional filters on one image, a float torch tensor of shape (c1, h-s+1, w-s+1). Here z_c[i] represents the linear logits the i-th convolutional filter.
    ---- Hints: --------
        * you could re-use the previous function conv2d_b() in this function. 
        * This problem can be solved using 5 line(s) of code.
'''
#---------------------
def conv2d_c(x_c, W_c, b_c):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    c, h, w = x_c.size()
    c1, c, s, s = W_c.size()
    z_c = th.empty((c1, h - s + 1, w - s + 1))
    for i in range(len(W_c)):
        z_c[i] = conv2d_b(x_c, W_c[i], b_c[i])
    #########################################
    return z_c
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_conv2d_c
        --- OR ---- 
        python3 -m nose -v test2.py:test_conv2d_c
        --- OR ---- 
        python -m nose -v test2.py:test_conv2d_c
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Convolutional Layer 1: Linear Logits) (Convolutional layer with c1 filters and c input/color channels on n images) Let's continue with first convolutional layer of the CNN. Here we have multiple (c1) filters and multiple (n) images in a mini-batch, where each image has multiple (c) color channels. Given c1 convolutional filters (with weights W1 and biases b1) and a mini-batch of images (x), please compute the 2D convolution on the mini-batch of images. Here we assume that stride=1 and padding = 0. 
    ---- Inputs: --------
        * x: a mini-batch of input images, a float torch tensor of shape (n, c, h, w).
        * W1: the weights of the filters in the first convolutional layer of CNN, a float torch Tensor of shape (c, s1, s1).
        * b1: the biases of filters in the first convolutional layer of CNN, a float torch vector of length c1.
    ---- Outputs: --------
        * z1: the linear logits of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1).
    ---- Hints: --------
        * You could use the conv2d() function in PyTorch.. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z1(x, W1, b1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    z1 = th.conv2d(x, W1, b1)
    #########################################
    return z1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_z1
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_z1
        --- OR ---- 
        python -m nose -v test2.py:test_compute_z1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Convolutional Layer 1:  ReLU activation) Given the linear logits (z1) of the first convolutional layer, please compute the ReLU activations. 
    ---- Inputs: --------
        * z1: the linear logits of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1).
    ---- Outputs: --------
        * a1: the ReLU activations of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_a1(z1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    a1 = th.nn.ReLU()(z1)
    #########################################
    return a1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_a1
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_a1
        --- OR ---- 
        python -m nose -v test2.py:test_compute_a1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Convolutional Layer 1:  Maxpooling) Given the activations (a1) of first convolutional layer, please compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2. 
    ---- Inputs: --------
        * a1: the ReLU activations of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1).
    ---- Outputs: --------
        * p1: the pooled activations (using max pooling) of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_p1(a1):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    p1 = th.nn.MaxPool2d(2)(a1)
    #########################################
    return p1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_p1
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_p1
        --- OR ---- 
        python -m nose -v test2.py:test_compute_p1
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Convolutional Layer 2: Linear Logits) In the second convolutional layer, suppose we have c2 filters, c1 input channels and a mini-batch of n images. Given c2 convolutional filters (with weights W2 and biases b2), please compute the 2D convolution on feature maps p1 of the mini-batch of images. Here we assume that stride=1 and padding = 0. 
    ---- Inputs: --------
        * p1: the pooled activations (using max pooling) of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1).
        * W2: the weights of the filters in the second convolutional layer of CNN, a float torch Tensor of shape (c1, s2, s2).
        * b2: the biases of filters in the second convolutional layer of CNN, a float torch vector of length c2.
    ---- Outputs: --------
        * z2: the linear logits of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z2(p1, W2, b2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    z2 = th.conv2d(p1, W2, b2)
    #########################################
    return z2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_z2
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_z2
        --- OR ---- 
        python -m nose -v test2.py:test_compute_z2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Convolutional Layer 2:  ReLU activation) Given the linear logits (z2) of the second convolutional layer, please compute the ReLU activations. 
    ---- Inputs: --------
        * z2: the linear logits of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1).
    ---- Outputs: --------
        * a2: the ReLU activations of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1 ).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_a2(z2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    a2 = th.nn.ReLU()(z2)
    #########################################
    return a2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_a2
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_a2
        --- OR ---- 
        python -m nose -v test2.py:test_compute_a2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Convolutional Layer 2:  Maxpooling) Given the activations (a2) of second convolutional layer, please compute the max pooling results. Here we assume that the size of the pooling window is 2 x 2. 
    ---- Inputs: --------
        * a2: the ReLU activations of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1 ).
    ---- Outputs: --------
        * p2: the pooled activations (using max pooling) of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h2, w2).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_p2(a2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    p2 = th.nn.MaxPool2d(2)(a2)
    #########################################
    return p2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_p2
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_p2
        --- OR ---- 
        python -m nose -v test2.py:test_compute_p2
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (flatten) Given the pooling results (p2) of the second convolutional layer of shape n x c2 x h2 x w2, please flatten the pooling results into a vector, so that it can be used as the input to the fully-connected layer. The flattened features will be a 2D matrix of shape (n x n_flat_features), where n_flat_features is computed as c2 x h2 x w2. 
    ---- Inputs: --------
        * p2: the pooled activations (using max pooling) of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h2, w2).
    ---- Outputs: --------
        * f: the input features to the fully connected layer after flattening the outputs of the second convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features ).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def flatten(p2):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    f = th.flatten(p2, 1)
    #########################################
    return f
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_flatten
        --- OR ---- 
        python3 -m nose -v test2.py:test_flatten
        --- OR ---- 
        python -m nose -v test2.py:test_flatten
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Fully Connected Layer) Given flattened features on a mini-batch of images, please compute the linear logits z3 of the fully-connected layer on the mini-batch of images. 
    ---- Inputs: --------
        * f: the input features to the fully connected layer after flattening the outputs of the second convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features ).
        * W3: the weights of fully connected layer (3rd layer) of CNN, which is used to predict the binary class label of each image, a float torch vector of length (n_flat_features).
        * b3: the bias value of fully connected layer (3rd layer) of CNN, a float torch scalar.
    ---- Outputs: --------
        * z3: the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n).
    ---- Hints: --------
        * Because we are solving a binary classification task instead of multi-class classification task, the fully-connected layer here only has one output neuron, instead of multiple output neurons. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_z3(f, W3, b3):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    z3 = f@W3 + b3
    #########################################
    return z3
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_z3
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_z3
        --- OR ---- 
        python -m nose -v test2.py:test_compute_z3
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given a convolutional neural network with parameters W1, b1, W2, b2, W3 and b3 and we have a mini-batch of images x. Please compute the linear logits in fully-connected layer on the mini-batch of images. 
    ---- Inputs: --------
        * x: a mini-batch of input images, a float torch tensor of shape (n, c, h, w).
        * W1: the weights of the filters in the first convolutional layer of CNN, a float torch Tensor of shape (c, s1, s1).
        * b1: the biases of filters in the first convolutional layer of CNN, a float torch vector of length c1.
        * W2: the weights of the filters in the second convolutional layer of CNN, a float torch Tensor of shape (c1, s2, s2).
        * b2: the biases of filters in the second convolutional layer of CNN, a float torch vector of length c2.
        * W3: the weights of fully connected layer (3rd layer) of CNN, which is used to predict the binary class label of each image, a float torch vector of length (n_flat_features).
        * b3: the bias value of fully connected layer (3rd layer) of CNN, a float torch scalar.
    ---- Outputs: --------
        * z3: the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n).
    ---- Hints: --------
        * It's easier to follow a certain order to compute all the values: z1, a1, p1, z2, .... 
        * This problem can be solved using 8 line(s) of code.
'''
#---------------------
def forward(x, W1, b1, W2, b2, W3, b3):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    z1 = compute_z1(x, W1, b1)
    a1 = compute_a1(z1)
    p1 = compute_p1(a1)
    z2 = compute_z2(p1, W2, b2)
    a2 = compute_a2(z2)
    p2 = compute_p2(a2)
    f = flatten(p2)
    z3 = compute_z3(f, W3, b3)
    #########################################
    return z3
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_forward
        --- OR ---- 
        python3 -m nose -v test2.py:test_forward
        --- OR ---- 
        python -m nose -v test2.py:test_forward
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given a convolutional neural network and suppose we have already computed the linear logits z3 in the last layer (fully-connected layer) on a mini-batch of training images. Suppose the labels of the training images are in y. Please compute the average binary cross-entropy loss on the mini-batch of training images. 
    ---- Inputs: --------
        * z3: the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n).
        * y: the binary labels of the images in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1.
    ---- Outputs: --------
        * L: the average of the binary cross entropy losses on a mini-batch of training images, a torch float scalar.
    ---- Hints: --------
        * Because the classification task is binary classification (e.g., predicting 'owner of the smartphone' or not) instead of multi-class classification (e.g., predicting which user in the image). So the loss function should be binary cross entropy loss instead of multi-class cross entropy loss. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_L(z3, y):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    L = th.nn.BCEWithLogitsLoss()(z3, y)
    #########################################
    return L
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_L
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_L
        --- OR ---- 
        python -m nose -v test2.py:test_compute_L
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Gradient Descent) Suppose we are given a convolutional neural network with parameters (W1, b1, W2 and b2) and we have a mini-batch of training data samples (x,y).  Suppose we have already computed the global gradients of the average loss L w.r.t. the weights W1, W2 and biases b1 and b2 on the mini-batch of data samples. Assume that we have already created an optimizer for the parameter W1, b1, W2 and b2. Please update the weights W1, W2 and biases b1 and b2 using gradient descent. After the update, the global gradients of W1, b1, W2 and b2 should be set to all zeros. 
    ---- Inputs: --------
        * optimizer: a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W1, b1, W2, b2, W3 and b3).
    ---- Hints: --------
        * You can re-use the functions in softmax regression.  For example, sr.function_name() represents the function_name() function in softmax regression. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_parameters(optimizer):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    sr.update_parameters(optimizer)
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_update_parameters
        --- OR ---- 
        python3 -m nose -v test2.py:test_update_parameters
        --- OR ---- 
        python -m nose -v test2.py:test_update_parameters
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training Convolutional Neural Network) Given a training dataset X (images), Y (labels) in a data loader, please train a convolutional neural network using mini-batch stochastic gradient descent: iteratively update the weights W1, W2, W3 and biases b1, b2, b3 using the gradients on each mini-batch of random data samples.  We repeat n_epoch passes over all the training samples. 
    ---- Inputs: --------
        * data_loader: the PyTorch loader of a dataset.
        * c: the number of color channels in each input image, an integer scalar.
        * c1: the number of filters in the first convolutional layer, an integer scalar.
        * c2: the number of filters in the second convolutional layer, an integer scalar.
        * h: the height of each input image, an integer scalar.
        * w: the width of each input image, an integer scalar.
        * s1: the size of filters (height = width = s1) in the first convolutional layer of CNN, an integer scalar.
        * s2: the size of filters (height = width = s2) in the second convolutional layer of CNN, an integer scalar.
        * alpha: the step-size parameter for gradient descent, a float scalar.
        * n_epoch: the number of passes to go through the training dataset in the training process, an integer scalar.
    ---- Outputs: --------
        * W1: the weights of the filters in the first convolutional layer of CNN, a float torch Tensor of shape (c, s1, s1).
        * b1: the biases of filters in the first convolutional layer of CNN, a float torch vector of length c1.
        * W2: the weights of the filters in the second convolutional layer of CNN, a float torch Tensor of shape (c1, s2, s2).
        * b2: the biases of filters in the second convolutional layer of CNN, a float torch vector of length c2.
        * W3: the weights of fully connected layer (3rd layer) of CNN, which is used to predict the binary class label of each image, a float torch vector of length (n_flat_features).
        * b3: the bias value of fully connected layer (3rd layer) of CNN, a float torch scalar.
    ---- Hints: --------
        * Step 1 Forward pass: compute the linear logits in the last layer z3 and the loss L. 
        * Step 2 Back propagation: compute the gradients of W1, b1, W2, b2, W3 and b3. 
        * Step 3 Gradient descent: update the parameters W1, b1, W2, b2, W3 and b3 using gradient descent. 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def train(data_loader, c, c1, c2, h, w, s1, s2, alpha=0.001, n_epoch=100):
    W1 = th.randn(c1, c, s1, s1, requires_grad=True) # initialize randomly using standard normal distribution
    b1 = th.zeros(c1, requires_grad=True) # initialize b as all zeros
    W2 = th.randn(c2, c1, s2, s2, requires_grad=True) # initialize randomly using standard normal distribution
    b2 = th.zeros(c2, requires_grad=True) # initialize b as all zeros
    h1 = (h-s1+1)//2
    w1 = (w-s1+1)//2
    h2 = (h1-s2+1)//2
    w2 = (w1-s2+1)//2
    n_flat_features = h2*w2*c2
    W3 = th.randn(n_flat_features, requires_grad=True) # initialize randomly using standard normal distribution
    b3 = th.zeros(1, requires_grad=True) # initialize b as zero
    optimizer = th.optim.SGD([W1,b1,W2,b2], lr=alpha) # SGD optimizer
    for _ in range(n_epoch): # iterate through the dataset n_epoch times
        for mini_batch in data_loader: # iterate through the dataset with one mini-batch of random training samples (x,y) at a time
            x=mini_batch[0] # the feature vectors of the data samples in a mini-batch
            y=mini_batch[1] # the labels of the samples in a mini-batch
            #########################################
            ## INSERT YOUR CODE HERE (2 points)
            update_parameters(optimizer)
            z3 = forward(x, W1, b1, W2, b2, W3, b3)
            L = compute_L(z3, y)
            L.backward()
            #########################################
    return W1, b1, W2, b2, W3, b3
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_train
        --- OR ---- 
        python3 -m nose -v test2.py:test_train
        --- OR ---- 
        python -m nose -v test2.py:test_train
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Using CNN model)  Given a trained CNN model with parameters W1, b1, W2, b2, W3 and b3. Suppose we have a mini-batch of test images. Please use the CNN model to predict the labels. 
    ---- Inputs: --------
        * x: a mini-batch of input images, a float torch tensor of shape (n, c, h, w).
        * W1: the weights of the filters in the first convolutional layer of CNN, a float torch Tensor of shape (c, s1, s1).
        * b1: the biases of filters in the first convolutional layer of CNN, a float torch vector of length c1.
        * W2: the weights of the filters in the second convolutional layer of CNN, a float torch Tensor of shape (c1, s2, s2).
        * b2: the biases of filters in the second convolutional layer of CNN, a float torch vector of length c2.
        * W3: the weights of fully connected layer (3rd layer) of CNN, which is used to predict the binary class label of each image, a float torch vector of length (n_flat_features).
        * b3: the bias value of fully connected layer (3rd layer) of CNN, a float torch scalar.
    ---- Outputs: --------
        * y_predict: the predicted labels of a mini-batch of test images, a torch integer vector of length n. y_predict[i] represents the predicted label (0 or 1) on the i-th test sample in the mini-batch.
    ---- Hints: --------
        * This is a binary classification task. When linear a logit in z is >0, then the label should be predicted as 1, otherwise 0. 
        * You could use the x>0 in PyTorch to convert a float tensor into a binary/boolean tensor using the element-wise operation (x[i]>0 returns True, otherwise return False). 
        * You could use the x.int() in PyTorch to convert a boolean tensor into an integer tensor. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def predict(x, W1, b1, W2, b2, W3, b3):
    #########################################
    ## INSERT YOUR CODE HERE (2 points)
    z = forward(x, W1, b1, W2, b2, W3, b3)
    y_predict = (z > 0).int()
    #########################################
    return y_predict
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_predict
        --- OR ---- 
        python3 -m nose -v test2.py:test_predict
        --- OR ---- 
        python -m nose -v test2.py:test_predict
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
        ----------- Problem 2 (32 points in total)--------------------- ... ok
        * (2 points) conv2d_a ... ok
        * (2 points) conv2d_b ... ok
        * (2 points) conv2d_c ... ok
        * (2 points) compute_z1 ... ok
        * (2 points) compute_a1 ... ok
        * (2 points) compute_p1 ... ok
        * (2 points) compute_z2 ... ok
        * (2 points) compute_a2 ... ok
        * (2 points) compute_p2 ... ok
        * (2 points) flatten ... ok
        * (2 points) compute_z3 ... ok
        * (2 points) forward ... ok
        * (2 points) compute_L ... ok
        * (2 points) update_parameters ... ok
        * (2 points) train ... ok
        * (2 points) predict ... ok
        ----------------------------------------------------------------------
        Ran 16 tests in 2.689s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  batch size, the number of images in a mini-batch, an integer scalar. 
* x:  a mini-batch of input images, a float torch tensor of shape (n, c, h, w). 
* y:  the binary labels of the images in a mini-batch, a torch integer vector of length n. The value of each element can be 0 or 1. 
* y_predict:  the predicted labels of a mini-batch of test images, a torch integer vector of length n. y_predict[i] represents the predicted label (0 or 1) on the i-th test sample in the mini-batch. 
* h:  the height of each input image, an integer scalar. 
* h1:  the height of the feature map after using the first convolutional layer and maxpooling, h1 = (h - s1 +1)/2, an integer scalar. 
* h2:  the height of the feature map after using the second convolutional layer and maxpooling, h2 = (h1 - s2 +1)/2, an integer scalar. 
* w:  the width of each input image, an integer scalar. 
* w1:  the width of the feature map after using the first convolutional layer and maxpooling, w1 = (w - s1 +1)/2, an integer scalar. 
* w2:  the width of the feature map after using the second convolutional layer and maxpooling, w2 = (w1 - s2 +1)/2, an integer scalar. 
* c:  the number of color channels in each input image, an integer scalar. 
* c1:  the number of filters in the first convolutional layer, an integer scalar. 
* c2:  the number of filters in the second convolutional layer, an integer scalar. 
* s1:  the size of filters (height = width = s1) in the first convolutional layer of CNN, an integer scalar. 
* s2:  the size of filters (height = width = s2) in the second convolutional layer of CNN, an integer scalar. 
* W1:  the weights of the filters in the first convolutional layer of CNN, a float torch Tensor of shape (c, s1, s1). 
* W2:  the weights of the filters in the second convolutional layer of CNN, a float torch Tensor of shape (c1, s2, s2). 
* W3:  the weights of fully connected layer (3rd layer) of CNN, which is used to predict the binary class label of each image, a float torch vector of length (n_flat_features). 
* b1:  the biases of filters in the first convolutional layer of CNN, a float torch vector of length c1. 
* b2:  the biases of filters in the second convolutional layer of CNN, a float torch vector of length c2. 
* b3:  the bias value of fully connected layer (3rd layer) of CNN, a float torch scalar. 
* z1:  the linear logits of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h- s1 + 1, w - s1 + 1). 
* z2:  the linear logits of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1). 
* z3:  the linear logit of the fully-connected layer of CNN on a mini-batch of data samples, a float torch vector of length (n). 
* a1:  the ReLU activations of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h - s1 + 1, w - s1 + 1). 
* a2:  the ReLU activations of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h1 - s2 + 1, w1 - s2 + 1 ). 
* p1:  the pooled activations (using max pooling) of the first convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c1, h1, w1). 
* p2:  the pooled activations (using max pooling) of the second convolutional layer of CNN on a mini-batch of data samples, a float torch tensor of shape (n, c2, h2, w2). 
* n_flat_features:  the number of input features to the fully connected layer after flattening the outputs of the last convolutional layer on a mini-batch of images,  an integer scalar. n_flat_features = c2 * h2 * w2. 
* f:  the input features to the fully connected layer after flattening the outputs of the second convolutional layer on a mini-batch of images, a float torch tensor of shape (n, n_flat_features ). 
* L:  the average of the binary cross entropy losses on a mini-batch of training images, a torch float scalar. 
* data_loader:  the PyTorch loader of a dataset. 
* alpha:  the step-size parameter for gradient descent, a float scalar. 
* n_epoch:  the number of passes to go through the training dataset in the training process, an integer scalar. 
* optimizer:  a PyTorch optimizer (such as SGD, ADAM, RMSProp) to handle the gradient descent for all the parameters in the model (W1, b1, W2, b2, W3 and b3). 
* s:  the size (height = width = s) of the filters in a convolutional layer, an integer scalar. 
* x_a:  one input image with 1 input/color channel with hight h and width w of pixels, a float torch tensor of shape (h, w). 
* W_a:  the weights of 1 convolutional filter with 1 color/input channel,  a float torch matrix of shape (s, s). 
* b_a:  the bias of 1 convolutional filter with 1 color/input channel,  a float torch scalar. 
* z_a:  the linear logits of the convolutional layer with 1 filter and 1 channel on one image,  a float torch matrix of shape (h-s+1, w-s+1). 
* x_b:  one input image with c color/input channels with hight h and width w of pixels, a float torch tensor of shape (c, h, w). Here x_b[i] represents i-th color/input channel of the color image. 
* W_b:  the weights of 1 convolutional filter with c color/input channels,  a float torch tensor of shape (c, s, s). Here W_b[i] represents the weights of the filter on the i-th input/color channel. 
* b_b:  the bias of 1 convolutional filter with c color/input channels,  a float torch scalar. 
* z_b:  the linear logits of the convolutional layer with 1 filter and c channels on one image,  a float torch matrix of shape (h-s+1, w-s+1). 
* x_c:  one input image with c color channels with hight h and width w of pixels, a float torch tensor of shape (c, h, w). 
* W_c:  the weights of c1 convolutional filters, where each filter has c color/input channels,  a float torch tensor of shape (c1, c, s, s). Here W_c[i] represents the weights of i-th convolutional filter. 
* b_c:  the biases of c1 convolutional filters,  a float torch vector of length c1. Here b_c[i] represents the bias of the i-th convolutional filter. 
* z_c:  the linear logits of the c1 convolutional filters on one image, a float torch tensor of shape (c1, h-s+1, w-s+1). Here z_c[i] represents the linear logits the i-th convolutional filter. 

'''
#--------------------------------------------