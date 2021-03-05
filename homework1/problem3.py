import numpy as np

# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: Support Vector Machine (with non-linear kernels) 
    In this problem, you will implement the SVM using SMO method. 
    A list of all variables being used in this problem is provided at the end of this file.
'''

#---------------------------------------------------
'''
    Compute the linear kernel between two data instances x1 and x2. 
    Inputs: 
        * x1: the feature vector of one data sample, a numpy vector of length p.
        * x2: the feature vector of another data sample, a numpy vector of length p.
    Outputs: 
        * k: the kernel similarity between the data instances x1 and x2, a float scalar.
    --------------------------------------
    Example:
        suppose we have two dimensional feature space (p=2).
        x1 = [ 1, 2]
        X2 = [ 3, 4]
        So now we want to compute the linear kernel similarity between the two instances x1 and x2 
        k = 1*3 + 2*4 = 11 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def linear_kernel(x1, x2):
    #########################################
    ## INSERT YOUR CODE HERE
    k = np.dot(x1, x2)
    #########################################
    return k
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_linear_kernel
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the linear kernel matrix between data instances in X1 and X2. 
    Inputs: 
        * X1: the feature matrix of one data set of samples, a numpy matrix of shape n1 by p.
        * X2: the feature matrix of another data set of samples, a numpy matrix of shape n2 by p.
    Outputs: 
        * K: the kernel matrix between the data instances in X1 and X2, a numpy float matrix of shape n1 by n2. The i,j-th element is the kernel between the i-th instance in X1, and j-th instance in X2.
    --------------------------------------
    Example:
        suppose we have two dimensional feature space (p=2).
        If X1 is a dataset with 3 samples, and X2 is another dataset with 2 samples
        X1 = [[ 1, 2],
              [ 2, 4],
              [ 3, 6]]
        X2 = [[ 1, 1],
              [-1,-1]]
        So now we want to compute the linear kernel between the two datasets (X1, and X2).
        The result kernel matrix K should be a matrix of shape 3 by 2, because  X1 has 3 samples and X2 has 2 samples.
        K[i,j] is the linear kernel between the i-th instance in X1 and j-th instance in X2.
        K[0,0] - is computed as the linear_kernel between  X1[0] and X2[0]: 
                which is the dot product between two vectors [1,2] and [1,1]: 1*1 + 2*1 = 3
        K[0,1] - is computed as the linear_kernel between  X1[0] and X2[1]: 
                which is the dot product between two vectors [1,2] and [-1,-1]: 1*(-1) + 2*(-1) = -3
        So the linear kernel matrix for X1 and X2 is:
        K = [[ 3,-3],
             [ 6,-6],
             [ 9,-9]] 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def linear_kernel_matrix(X1, X2):
    #########################################
    ## INSERT YOUR CODE HERE
    K = np.array([[linear_kernel(x1, x2) for x2 in X2] for x1 in X1])
    #########################################
    return K
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_linear_kernel_matrix
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the polynomial kernel between two data instances x1 and x2. 
    Inputs: 
        * x1: the feature vector of one data sample, a numpy vector of length p.
        * x2: the feature vector of another data sample, a numpy vector of length p.
        * d: the degree of polynomials in polynomial kernel, an integer scalar.
    Outputs: 
        * k: the kernel similarity between the data instances x1 and x2, a float scalar.
    --------------------------------------
    Example:
        suppose we have two dimensional feature space (p=2).
        x1 = [ 1, 2]
        X2 = [ 3, 4]
        So now we want to compute the polynomial kernel (degree=2) similarity between the two instances x1 and x2
        k =(1*3 + 2*4 +1)^2 = 12^2 = 144 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def polynomial_kernel(x1, x2, d=2):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return k
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_polynomial_kernel
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the polynomial kernel matrix between data instances in X1 and X2. 
    Inputs: 
        * X1: the feature matrix of one data set of samples, a numpy matrix of shape n1 by p.
        * X2: the feature matrix of another data set of samples, a numpy matrix of shape n2 by p.
        * d: the degree of polynomials in polynomial kernel, an integer scalar.
    Outputs: 
        * K: the kernel matrix between the data instances in X1 and X2, a numpy float matrix of shape n1 by n2. The i,j-th element is the kernel between the i-th instance in X1, and j-th instance in X2.
    --------------------------------------
    Example:
        suppose we have two dimensional feature space (p=2).
        If X1 is a dataset with 3 samples, and X2 is another dataset with 2 samples
        X1 = [[ 1, 2],
              [ 2, 4],
              [ 3, 6]]
        X2 = [[ 1, 1],
              [-1,-1]]
        So now we want to compute the polynomial kernel (degree d=2) between the two datasets (X1, and X2).
        The result kernel matrix K should be a matrix of shape 3 by 2, because  X1 has 3 samples and X2 has 2 samples.
        K[i,j] is the polynomial kernel (degree d=2) between the i-th instance in X1 and j-th instance in X2.
        K[0,0] - is computed as the polynomial kernel (degree d=2) between  X1[0] and X2[0]: 
                which is the polynomial kernel (degree d=2) between two vectors [1,2] and [1,1]: (1*1 + 2*1 +1)^2 = (1+2+1)^2 = 4^2 = 4*4 = 16
        K[0,1] - is computed as the polynomial kernel (degree d=2) between  X1[0] and X2[1]: 
                which is the polynomial kernel (degree d=2) between two vectors [1,2] and [-1,-1]: (1*(-1) + 2*(-1) +1)^2 = (-2)^2 = 4
        So the linear kernel matrix for X1 and X2 is:
        K = [[  16,  4 ],
             [  49, 25 ],
             [ 100, 64 ]] 
    --------------------------------------
    Hints: 
        * You could use np.power(A, d) to compute the d-th power of each element in matrix A. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def polynomial_kernel_matrix(X1, X2, d=2):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return K
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_polynomial_kernel_matrix
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the Gaussian (RBF) kernel between two data instances x1 and x2. 
    Inputs: 
        * x1: the feature vector of one data sample, a numpy vector of length p.
        * x2: the feature vector of another data sample, a numpy vector of length p.
        * sigma: the standard deviation of Gaussian kernel, a float scalar.
    Outputs: 
        * k: the kernel similarity between the data instances x1 and x2, a float scalar.
    --------------------------------------
    Example:
        suppose we have two dimensional feature space (p=2).
        x1 = [ 1, 2]
        X2 = [ 3, 4]
        So now we want to compute the Gaussian kernel (sigma=5) similarity between the two instances x1 and x2
        k =  exp[ - ((1-3)^2 + (2-4)^2) / (2*5*5) ] = 0.8521 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def gaussian_kernel(x1, x2, sigma=1.0):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return k
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_gaussian_kernel
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the Gaussian (RBF) kernel matrix between data instances in X1 and X2. 
    Inputs: 
        * X1: the feature matrix of one data set of samples, a numpy matrix of shape n1 by p.
        * X2: the feature matrix of another data set of samples, a numpy matrix of shape n2 by p.
        * sigma: the standard deviation of Gaussian kernel, a float scalar.
    Outputs: 
        * K: the kernel matrix between the data instances in X1 and X2, a numpy float matrix of shape n1 by n2. The i,j-th element is the kernel between the i-th instance in X1, and j-th instance in X2.
    --------------------------------------
    Example:
        suppose we have two dimensional feature space (p=2).
        If X1 is a dataset with 3 samples, and X2 is another dataset with 2 samples
        X1 = [[ 1, 0],
              [ 0, 1],
              [ 1, 1]]
        X2 = [[ 1, 1],
              [-1,-1]]
        So now we want to compute the Gaussian kernel between the two datasets (X1, and X2).
        The result kernel matrix K should be a matrix of shape 3 by 2, because  X1 has 3 samples and X2 has 2 samples.
        K[i,j] is the Gaussian kernel between the i-th instance in X1 and j-th instance in X2.
        K[0,0] - is computed as the Gaussian kernel (sigma = 1) between  X1[0] and X2[0],
                which is the Gaussian kernel (sigma = 1) between two vectors [0,1] and [1,1]: exp[ -( (0-1)^2 + (1-1)^2 )  / (2* 1^2 ) ] = exp[ -1 / 2 ] = 0.60653066
        So the linear kernel matrix for X1 and X2 is
        K = [[0.60653066,0.082085  ],
             [0.60653066,0.082085  ],
             [1.        ,0.01831564]] 
    --------------------------------------
    Hints: 
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def gaussian_kernel_matrix(X1, X2, sigma=1.0):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return K
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_gaussian_kernel_matrix
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Given a test instance x, compute the output of the SVM model f(x) = sum_i alpha_i * y_i * K(x_i, x) + b. 
    Inputs: 
        * Kt: the kernel similarity between the test instance and all training instances, a numpy vector of length n. Kt[i] represents the kernel similarity between the test instance (x) and the i-th training instance: K(x, x_i ).
        * a: the alpha values on the training instances, a numpy float vector of length n.
        * y: the labels of all training data samples, a numpy vector of length n. If the i-th instance is predicted as positive, y[i]= 1, otherwise -1.
        * b: the bias of the SVM model, a float scalar.
    Outputs: 
        * fx: f(x), the output of SVM model on the test instance (x), a float scalar.
    --------------------------------------
    Example:
        Suppose we have 3 samples in the training set.
        Suppose the kernel similarity between the test sample (x) and all training instances are:
            Kt = [ 1, 2, 3]
        where Kt[i] represents the kernel similarity between the test instance (x) and the i-th training instance.
        Suppose the labels of the 3 training samples are y = [1, -1, 1]
        and the alpha values on these 3 training instances are alpha = [0, 1, 1]. 
        The bias b = 4.
        We want to compute the output of the SVM model on the test instance: f(x) 
        f(x) =  alpha[0]*y[0]*Kt[0] + alpha[1]*y[1]*Kt[1] + alpha[2]*y[2]*Kt[2] + b
                              = 0*1*1 + 1*(-1)*2 + 1*1*3 + 4 = 5 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_fx(Kt, a, y, b):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return fx
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_fx
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Predict the labels of testing instances.  For each test instance (x), we could use f(x) = sum_i alpha_i * y_i * K(x_i, x) + b to compute the score. Here x_i is the i-th training instance, y_i is the label the i-th training instance.. 
    Inputs: 
        * K_test_train: the kernel matrix between the testing instances and training instances, a numpy matrix of shape n_test by n_train.  Here n_test is the number of testing instances.  n_train is the number of training instances.  K[i,j] represents the kernel similarity between the i-th test instance and the j-th training instance.
        * a: the alpha values on the training instances, a numpy float vector of length n.
        * y: the labels of all training data samples, a numpy vector of length n. If the i-th instance is predicted as positive, y[i]= 1, otherwise -1.
        * b: the bias of the SVM model, a float scalar.
    Outputs: 
        * y_test: the labels of all testing instances, a numpy vector of length n_test. If the output of SVM model on i-th instance f(xi) is predicted as non-negative, y[i]= 1, otherwise -1.
    --------------------------------------
    Example:
        Suppose we have 3 samples in the training set, and 2 samples in the test set.
        Our goal is to predict the labels of the 2 test samples.
        Suppose the kernel between test samples and training samples is
            K = [[ 1, 2, 4],
                 [ 6, 3, 5]]
        where K[i,j] represents the kernel similarity between the i-th test instance and the j-th training instance.
        Suppose the labels of the 3 training samples are y = [1, -1, 1]
        and the alpha values on these 3 training samples are alpha = [0, 1, 1]. The bias b = 0
        We want to predict the label of the 2 test instances. 
        For the first test instance (x1), the label can be predicted as:
        We first compute f(x1) =  alpha[0]*y[0]*K[0,0] + alpha[1]*y[1]*K[0,1] + alpha[2]*y[2]*K[0,2] + b
                              = 0*1*1 + 1*(-1)*2 + 1*1*4 + 0 = 2
        Similarly we can compute on the second test instance (x2):
        f(x2) =  alpha[0]*y[0]*K[1,0] + alpha[1]*y[1]*K[1,1] + alpha[2]*y[2]*K[1,2] + b
              = 0*1*6 + 1*(-1)*4 + 1*1*5 + 0 = 2
        So the label y_test is predicted as [1,1], where both are predicted with positive label. 
    --------------------------------------
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def predict(K_test_train, a, y, b):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return y_test
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_predict
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the clipping range of a[i] when pairing with a[j]. 
    Inputs: 
        * ai: the alpha on the i-th training instance, a float scalar, 0<= a_i <= C.
        * yi: the label of the i-th instance, a float scalar of value -1 or 1.
        * aj: the alpha value on the j-th training instance, a float scalar, value 0<= a_j <= C.
        * yj: the label of the j-th instance, a float scalar of value -1 or 1.
        * C: the weight of the hinge loss.
    Outputs: 
        * H: the upper-bound of the range of ai, a float scalar, between 0 and C.
        * L: the lower-bound of the range of ai, a float scalar, between 0 and C.
    Hints: 
        * This problem can be solved using 6 line(s) of code.
'''
#---------------------
def compute_HL(ai, yi, aj, yj, C=1.0):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return H, L
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_HL
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the error on the i-th instance: Ei = f(x[i]) - y[i]. 
    Inputs: 
        * Ki: the i-th row of kernel matrix between the training instances, a numpy vector of length n_train. Here n_train is the number of training instances.
        * a: the alpha values on the training instances, a numpy float vector of length n.
        * y: the labels of all training data samples, a numpy vector of length n. If the i-th instance is predicted as positive, y[i]= 1, otherwise -1.
        * b: the bias of the SVM model, a float scalar.
        * i: the index of the i-th instance, an integer scalar.
    Outputs: 
        * Ei: the error on the i-th training instance, a float scalar.
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_E(Ki, a, y, b, i):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return Ei
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_E
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Compute the eta on the (i,j) pair of instances: eta = 2* Kij - Kii - Kjj. 
    Inputs: 
        * Kii: the kernel between the i,i-th instances, a float scalar.
        * Kjj: the kernel between the j,j-th instances, a float scalar.
        * Kij: the kernel between the i,j-th instances, a float scalar.
    Outputs: 
        * eta: the eta of the (i,j)-th pair of instances, a float scalar.
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_eta(Kii, Kjj, Kij):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return eta
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_eta
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Update the a[i] when considering the (i,j) pair of instances. 
    Inputs: 
        * Ei: the error on the i-th training instance, a float scalar.
        * Ej: the error of the j-th instance, a float scalar.
        * eta: the eta of the (i,j)-th pair of instances, a float scalar.
        * ai: the alpha on the i-th training instance, a float scalar, 0<= a_i <= C.
        * yi: the label of the i-th instance, a float scalar of value -1 or 1.
        * H: the upper-bound of the range of ai, a float scalar, between 0 and C.
        * L: the lower-bound of the range of ai, a float scalar, between 0 and C.
    Outputs: 
        * ai_new: the updated alpha on the i-th instance, a float scalar, value 0<= ai_new <= C.
    Hints: 
        * This problem can be solved using 9 line(s) of code.
'''
#---------------------
def update_ai(Ei, Ej, eta, ai, yi, H, L):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return ai_new
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_update_ai
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Update a[j] when considering the (i,j) pair of instances. 
    Inputs: 
        * aj: the alpha value on the j-th training instance, a float scalar, value 0<= a_j <= C.
        * ai: the alpha on the i-th training instance, a float scalar, 0<= a_i <= C.
        * ai_new: the updated alpha on the i-th instance, a float scalar, value 0<= ai_new <= C.
        * yi: the label of the i-th instance, a float scalar of value -1 or 1.
        * yj: the label of the j-th instance, a float scalar of value -1 or 1.
    Outputs: 
        * aj_new: the updated alpha of the j-th instance, a float scalar, value 0<= aj_new <= C.
    Hints: 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def update_aj(aj, ai, ai_new, yi, yj):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return aj_new
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_update_aj
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Update the bias term. 
    Inputs: 
        * b: the bias of the SVM model, a float scalar.
        * ai_new: the updated alpha on the i-th instance, a float scalar, value 0<= ai_new <= C.
        * aj_new: the updated alpha of the j-th instance, a float scalar, value 0<= aj_new <= C.
        * ai: the alpha on the i-th training instance, a float scalar, 0<= a_i <= C.
        * aj: the alpha value on the j-th training instance, a float scalar, value 0<= a_j <= C.
        * yi: the label of the i-th instance, a float scalar of value -1 or 1.
        * yj: the label of the j-th instance, a float scalar of value -1 or 1.
        * Ei: the error on the i-th training instance, a float scalar.
        * Ej: the error of the j-th instance, a float scalar.
        * Kii: the kernel between the i,i-th instances, a float scalar.
        * Kjj: the kernel between the j,j-th instances, a float scalar.
        * Kij: the kernel between the i,j-th instances, a float scalar.
        * C: the weight of the hinge loss.
    Outputs: 
        * b: the bias of the SVM model, a float scalar.
    Hints: 
        * This problem can be solved using 8 line(s) of code.
'''
#---------------------
def update_b(b, ai_new, aj_new, ai, aj, yi, yj, Ei, Ej, Kii, Kjj, Kij, C=1.0):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return b
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_update_b
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Train the SVM model using simplified SMO algorithm. 
    Inputs: 
        * K_train: the kernel matrix between the training instances, a numpy float matrix of shape n by n.
        * y: the labels of all training data samples, a numpy vector of length n. If the i-th instance is predicted as positive, y[i]= 1, otherwise -1.
        * C: the weight of the hinge loss.
        * n_epoch: the number of rounds to iterate through all training example pairs.
    Outputs: 
        * a: the alpha values on the training instances, a numpy float vector of length n.
        * b: the bias of the SVM model, a float scalar.
    Hints: 
        * Step 1 compute the bounds of ai (H, L). 
        * Step 2 if H==L, no change is needed, skip to next j. 
        * Step 3 compute Ei and Ej. 
        * Step 4 compute eta. 
        * Step 5 update ai, aj, and b. 
        * This problem can be solved using 11 line(s) of code.
'''
#---------------------
def train(K_train, y, C=1.0, n_epoch=10):
    n = K_train.shape[0] # number of training instances
    a,b = np.zeros(n), 0.  # initialize alpha and b
    for _ in range(n_epoch): # iterate n_epoch passes through all training sample pairs
        indices_i = np.random.permutation(n) # shuffle the indices of all instances
        indices_j = np.random.permutation(n) # shuffle the indices of all instances
        for i in indices_i:
            for j in indices_j:
                # train SVM on a random (i,j) pair of training instances
                ai = a[i]
                aj = a[j]
                yi = y[i]
                yj = y[j]
                #########################################
                ## INSERT YOUR CODE HERE
    
                #########################################
    return a, b
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_train
        ---------------------------------------------------
    '''
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_svm_linear
        ---------------------------------------------------
    '''''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_svm_poly
        ---------------------------------------------------
    '''''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_svm_RBF
        ---------------------------------------------------
    '''
    

#--------------------------------------------

''' 
    TEST problem 3: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 3 (60 points in total)--------------------- ... ok
        * (2 point) linear kernel ... ok 
        * (2 point) linear kernel matrix ... ok 
        * (2 point) polynomial kernel ... ok 
        * (2 point) polynomial kernel matrix ... ok 
        * (2 point) Gaussian kernel ... ok 
        * (1 point) Gaussian kernel matrix ... ok 
        * (2 points) compute_fx ... ok 
        * (3 points) predict ... ok 
        * (5 points) compute_HL ... ok 
        * (5 points) compute_E ... ok 
        * (5 points) compute_eta ... ok 
        * (5 points) update_ai ... ok 
        * (5 points) update_aj ... ok 
        * (5 point) update_b ... ok 
        * (5 points) train ... ok 
        * (3 point) SVM (linear kernel) ... ok 
        * (3 point) SVM (polynomial kernel) ... ok 
        * (3 point) SVM (Gaussian kernel) ... ok 
        ----------------------------------------------------------------------
        Ran 18 tests in 0.959s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* p:  the number of dimensions in the feature space. 
* n:  the number of data instances in the training set. 
* x1:  the feature vector of one data sample, a numpy vector of length p. 
* x2:  the feature vector of another data sample, a numpy vector of length p. 
* k:  the kernel similarity between the data instances x1 and x2, a float scalar. 
* X1:  the feature matrix of one data set of samples, a numpy matrix of shape n1 by p. 
* X2:  the feature matrix of another data set of samples, a numpy matrix of shape n2 by p. 
* K:  the kernel matrix between the data instances in X1 and X2, a numpy float matrix of shape n1 by n2. The i,j-th element is the kernel between the i-th instance in X1, and j-th instance in X2. 
* d:  the degree of polynomials in polynomial kernel, an integer scalar. 
* sigma:  the standard deviation of Gaussian kernel, a float scalar. 
* Kt:  the kernel similarity between the test instance and all training instances, a numpy vector of length n. Kt[i] represents the kernel similarity between the test instance (x) and the i-th training instance: K(x, x_i ). 
* a:  the alpha values on the training instances, a numpy float vector of length n. 
* b:  the bias of the SVM model, a float scalar. 
* X:  the feature matrix of all training data samples, a numpy matrix of shape n by p. 
* y:  the labels of all training data samples, a numpy vector of length n. If the i-th instance is predicted as positive, y[i]= 1, otherwise -1. 
* fx:  f(x), the output of SVM model on the test instance (x), a float scalar. 
* K_test_train:  the kernel matrix between the testing instances and training instances, a numpy matrix of shape n_test by n_train.  Here n_test is the number of testing instances.  n_train is the number of training instances.  K[i,j] represents the kernel similarity between the i-th test instance and the j-th training instance. 
* K_train:  the kernel matrix between the training instances, a numpy float matrix of shape n by n. 
* y_test:  the labels of all testing instances, a numpy vector of length n_test. If the output of SVM model on i-th instance f(xi) is predicted as non-negative, y[i]= 1, otherwise -1. 
* ai:  the alpha on the i-th training instance, a float scalar, 0<= a_i <= C. 
* aj:  the alpha value on the j-th training instance, a float scalar, value 0<= a_j <= C. 
* yi:  the label of the i-th instance, a float scalar of value -1 or 1. 
* yj:  the label of the j-th instance, a float scalar of value -1 or 1. 
* C:  the weight of the hinge loss. 
* H:  the upper-bound of the range of ai, a float scalar, between 0 and C. 
* L:  the lower-bound of the range of ai, a float scalar, between 0 and C. 
* Ki:  the i-th row of kernel matrix between the training instances, a numpy vector of length n_train. Here n_train is the number of training instances. 
* i:  the index of the i-th instance, an integer scalar. 
* Ei:  the error on the i-th training instance, a float scalar. 
* Ej:  the error of the j-th instance, a float scalar. 
* Kii:  the kernel between the i,i-th instances, a float scalar. 
* Kjj:  the kernel between the j,j-th instances, a float scalar. 
* Kij:  the kernel between the i,j-th instances, a float scalar. 
* eta:  the eta of the (i,j)-th pair of instances, a float scalar. 
* ai_new:  the updated alpha on the i-th instance, a float scalar, value 0<= ai_new <= C. 
* aj_new:  the updated alpha of the j-th instance, a float scalar, value 0<= aj_new <= C. 
* n_epoch:  the number of rounds to iterate through all training example pairs. 

'''
#--------------------------------------------