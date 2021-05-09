import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Probabilistic Latent Semantic Analysis (40 points)
    In this problem, we will implement the probabilistic latent semantic analysis (PLSA) and use PLSA to perform topic analysis on text documents
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Update Phi with 1 word in 1 document) Given a word ID (w) in the vocabulary, the current model parameters (Beta) and the variational parameters (theta_d) on the document, update the variational parameter phi_w based upon one observed word (w) in the text document. 
    ---- Inputs: --------
        * w: the ID of a word in the vocabulary, an integer scalar, which can be 0,1, ..., or v-1.
        * Beta: the word probability distribution for all topics, a float numpy matrix of shape (c, v); Beta[i,j] represents the probability P(W=j | T =i), which is the conditional probability of the j-th word in the vocabulary given the i-th topic.
        * theta_d: the topic mixture in one text document (d), a numpy float vector of length c. Here theta_d[i] is the probability P(T=i | D=d), which is the conditional probability of the i-th topic given the document.
    ---- Outputs: --------
        * phi_w: the variational parameter (phi) of a categorical distribution to generate the topic (z) for a word (ID=w) in a document, a numpy float vector of length c. phi_w[i] represents the probability of generating the i-th topic for word (w) in the document.
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_phi_w(w, Beta: np.ndarray, theta_d):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    denominator = np.sum([theta_d[i] * Beta[i, w] for i in range(Beta.shape[0])])
    phi_w = (theta_d * Beta[:, w]) / denominator
    #########################################
    return phi_w
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_phi_w
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_phi_w
        --- OR ---- 
        python -m nose -v test2.py:test_compute_phi_w
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Update Phi with all the words in 1 document) Given the word counts in a document, the current model parameters (Beta and gamma_d) on the document, update the variational parameter Phi in the document. 
    ---- Inputs: --------
        * Beta: the word probability distribution for all topics, a float numpy matrix of shape (c, v); Beta[i,j] represents the probability P(W=j | T =i), which is the conditional probability of the j-th word in the vocabulary given the i-th topic.
        * theta_d: the topic mixture in one text document (d), a numpy float vector of length c. Here theta_d[i] is the probability P(T=i | D=d), which is the conditional probability of the i-th topic given the document.
    ---- Outputs: --------
        * phi_d: the variational parameters (phi) of a list of categorical distributions to generate the topic (z) in all words of a document, a numpy float matrix of shape v by c. Each row represents the parameters of a categorical distribution to generate different topics in one word in the document; phi_d[i,j] represents the probability of generating the j-th topic for the word ID=i in the document.
    ---- Hints: --------
        * When computing the sum() function on a numpy array, if you use parameter (keepdims=True), the axis being summed will be kept. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_phi_d(Beta, theta_d):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    return phi_d
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_phi_d
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_phi_d
        --- OR ---- 
        python -m nose -v test2.py:test_compute_phi_d
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (E-step: Update Phi with all the words in all documents) Given the model parameters Beta and Theta of the PLSA, compute the distribution of topics by computing the probabilities of different topics in each word and each document. 
    ---- Inputs: --------
        * Theta: the topic mixture in each text document, a float numpy matrix of shape (n, c); Theta[i,j] represents the probability P(T=j | D=i), which is the conditional probability of the j-th topic given the i-th document in the dataset.
        * Beta: the word probability distribution for all topics, a float numpy matrix of shape (c, v); Beta[i,j] represents the probability P(W=j | T =i), which is the conditional probability of the j-th word in the vocabulary given the i-th topic.
    ---- Outputs: --------
        * Phi: the topic probability matrix P(T|W,D) for all documents, a float numpy tensor of shape (n, v, c); Phi[i,j,k] represents the probability P(T=k | W = j,D=i), which is the conditional probability of the k-th topic on the j-th word in the vocabulary in the i-th document.
    ---- Hints: --------
        * You could use einsum() function in numpy to solve this problem efficiently. 
        * When computing the sum of a numpy array, if you use (keepdims=True), the axis being summed will be kept. 
        * You could use the element-wise operations of two numpy arrays with broadcasting to solve this problem efficiently. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_Phi(Theta, Beta):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    return Phi
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_Phi
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_Phi
        --- OR ---- 
        python -m nose -v test2.py:test_compute_Phi
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Update Theta in 1 document) Given the variational parameters (phi_d) and the word counts (C_d) in a document, compute the model parameter theta_d on the document (d). 
    ---- Inputs: --------
        * C_d: word frequency counts in a text document, an integer numpy vector of length v; C[i] represents how many times the i-th word in the vocabulary has been used in the document.
        * phi_d: the variational parameters (phi) of a list of categorical distributions to generate the topic (z) in all words of a document, a numpy float matrix of shape v by c. Each row represents the parameters of a categorical distribution to generate different topics in one word in the document; phi_d[i,j] represents the probability of generating the j-th topic for the word ID=i in the document.
    ---- Outputs: --------
        * theta_d: the topic mixture in one text document (d), a numpy float vector of length c. Here theta_d[i] is the probability P(T=i | D=d), which is the conditional probability of the i-th topic given the document.
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_theta_d(C_d, phi_d):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    return theta_d
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_theta_d
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_theta_d
        --- OR ---- 
        python -m nose -v test2.py:test_compute_theta_d
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (M-step: Update Theta on all documents) Given a collection of text documents, represented as word-frequency format (C), and inferred topic distributions (Phi), please compute the maximum likelihood solution of the parameter Theta: the topic mixtures of each document, i.e., the conditional probabilities of P(T|D). 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * Phi: the topic probability matrix P(T|W,D) for all documents, a float numpy tensor of shape (n, v, c); Phi[i,j,k] represents the probability P(T=k | W = j,D=i), which is the conditional probability of the k-th topic on the j-th word in the vocabulary in the i-th document.
    ---- Outputs: --------
        * Theta: the topic mixture in each text document, a float numpy matrix of shape (n, c); Theta[i,j] represents the probability P(T=j | D=i), which is the conditional probability of the j-th topic given the i-th document in the dataset.
    ---- Hints: --------
        * You could use einsum() function in numpy to solve this problem efficiently. 
        * When computing the sum of a numpy array, if you use (keepdims=True), the axis being summed will be kept. 
        * You could use the element-wise operations of two numpy arrays with broadcasting to solve this problem efficiently. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_Theta(C, Phi):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    return Theta
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_Theta
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_Theta
        --- OR ---- 
        python -m nose -v test2.py:test_compute_Theta
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Update Parameter Beta on the t-th topic) Given a collection of text documents, represented as word-frequency format (C), and inferred topic distributions (Phi), please compute the maximum likelihood solution of the parameter Beta on the t-th topic: the word distribution of the t-th topic, i.e., the conditional probabilities of P(W|T=t). 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * phi_t: the variational parameters (phi) of a list of categorical distributions to generate the t-th topic in all words of all document, a numpy float matrix of shape n by v. phi_d[i,j] represents the probability of generating the t-th topic for the word ID=j in the i-th document.
    ---- Outputs: --------
        * beta_t: the word probability distribution for one topic (t), a float numpy vector of length v; beta_t[i] represents the probability P(W=i | T =t), which is the conditional probability of generating the i-th word in the vocabulary in the topic (t).
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_beta_t(C, phi_t):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    return beta_t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_beta_t
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_beta_t
        --- OR ---- 
        python -m nose -v test2.py:test_compute_beta_t
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (M-step: Computing word distribution of each topic) Given a collection of text documents, represented as word-frequency format (C), and inferred topic distributions (Phi), please compute the maximum likelihood solution of the parameter Beta: the word distribution of each topic, i.e., the conditional probabilities of P(W|T). 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * Phi: the topic probability matrix P(T|W,D) for all documents, a float numpy tensor of shape (n, v, c); Phi[i,j,k] represents the probability P(T=k | W = j,D=i), which is the conditional probability of the k-th topic on the j-th word in the vocabulary in the i-th document.
    ---- Outputs: --------
        * Beta: the word probability distribution for all topics, a float numpy matrix of shape (c, v); Beta[i,j] represents the probability P(W=j | T =i), which is the conditional probability of the j-th word in the vocabulary given the i-th topic.
    ---- Hints: --------
        * You could use einsum() function in numpy to solve this problem efficiently. 
        * When computing the sum of a numpy array, if you use (keepdims=True), the axis being summed will be kept. 
        * You could use the element-wise operations of two numpy arrays with broadcasting to solve this problem efficiently. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_Beta(C, Phi):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    
    #########################################
    return Beta
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_compute_Beta
        --- OR ---- 
        python3 -m nose -v test2.py:test_compute_Beta
        --- OR ---- 
        python -m nose -v test2.py:test_compute_Beta
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (PLSA Method) Given a collection of text documents, represented as word-frequency format (C), and inferred topic distributions (Phi), please use EM algorithm to estimate the parameters Theta and Beta, i.e., the conditional probabilities of P(T|D) and P(W|T). 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * c: the number of possible topics (categories) in the model, an integer scalar.
        * n_iter: the number of iterations in the EM algorithm, an integer scalar.
    ---- Outputs: --------
        * Theta: the topic mixture in each text document, a float numpy matrix of shape (n, c); Theta[i,j] represents the probability P(T=j | D=i), which is the conditional probability of the j-th topic given the i-th document in the dataset.
        * Beta: the word probability distribution for all topics, a float numpy matrix of shape (c, v); Beta[i,j] represents the probability P(W=j | T =i), which is the conditional probability of the j-th word in the vocabulary given the i-th topic.
        * Phi: the topic probability matrix P(T|W,D) for all documents, a float numpy tensor of shape (n, v, c); Phi[i,j,k] represents the probability P(T=k | W = j,D=i), which is the conditional probability of the k-th topic on the j-th word in the vocabulary in the i-th document.
    ---- Hints: --------
        * Step 1 (E step): Compute Phi based upon the current values of Theta and Beta. 
        * Step 2 (M step): update the parameters Theta and Beta based upon the new values of Phi. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def PLSA(C, c, n_iter=30):
    n,v = C.shape # n: the number documents, v: the number of words in the vocabulary
    Beta = np.ones((c,v))/v # initialize Beta as uniform distribution
    Theta = np.ones((n,c)) + 0.2*np.random.rand(n,c) # initialize Theta as almost uniform distribution with small noise for symmetry breaking
    Theta = Theta/Theta.sum(axis=1,keepdims=True)
    for i in range(n_iter): # iterate multiple times
        pass # no operation (you can ignore this line)
        #########################################
        ## INSERT YOUR CODE HERE (5 points)
    
        #########################################
    return Theta, Beta, Phi
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_PLSA
        --- OR ---- 
        python3 -m nose -v test2.py:test_PLSA
        --- OR ---- 
        python -m nose -v test2.py:test_PLSA
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
        ----------- Problem 2 (40 points in total)--------------------- ... ok
        * (5 points) compute_phi_w ... ok
        * (5 points) compute_phi_d ... ok
        * (5 points) compute_Phi ... ok
        * (5 points) compute_theta_d ... ok
        * (5 points) compute_Theta ... ok
        * (5 points) compute_beta_t ... ok
        * (5 points) compute_Beta ... ok
        * (5 points) PLSA ... ok
        ----------------------------------------------------------------------
        Ran 8 tests in 0.586s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of text documents in the dataset, an integer scalar. 
* v:  the number of possible words in the vocabulary, an integer scalar. 
* c:  the number of possible topics (categories) in the model, an integer scalar. 
* w:  the ID of a word in the vocabulary, an integer scalar, which can be 0,1, ..., or v-1. 
* C_d:  word frequency counts in a text document, an integer numpy vector of length v; C[i] represents how many times the i-th word in the vocabulary has been used in the document. 
* C:  word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document. 
* phi_w:  the variational parameter (phi) of a categorical distribution to generate the topic (z) for a word (ID=w) in a document, a numpy float vector of length c. phi_w[i] represents the probability of generating the i-th topic for word (w) in the document. 
* phi_d:  the variational parameters (phi) of a list of categorical distributions to generate the topic (z) in all words of a document, a numpy float matrix of shape v by c. Each row represents the parameters of a categorical distribution to generate different topics in one word in the document; phi_d[i,j] represents the probability of generating the j-th topic for the word ID=i in the document. 
* phi_t:  the variational parameters (phi) of a list of categorical distributions to generate the t-th topic in all words of all document, a numpy float matrix of shape n by v. phi_d[i,j] represents the probability of generating the t-th topic for the word ID=j in the i-th document. 
* Phi:  the topic probability matrix P(T|W,D) for all documents, a float numpy tensor of shape (n, v, c); Phi[i,j,k] represents the probability P(T=k | W = j,D=i), which is the conditional probability of the k-th topic on the j-th word in the vocabulary in the i-th document. 
* theta_d:  the topic mixture in one text document (d), a numpy float vector of length c. Here theta_d[i] is the probability P(T=i | D=d), which is the conditional probability of the i-th topic given the document. 
* Theta:  the topic mixture in each text document, a float numpy matrix of shape (n, c); Theta[i,j] represents the probability P(T=j | D=i), which is the conditional probability of the j-th topic given the i-th document in the dataset. 
* beta_t:  the word probability distribution for one topic (t), a float numpy vector of length v; beta_t[i] represents the probability P(W=i | T =t), which is the conditional probability of generating the i-th word in the vocabulary in the topic (t). 
* Beta:  the word probability distribution for all topics, a float numpy matrix of shape (c, v); Beta[i,j] represents the probability P(W=j | T =i), which is the conditional probability of the j-th word in the vocabulary given the i-th topic. 
* n_iter:  the number of iterations in the EM algorithm, an integer scalar. 

'''
#--------------------------------------------