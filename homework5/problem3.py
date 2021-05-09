import numpy as np
import problem2 as p2
from scipy.special import psi,polygamma
from scipy.linalg import inv
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: LDA (Latent Dirichlet Allocation) using Variational EM method (40 points)
    In this problem, we will implement the Latent Dirichlet Allocation (variational EM solution) to model text documents
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    Let's start by building some utility functions. Compute the digamma function, which is the gradient of the log Gamma function. If the input (x) is scalar value, the output (dx) is the digamma value on x; If the input (x) is a vector, the output dx is a vector, where each element is the digamma value of the corresponding element in vector x. 
    ---- Inputs: --------
        * x_g: the input to the digamma function, a float scalar or a numpy vector.
    ---- Outputs: --------
        * dx_g: the output of the digamma function, a float scalar or a numpy vector.
    ---- Hints: --------
        * You could use a function in scipy package to compute digamma function. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_digamma(x_g):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    dx_g = psi(x_g)
    #########################################
    return dx_g
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_digamma
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_digamma
        --- OR ---- 
        python -m nose -v test3.py:test_compute_digamma
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Update Phi with 1 word in 1 document) Given a word ID (w) in a document (d), the current model parameters (Beta) and the variational parameters (gamma_d) in the document (d), update the variational parameter phi_w for the word (w) in the text document (d). 
    ---- Inputs: --------
        * w: the ID of a word in the vocabulary, an integer scalar, which can be 0,1, ..., or v-1.
        * Beta: the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic.
        * gamma_d: the variational parameter (Gamma) for a Dirichlet distribution to generate the topic-mixtures (Theta) in one document (d), a numpy float vector of length c. Gamma[i] represent the parameter of the Dirichlet distribution on the i-th topic when generating the topic mixture for the document.
    ---- Outputs: --------
        * phi_w: the variational parameter (phi) of a categorical distribution to generate the topic (z) of a word (w) in a document, a numpy float vector of length c. phi_w[i] represents the probability of generating the i-th topic for word (w) in the document.
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_phi_w(w, Beta, gamma_d):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)

    #########################################
    return phi_w
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_phi_w
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_phi_w
        --- OR ---- 
        python -m nose -v test3.py:test_compute_phi_w
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Update Phi with all the words in 1 document) Given the current model parameters (Beta) and the variational parameters (gamma_d) on a document (d), update the variational parameter Phi in the document (d). 
    ---- Inputs: --------
        * Beta: the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic.
        * gamma_d: the variational parameter (Gamma) for a Dirichlet distribution to generate the topic-mixtures (Theta) in one document (d), a numpy float vector of length c. Gamma[i] represent the parameter of the Dirichlet distribution on the i-th topic when generating the topic mixture for the document.
    ---- Outputs: --------
        * phi_d: the variational parameters (phi) of a list of categorical distributions to generate the topic (z) in one document, a numpy float matrix of shape m by c. Each row represents the parameters of a categorical distribution to generate different topics in one word in the document; phi_d[i,j] represents the probability of generating the j-th topic for the i-th word.
    ---- Hints: --------
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_phi_d(Beta, gamma_d):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)

    #########################################
    return phi_d
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_phi_d
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_phi_d
        --- OR ---- 
        python -m nose -v test3.py:test_compute_phi_d
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (E-Step: Update Phi on all words in all documents) Given the current model parameters (Beta) and the variational parameters (Gamma) in all the documents, update the variational parameters Phi in all documents. 
    ---- Inputs: --------
        * Beta: the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic.
        * Gamma: the variational parameters (Gamma) for multiple Dirichlet distributions to generate the topic-mixtures (Theta) in all documents, a numpy float matrix of shape n by c. Gamma[i] represent the parameter of a Dirichlet distribution to generate the topic-mixture in the i-th document.
    ---- Outputs: --------
        * Phi: the variational parameters (Phi) of categorical distributions (one distribution on each word of each document) to generate the topics (z) in all words of all document, a numpy float tensor of shape n by m by c. Phi[i] represents the phi values on the i-th text document; Phi[i,j,k] = P(T=k | W_j,D=i) represents the probability of generating the k-th topic for the j-th word in the i-th text document.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_Phi(Beta, Gamma):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)

    #########################################
    return Phi
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_Phi
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_Phi
        --- OR ---- 
        python -m nose -v test3.py:test_compute_Phi
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Update Gamma in 1 document) Given the variational parameters (phi_d) on a text document, update the variational parameter gamma_d on the document. 
    ---- Inputs: --------
        * phi_d: the variational parameters (phi) of a list of categorical distributions to generate the topic (z) in one document, a numpy float matrix of shape m by c. Each row represents the parameters of a categorical distribution to generate different topics in one word in the document; phi_d[i,j] represents the probability of generating the j-th topic for the i-th word.
        * C_d: word frequency counts in a text document, an integer numpy vector of length v; C[i] represents how many times the i-th word in the vocabulary has been used in the document.
        * Alpha: the parameters of the prior probability distribution (a Dirichlet distribution) for generating topic-mixture for each document, a float vector of length c.
    ---- Outputs: --------
        * gamma_d: the variational parameter (Gamma) for a Dirichlet distribution to generate the topic-mixtures (Theta) in one document (d), a numpy float vector of length c. Gamma[i] represent the parameter of the Dirichlet distribution on the i-th topic when generating the topic mixture for the document.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_gamma_d(phi_d, C_d, Alpha):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return gamma_d
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_gamma_d
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_gamma_d
        --- OR ---- 
        python -m nose -v test3.py:test_compute_gamma_d
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (E-step: Update Gamma in all documents) Given the variational parameters (Phi) on all text documents, update the variational parameters Gamma on all documents. 
    ---- Inputs: --------
        * Phi: the variational parameters (Phi) of categorical distributions (one distribution on each word of each document) to generate the topics (z) in all words of all document, a numpy float tensor of shape n by m by c. Phi[i] represents the phi values on the i-th text document; Phi[i,j,k] = P(T=k | W_j,D=i) represents the probability of generating the k-th topic for the j-th word in the i-th text document.
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * Alpha: the parameters of the prior probability distribution (a Dirichlet distribution) for generating topic-mixture for each document, a float vector of length c.
    ---- Outputs: --------
        * Gamma: the variational parameters (Gamma) for multiple Dirichlet distributions to generate the topic-mixtures (Theta) in all documents, a numpy float matrix of shape n by c. Gamma[i] represent the parameter of a Dirichlet distribution to generate the topic-mixture in the i-th document.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_Gamma(Phi, C, Alpha):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return Gamma
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_Gamma
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_Gamma
        --- OR ---- 
        python -m nose -v test3.py:test_compute_Gamma
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Expectation Step of EM algorithm) Given the current model parameters (Alpha and Beta), compute the optimal values for variational parameters (Phi and Gamma). 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * Alpha: the parameters of the prior probability distribution (a Dirichlet distribution) for generating topic-mixture for each document, a float vector of length c.
        * Beta: the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic.
        * n_iter_var: the number of iterations for iteratively updating Phi and Gamma during variational inference.
    ---- Outputs: --------
        * Phi: the variational parameters (Phi) of categorical distributions (one distribution on each word of each document) to generate the topics (z) in all words of all document, a numpy float tensor of shape n by m by c. Phi[i] represents the phi values on the i-th text document; Phi[i,j,k] = P(T=k | W_j,D=i) represents the probability of generating the k-th topic for the j-th word in the i-th text document.
        * Gamma: the variational parameters (Gamma) for multiple Dirichlet distributions to generate the topic-mixtures (Theta) in all documents, a numpy float matrix of shape n by c. Gamma[i] represent the parameter of a Dirichlet distribution to generate the topic-mixture in the i-th document.
    ---- Hints: --------
        * (Step 1) update Phi with Gamma. 
        * (Step 2) update Gamma with Phi. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def E_step(C, Alpha, Beta, n_iter_var=20):
    n = C.shape[0] # n documents
    c, v = Beta.shape # c topics, v words in the vocabulary
    #initialize variational parameters
    Gamma = np.ones((n,c))*Alpha
    for _ in range(n_iter_var): #repeat multiple passes
        pass #no operation (you can ignore this line)
        #########################################
        ## INSERT YOUR CODE HERE (4 points)
    
        #########################################
    return Phi, Gamma
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_E_step
        --- OR ---- 
        python3 -m nose -v test3.py:test_E_step
        --- OR ---- 
        python -m nose -v test3.py:test_E_step
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
        * You could use some function in the previous problem to solve this question. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_beta_t(C, phi_t):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return beta_t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_beta_t
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_beta_t
        --- OR ---- 
        python -m nose -v test3.py:test_compute_beta_t
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (M-step: Computing word distribution of each topic) Given a collection of text documents, represented as word-frequency format (C), and inferred topic distributions (Phi), please compute the maximum likelihood solution of the parameter Beta: the word distribution of each topic, i.e., the conditional probabilities of P(W|T). 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * Phi: the variational parameters (Phi) of categorical distributions (one distribution on each word of each document) to generate the topics (z) in all words of all document, a numpy float tensor of shape n by m by c. Phi[i] represents the phi values on the i-th text document; Phi[i,j,k] = P(T=k | W_j,D=i) represents the probability of generating the k-th topic for the j-th word in the i-th text document.
    ---- Outputs: --------
        * Beta: the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic.
    ---- Hints: --------
        * You could use some function in the previous problem to solve this question. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_Beta(C, Phi):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return Beta
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_compute_Beta
        --- OR ---- 
        python3 -m nose -v test3.py:test_compute_Beta
        --- OR ---- 
        python -m nose -v test3.py:test_compute_Beta
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Variational EM method for LDA Model) Given the word counts of a set of documents, optimize the model parameters (Beta) using Variational EM. 
    ---- Inputs: --------
        * C: word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document.
        * Alpha: the parameters of the prior probability distribution (a Dirichlet distribution) for generating topic-mixture for each document, a float vector of length c.
        * n_iter_var: the number of iterations for iteratively updating Phi and Gamma during variational inference.
        * n_iter_EM: the number of iterations for EM algorithm.
    ---- Outputs: --------
        * Beta: the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic.
    ---- Hints: --------
        * Step 1 (E step): Compute Phi and Gamma based upon the current values of Alpha and Beta. 
        * Step 2 (M step): update the parameter Beta based upon the new values of Phi and Gamma. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def LDA(C, Alpha, n_iter_var=20, n_iter_EM=10):
    c = len(Alpha)
    v = C.shape[1] 
    Beta = np.random.rand(c,v) # initialize Beta
    Beta = Beta/Beta.sum(1,keepdims=True)
    for _ in range(n_iter_EM): # repeat multiple iterations of E and M steps
        pass # you could ignore this line
        #########################################
        ## INSERT YOUR CODE HERE (4 points)
    
        #########################################
    return Beta
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test3.py:test_LDA
        --- OR ---- 
        python3 -m nose -v test3.py:test_LDA
        --- OR ---- 
        python -m nose -v test3.py:test_LDA
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
        ----------- Problem 3 (40 points in total)--------------------- ... ok
        * (4 points) compute_digamma ... ok
        * (4 points) compute_phi_w ... ok
        * (4 points) compute_phi_d ... ok
        * (4 points) compute_Phi ... ok
        * (4 points) compute_gamma_d ... ok
        * (4 points) compute_Gamma ... ok
        * (4 points) E_step ... ok
        * (4 points) compute_beta_t ... ok
        * (4 points) compute_Beta ... ok
        * (4 points) LDA ... ok
        ----------------------------------------------------------------------
        Ran 10 tests in 0.586s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of text documents in the dataset, an integer scalar. 
* v:  the number of possible words in the vocabulary, an integer scalar. 
* c:  the number of possible topics (categories) in the model, an integer scalar. 
* x_g:  the input to the digamma function, a float scalar or a numpy vector. 
* dx_g:  the output of the digamma function, a float scalar or a numpy vector. 
* w:  the ID of a word in the vocabulary, an integer scalar, which can be 0,1, ..., or v-1. 
* C_d:  word frequency counts in a text document, an integer numpy vector of length v; C[i] represents how many times the i-th word in the vocabulary has been used in the document. 
* C:  word frequency counts in the text documents, an integer numpy matrix of shape (n, v); C[i,j] represents how many times the j-th word in the vocabulary has been used in the i-th document. 
* Alpha:  the parameters of the prior probability distribution (a Dirichlet distribution) for generating topic-mixture for each document, a float vector of length c. 
* beta_t:  the word probability distribution for one topic (t), a float numpy vector of length v; beta_t[i] represents the probability P(W=i | T =t), which is the conditional probability of generating the i-th word in the vocabulary in the topic (t). 
* Beta:  the parameters for word distribution on c topics, a numpy float matrix of shape c by v. Beta[i,j] represents the probability of generating the j-th word (ID=j) in the i-th topic. 
* phi_w:  the variational parameter (phi) of a categorical distribution to generate the topic (z) of a word (w) in a document, a numpy float vector of length c. phi_w[i] represents the probability of generating the i-th topic for word (w) in the document. 
* phi_d:  the variational parameters (phi) of a list of categorical distributions to generate the topic (z) in one document, a numpy float matrix of shape m by c. Each row represents the parameters of a categorical distribution to generate different topics in one word in the document; phi_d[i,j] represents the probability of generating the j-th topic for the i-th word. 
* phi_t:  the variational parameters (phi) of a list of categorical distributions to generate the t-th topic in all words of all document, a numpy float matrix of shape n by v. phi_d[i,j] represents the probability of generating the t-th topic for the word ID=j in the i-th document. 
* Phi:  the variational parameters (Phi) of categorical distributions (one distribution on each word of each document) to generate the topics (z) in all words of all document, a numpy float tensor of shape n by m by c. Phi[i] represents the phi values on the i-th text document; Phi[i,j,k] = P(T=k | W_j,D=i) represents the probability of generating the k-th topic for the j-th word in the i-th text document. 
* gamma_d:  the variational parameter (Gamma) for a Dirichlet distribution to generate the topic-mixtures (Theta) in one document (d), a numpy float vector of length c. Gamma[i] represent the parameter of the Dirichlet distribution on the i-th topic when generating the topic mixture for the document. 
* Gamma:  the variational parameters (Gamma) for multiple Dirichlet distributions to generate the topic-mixtures (Theta) in all documents, a numpy float matrix of shape n by c. Gamma[i] represent the parameter of a Dirichlet distribution to generate the topic-mixture in the i-th document. 
* n_iter_var:  the number of iterations for iteratively updating Phi and Gamma during variational inference. 
* n_iter_EM:  the number of iterations for EM algorithm. 

'''
#--------------------------------------------