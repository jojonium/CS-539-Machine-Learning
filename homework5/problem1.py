import numpy as np
from collections import Counter
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Categorical Naive Bayes (20 points)
    In this problem, we will implement the categorical naive Bayes method for spam email detection
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
    Read_and_Agree = False
    #*******************************************
    return Read_and_Agree

#----------------------------------------------------
'''
    (Training: Computing Probability of T) Given a collection of email types (spam: 1, normal: 0), please compute maximum likelihood solution for the P(T=spam), here T is the random variable for the type of an email. 
    ---- Inputs: --------
        * S: a collection of training samples for email types, an integer numpy vector of length n; S[i]=1, if the i-th email in the training set is a spam email, otherwise S[i]=0 (normal email).
    ---- Outputs: --------
        * s: the probability of email type to be spam P(T=spam)=s, a float scalar. It means that P(T=normal) = 1-s.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_PT(S):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return s
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_compute_PT
        --- OR ---- 
        python3 -m nose -v test1.py:test_compute_PT
        --- OR ---- 
        python -m nose -v test1.py:test_compute_PT
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: count the frequency of each word in the vocabulary) Given a collection of word samples (W), please compute the number of times each word of the vocabulary being used in the collection. 
    ---- Inputs: --------
        * W: a collection of words, represented as an array of word IDs, a numpy integer array, W[i] is the ID of the i-th word in the collection; each word ID W[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary.
        * c: the number of possible words in the vocabulary, an integer scalar.
    ---- Outputs: --------
        * C: the frequency counts of each word in the vocabulary, a numpy integer array of length c. C[i] represents the frequency (count of occurrence) of the i-th word in the vocabulary.
    ---- Hints: --------
        * You could use the Counter class in python to to count the frequency of different items in an array. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def count_frequency(W, c):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return C
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_count_frequency
        --- OR ---- 
        python3 -m nose -v test1.py:test_count_frequency
        --- OR ---- 
        python -m nose -v test1.py:test_count_frequency
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: Laplace smoothing) Given the frequency counts of each word in the vocabulary, compute the probability of each word with Laplace smoothing. 
    ---- Inputs: --------
        * C: the frequency counts of each word in the vocabulary, a numpy integer array of length c. C[i] represents the frequency (count of occurrence) of the i-th word in the vocabulary.
        * k: the number of fake samples to add into each word of the vocabulary for Laplace smoothing method, a float scalar.
    ---- Outputs: --------
        * PW: the probabilities of all the words in the vocabulary, a numpy float vector of length c, PW[i] is the probability of the i-th word in the vocabulary.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def laplace_smoothing(C, k):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return PW
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_laplace_smoothing
        --- OR ---- 
        python3 -m nose -v test1.py:test_laplace_smoothing
        --- OR ---- 
        python -m nose -v test1.py:test_laplace_smoothing
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Training: Computing Conditional Probability W given T) Given a collection of word samples being used in Spam emails and a collection of word samples being used in normal emails, please compute the conditional probabilities of P(W|T) with Laplace smoothing. Here T is the random variable for email type and W is the random variable for each Word in email. 
    ---- Inputs: --------
        * Ws: the collection of all word samples from all the spam emails in the training dataset, a numpy integer array, Ws[i] is the ID of the i-th word sample in the collection; each word ID Ws[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary.
        * Wn: the collection of all word samples from all the normal emails in the training dataset, a numpy integer array, Wn[i] is the ID of the i-th word sample in the collection; each word ID Wn[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary.
        * k: the number of fake samples to add into each word of the vocabulary for Laplace smoothing method, a float scalar.
        * c: the number of possible words in the vocabulary, an integer scalar.
    ---- Outputs: --------
        * Ps: P(W|T=spam) the conditional probability of each word value given that the email type is spam, a numpy float vector of length c. Ps[i] represents the probability of using the i-th word in the vocabulary if we know the email type is spam.
        * Pn: P(W|T=normal) the conditional probability of each word value given that the email type is normal, a numpy float vector of length c. Pn[i] represents the probability of using the i-th word in the vocabulary if we know the email type is normal.
    ---- Hints: --------
        * This problem can be solved using 4 line(s) of code.
'''
#---------------------
def compute_PW_T(Ws, Wn, k, c):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return Ps, Pn
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_compute_PW_T
        --- OR ---- 
        python3 -m nose -v test1.py:test_compute_PW_T
        --- OR ---- 
        python -m nose -v test1.py:test_compute_PW_T
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Prediction: compute likelihood ratio) Given the conditional and marginal probabilities from a training dataset and a test sample email W (a sequence of word IDs), please compute the likelihood ratio of the test sample email to be spam vs normal. 
    ---- Inputs: --------
        * W: a collection of words, represented as an array of word IDs, a numpy integer array, W[i] is the ID of the i-th word in the collection; each word ID W[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary.
        * Ps: P(W|T=spam) the conditional probability of each word value given that the email type is spam, a numpy float vector of length c. Ps[i] represents the probability of using the i-th word in the vocabulary if we know the email type is spam.
        * Pn: P(W|T=normal) the conditional probability of each word value given that the email type is normal, a numpy float vector of length c. Pn[i] represents the probability of using the i-th word in the vocabulary if we know the email type is normal.
        * s: the probability of email type to be spam P(T=spam)=s, a float scalar. It means that P(T=normal) = 1-s.
    ---- Outputs: --------
        * r: the likelihood ratio of the email being spam vs normal, a float scalar.
    ---- Hints: --------
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def likelihood_ratio(W, Ps, Pn, s):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    
    #########################################
    return r
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_likelihood_ratio
        --- OR ---- 
        python3 -m nose -v test1.py:test_likelihood_ratio
        --- OR ---- 
        python -m nose -v test1.py:test_likelihood_ratio
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
        ----------- Problem 1 (20 points in total)--------------------- ... ok
        * (4 points) compute_PT ... ok
        * (4 points) count_frequency ... ok
        * (4 points) laplace_smoothing ... ok
        * (4 points) compute_PW_T ... ok
        * (4 points) likelihood_ratio ... ok
        ----------------------------------------------------------------------
        Ran 5 tests in 0.586s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* n:  the number of sample emails in the training set, an integer scalar. 
* c:  the number of possible words in the vocabulary, an integer scalar. 
* S:  a collection of training samples for email types, an integer numpy vector of length n; S[i]=1, if the i-th email in the training set is a spam email, otherwise S[i]=0 (normal email). 
* s:  the probability of email type to be spam P(T=spam)=s, a float scalar. It means that P(T=normal) = 1-s. 
* W:  a collection of words, represented as an array of word IDs, a numpy integer array, W[i] is the ID of the i-th word in the collection; each word ID W[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary. 
* C:  the frequency counts of each word in the vocabulary, a numpy integer array of length c. C[i] represents the frequency (count of occurrence) of the i-th word in the vocabulary. 
* k:  the number of fake samples to add into each word of the vocabulary for Laplace smoothing method, a float scalar. 
* PW:  the probabilities of all the words in the vocabulary, a numpy float vector of length c, PW[i] is the probability of the i-th word in the vocabulary. 
* Ws:  the collection of all word samples from all the spam emails in the training dataset, a numpy integer array, Ws[i] is the ID of the i-th word sample in the collection; each word ID Ws[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary. 
* Wn:  the collection of all word samples from all the normal emails in the training dataset, a numpy integer array, Wn[i] is the ID of the i-th word sample in the collection; each word ID Wn[i] can be an integer of 0, 1,..., or c-1, representing the ID of the word in the vocabulary. 
* Ps:  P(W|T=spam) the conditional probability of each word value given that the email type is spam, a numpy float vector of length c. Ps[i] represents the probability of using the i-th word in the vocabulary if we know the email type is spam. 
* Pn:  P(W|T=normal) the conditional probability of each word value given that the email type is normal, a numpy float vector of length c. Pn[i] represents the probability of using the i-th word in the vocabulary if we know the email type is normal. 
* r:  the likelihood ratio of the email being spam vs normal, a float scalar. 

'''
#--------------------------------------------