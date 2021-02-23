import numpy as np

# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Linear Regression (Maximum Likelihood) 
    In this problem, you will implement the linear regression method based upon maximum likelihood (least square).
    -------------------------
    Package(s) to Install:
        Please install python version 3.7 or above and the following package(s):
        * nose (for unit tests)
        * numpy (for n-dimensional arrays)
    How to Install:
        * nose: To install 'nose' using pip, you could type in the terminal: 
            pip3 install nose
        * numpy: To install 'numpy' using pip, you could type in the terminal: 
            pip3 install numpy
    -------------------------
    A list of all variables being used in this problem is provided at the end of this file.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your Dropbox automatically copied your solution from your desktop computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or use pseudo-code. Don't discuss about the solution at the code level. For example, two students discuss about the solution of a function (which needs 5 lines of code to solve) and they then work on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences (variable names are different). In this case, the two students violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Note: we may use the Stanford Moss system to check your code for code similarity. https://theory.stanford.edu/~aiken/moss/
      Historical Data: in one year, we ended up finding 25% of the students in that class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #*******************************************
    # CHANGE HERE: if you have read and agree with the term above, change "False" to "True".
    Read_and_Agree = False
    #*******************************************
    return Read_and_Agree

#---------------------------------------------------
'''
    Fit a linear model on training samples. Compute the parameter w using Maximum likelihood (equal to least square). 
    Inputs: 
        * X: the feature matrix of the training samples, a numpy matrix of shape n by p.
        * y: the sample labels, a numpy vector of length n.
    Outputs: 
        * w: the weights of the linear regression model, a numpy float vector of length p.
    Hints: 
        * You could use np.linalg.inv() to compute the inverse of a matrix. 
        * You could use @ operator in numpy for matrix multiplication: A@B represents the matrix multiplication between matrices A and B. 
        * You could use A.T to compute the transpose of matrix A. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def least_square(X, y):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return w
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_least_square
        ---------------------------------------------------
    '''
    
    
#---------------------------------------------------
'''
    Fit a linear model on training samples. Compute the parameter w using Maximum posterior (least square regression with L2 regularization). 
    Inputs: 
        * X: the feature matrix of the training samples, a numpy matrix of shape n by p.
        * y: the sample labels, a numpy vector of length n.
        * alpha: the weight of the L2 regularization term in ridge regression, a float scalar.
    Outputs: 
        * w: the weights of the linear regression model, a numpy float vector of length p.
    Hints: 
        * You could use np.eye() to generate an identity matrix. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def ridge_regression(X, y, alpha=0.001):
    #########################################
    ## INSERT YOUR CODE HERE
    
    #########################################
    return w
#---------------------
    ''' 
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_ridge_regression
        ---------------------------------------------------
    '''
    
    

#--------------------------------------------

''' 
    TEST problem 1: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 1 (10 points in total)--------------------- ... ok
        * (5 points) least square ... ok 
        * (5 points) ridge regression ... ok 
        ----------------------------------------------------------------------
        Ran 2 tests in 0.004s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* p:  the number of dimensions in the feature space. 
* X:  the feature matrix of the training samples, a numpy matrix of shape n by p. 
* y:  the sample labels, a numpy vector of length n. 
* w:  the weights of the linear regression model, a numpy float vector of length p. 
* alpha:  the weight of the L2 regularization term in ridge regression, a float scalar. 

'''
#--------------------------------------------