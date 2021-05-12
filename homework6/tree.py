import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------

#-----------------------------------------------
#    Utility functions 
#-----------------------------------------------

#--------------------------
def entropy(Y):
    '''
        Compute the entropy of a list of values.
        Input:
            Y: a list of values, a numpy array of int/float/string values.
        Output:
            e: the entropy of the list of values, a float scalar
    '''
    
    n = len(Y) # total number of values 
    c = Counter(Y) # create a counter on the list
    e = 0.
    for k,v in c.items():
        p = v/n
        e -=  p*math.log(p,2)
    return e  
#--------------------------
def conditional_entropy(Y,X):
    '''
        Compute the conditional entropy of y given x.
        Input:
            Y: a list of values, a numpy array of int/float/string values.
            X: a list of values, a numpy array of int/float/string values.
        Output:
            ce: the conditional entropy of y given x, a float scalar
    '''
    n = len(Y) # total number of values 
    c = Counter(X) # create a counter on the list
    ce = 0.
    for k,v in c.items():
        ce += entropy(Y[X==k])*v/n
    return ce 
#--------------------------
def information_gain(Y,X):
    '''
        Compute the information gain of y after splitting over attribute x
        Input:
            X: a list of values, a numpy array of int/float/string values.
            Y: a list of values, a numpy array of int/float/string values.
        Output:
            g: the information gain of y after splitting over x, a float scalar
    '''
    g = entropy(Y) - conditional_entropy(Y,X) 
    return g

#-----------------------------------------------
#     A Decision Tree Node
#-----------------------------------------------
class Node:
    '''
        The class of a Decision Tree Node 

        Properties
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the first child node for values smaller than threshold
            C2: the second child node for values larger than threshold
    '''
    #--------------------------
    def __init__(self,X,Y):
        '''
        Create a decision tree node
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
               Each row of X represents one attribute, each column of X represents a data instance.
            Y: the class labels in the node, a numpy array of length n.
               Each element can be int/float/string.
        '''
        assert len(X)>0
        assert len(Y)>0
        self.X = X
        self.Y = Y
        # compute the most common label in the node for prediction
        self.p = Counter(Y).most_common(1)[0][0]
        # test whether or not if the node is a leaf node
        self.isleaf = True
        for f in X:
            if not len(np.unique(f))==1:
                self.isleaf = False
                break
        if (len(np.unique(Y))==1):
            self.isleaf = True

    #--------------------------
    def cutting_points(self,i):
        '''
            Find all possible cutting points in the i-th attribute. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            Input:
                i: the index of the attribute to be used, an integer scalar 
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        x = self.X[i] # the i-th attribute
        cp = [] 
        z = sorted(np.unique(x)) # unique values in the attribute
        for i in range(len(z)-1):
            idx = np.logical_or(x==z[i], x==z[i+1])
            ys = self.Y[idx]
            if len(np.unique(ys))>1: 
                cp.append((z[i] + z[i+1])/2.) # add a candidate cutting point
        return cp
   
    #--------------------------
    def best_threshold(self,i):
        '''
            Find the best threshold among all possible cutting points in the i-th attribute. 
            Input:
                i: the index of the attribute to be used, an integer scalar 
            Output:
                th: the best threshold, a float scalar. 
                g: the information gain by using the best threshold, a float scalar. 
        '''
        X = self.X[i] # the i-th attribute
        cp = self.cutting_points(i)
        assert len(cp)>0
        g= -math.inf
        for c in cp:
            x = X>=c
            g_new = information_gain(self.Y,x)
            if g_new>g:
                g = g_new
                th = c
        return th,g 
    #--------------------------
    def best_attribute(self):
        '''
            Find the best attribute to split the node. The attributes have continuous values (int/float).
            Here we use information gain to evaluate the attributes. 
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        g= -math.inf
        for k in range(self.X.shape[0]):
            if len(np.unique(self.X[k]))>1:
                tk,gk = self.best_threshold(k)
                if gk>g:
                    g = gk
                    self.i = k
                    self.th = tk

    #--------------------------
    def add_children_nodes(self):
        '''
            build the children nodes C1 and C2 with the best attribute and threshold in the node 
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        self.best_attribute()
        i,th = self.i, self.th 
        x = self.X[i] 
        self.C1 = Node(self.X[:,x<th], self.Y[x<th])
        self.C2 = Node(self.X[:,x>=th], self.Y[x>=th])

    #--------------------------
    def build_tree(self):
        '''
            Recursively build a subtree from the current tree node.
        '''
        if self.isleaf==False: 
            self.add_children_nodes()
            self.C1.build_tree()
            self.C2.build_tree()
    #--------------------------
    def predict_1(self,x):
        '''
            Using the decision tree starting from the current node to predict the label on one test instance
            Input:
                x: the attribute vector, a numpy vector of shape p.
                   Each attribute value can be int/float
            Output:
                p: the label prediction on the test instance, a scalar, can be int/float/string.
        '''
        if self.isleaf:
            return self.p
        if x[self.i] <= self.th:
            return self.C1.predict_1(x)
        else:
            return self.C2.predict_1(x)


#-----------------------------------------------
#     A Decision Tree 
#-----------------------------------------------
class DecisionTree:
    '''
        The Class of Decision Tree
        Properties
            root: the root node of the decision tree
    '''
    #--------------------------
    def __init__(self,X,Y):
        '''
        Create a decision tree
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
               Each row of X represents one attribute, each column of X represents a data instance.
            Y: the class labels in the node, a numpy array of length n.
               Each element can be int/float/string.
        '''
        self.root = Node(X,Y)  # create root node
        self.root.build_tree() # build the tree

    #--------------------------
    def predict_1(self,x):
        '''
            Using the decision tree to predict the label on one test instance
            Input:
                x: the attribute vector, a numpy vector of shape p.
                   Each attribute value can be int/float
            Output:
                p: the label prediction on the test instance, a scalar, can be int/float/string.
        '''
        return self.root.predict_1(x)


    #--------------------------
    def predict(self,X):
        '''
            Using the decision tree to predict the label on all test instances
            Input:
                X: the feature matrix of all test instances, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                P: the predicted class labels on all test instances, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        return np.array([self.predict_1(x) for x in X.T])

