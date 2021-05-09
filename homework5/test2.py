from problem2 import *
import sys
import math

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (40 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_phi_w():
    ''' (5 points) compute_phi_w'''
    # 3 topics, 4 words in the vocabulary
    w=0 # the word id =0, the first word in the vocabulary 
    Beta = np.array([[ 0.1, 0.4, 0.3, 0.2], # word probability distribution in the 1st topic
                     [ 0.2, 0.5, 0.1, 0.2], # word probability distribution in the 2nd topic
                     [ 0.3, 0.6, 0.1, 0.0]]) # word probability distribution in the 3rd topic
    theta_d = np.array([0.5,0.3,0.2]) # the topic distribution on the 3 topics in the document
    phi_w = compute_phi_w(w,Beta, theta_d)
    phi_w_true = [0.29411765,0.35294118,0.35294118]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
    w=1 # the word id =1, the 2nd word in the vocabulary 
    phi_w = compute_phi_w(w,Beta, theta_d)
    phi_w_true = [0.42553191,0.31914894,0.25531915]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
    w=2 # the word id =2, the 3rd word in the vocabulary 
    phi_w = compute_phi_w(w,Beta, theta_d)
    phi_w_true = [0.75,0.15,0.1 ]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
    w=3 # the word id =3, the 4th word in the vocabulary 
    phi_w = compute_phi_w(w,Beta, theta_d)
    phi_w_true = [0.625,0.375,0.   ]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
#---------------------------------------------------
def test_compute_phi_d():
    ''' (5 points) compute_phi_d'''
    # 3 topics, 4 words in the vocabulary
    Beta = np.array([[ 0.1, 0.4, 0.3, 0.2], # word probability distribution in the 1st topic
                     [ 0.2, 0.5, 0.1, 0.2], # word probability distribution in the 2nd topic
                     [ 0.3, 0.6, 0.1, 0.0]]) # word probability distribution in the 3rd topic
    theta_d = np.array([0.5,0.3,0.2]) # the topic distribution on the 3 topics in the document
    phi_d = compute_phi_d(Beta, theta_d)
    phi_true =[[0.29411765,0.35294118,0.35294118],
               [0.42553191,0.31914894,0.25531915],
               [0.75      ,0.15      ,0.1       ],
               [0.625     ,0.375     ,0.        ]]  
    assert np.allclose(phi_d,phi_true,atol=0.01)
#---------------------------------------------------
def test_compute_Phi():
    ''' (5 points) compute_Phi'''
    # example of 3 topics, 4 words, 5 documents
    # topic mixture of each document, matrix of (5 documents X 3 topic)
    Theta = np.array([[0.5,0.5,0.0],
                      [0.0,0.2,0.8],
                      [0.0,0.5,0.5],
                      [0.4,0.6,0.0],
                      [0.0,0.0,1.0]])
    # word distribution in each topic, matrix of (3 documents X 4 topic)
    Beta= np.array([[0.4,0.4,0.1,0.1],
                    [0.4,0.3,0.2,0.1],
                    [0.1,0.1,0.4,0.4]])
    Phi = compute_Phi(Theta,Beta) 
    Phi_true = np.array([[[0.5       ,0.5       ,0.        ],
                          [0.57142857,0.42857143,0.        ],
                          [0.33333333,0.66666667,0.        ],
                          [0.5       ,0.5       ,0.        ]],
                         [[0.        ,0.5       ,0.5       ],
                          [0.        ,0.42857143,0.57142857],
                          [0.        ,0.11111111,0.88888889],
                          [0.        ,0.05882353,0.94117647]],
                        
                         [[0.        ,0.8       ,0.2       ],
                          [0.        ,0.75      ,0.25      ],
                          [0.        ,0.33333333,0.66666667],
                          [0.        ,0.2       ,0.8       ]],
                        
                         [[0.4       ,0.6       ,0.        ],
                          [0.47058824,0.52941176,0.        ],
                          [0.25      ,0.75      ,0.        ],
                          [0.4       ,0.6       ,0.        ]],
                        
                         [[0.        ,0.        ,1.        ],
                          [0.        ,0.        ,1.        ],
                          [0.        ,0.        ,1.        ],
                          [0.        ,0.        ,1.        ]]])
    assert type(Phi) == np.ndarray
    assert Phi.shape == (5,4,3)
    assert np.allclose(Phi, Phi_true, atol=0.01)
    # test with random sizes
    for _ in range(10):
        c = np.random.randint(2,8)
        v = np.random.randint(2,8)
        n = np.random.randint(2,8)
        Theta = np.random.rand(n,c) + 0.01
        Theta = Theta/Theta.sum(axis = 1,keepdims=True)
        Beta = np.random.rand(c,v) +0.01
        Beta = Beta/Beta.sum(axis=1,keepdims=True)
        Phi = compute_Phi(Theta,Beta) 
        assert Phi.shape == (n,v,c)
        assert np.allclose(Phi.sum(axis=2), np.ones((n,v)))
#---------------------------------------------------
def test_compute_theta_d():
    ''' (5 points) compute_theta_d'''
    # 3 topics, 4 words in the vocabulary
    phi_d = np.array([[0.1,0.2,0.7], # (4 words, 3 topics)
                      [0.2,0.3,0.5],
                      [0.3,0.4,0.3],
                      [0.4,0.5,0.1]])
    C_d = np.array([3,1,2,4]) # word counts in the document: 1st word in the vocabulary is used 3 times in the document;
                              # the 2nd word in the vocabulary is used once in the document;
                              # the 3rd word in the vocabulary is used twice in the document;
                              # the 4th word in the vocabulary is used 4 times in the document
    theta_d = compute_theta_d(C_d,phi_d)
    assert np.allclose(theta_d,[0.27,0.37,0.36],atol=0.01)
    phi_d = np.array([[0.5,0.5,0.0],
                      [0.2,0.4,0.4],
                      [0.3,0.6,0.1],
                      [0.5,0.5,0.0]])
    C_d = np.array([1,1,1,1]) 
    theta_d = compute_theta_d(C_d,phi_d)
    assert np.allclose(theta_d,[0.375,0.5  ,0.125],atol=0.01)
    phi_d = np.array([[0.0,0.5,0.5],
                     [0.1,0.4,0.5],
                     [0.1,0.1,0.8],
                     [0.1,0.0,0.9]])
    C_d = np.array([2,2,2,2]) 
    theta_d = compute_theta_d(C_d,phi_d)
    assert np.allclose(theta_d,[0.075,0.25 ,0.675],atol=0.01)
    phi_d = np.array( [[0.0,0.8,0.2],
                       [0.1,0.7,0.2],
                       [0.1,0.3,0.6],
                       [0.0,0.2,0.8]])
    C_d = np.array([0,0,1,1]) 
    theta_d = compute_theta_d(C_d,phi_d)
    assert np.allclose(theta_d,[0.05 ,0.25 ,0.7  ],atol=0.01)
    phi_d = np.array( [[0.4,0.6,0.0],
                       [0.4,0.5,0.1],
                       [0.2,0.7,0.1],
                       [0.4,0.6,0.0]])
    C_d = np.array([0,0,2,2]) 
    theta_d = compute_theta_d(C_d,phi_d)
    assert np.allclose(theta_d,[0.3  ,0.65 ,0.05 ],atol=0.01)
#---------------------------------------------------
def test_compute_Theta():
    ''' (5 points) compute_Theta'''
    # example of 3 topics, 4 words, 5 documents
    # tensor of shape (5,4,3)
    Phi = np.array([[[0.1,0.2,0.7], # (4 words, 3 topics)
                     [0.2,0.3,0.5],
                     [0.3,0.4,0.3],
                     [0.4,0.5,0.1]],
                    [[0.5,0.5,0.0],
                     [0.2,0.4,0.4],
                     [0.3,0.6,0.1],
                     [0.5,0.5,0.0]],
                    [[0.0,0.5,0.5],
                     [0.1,0.4,0.5],
                     [0.1,0.1,0.8],
                     [0.1,0.0,0.9]],
                    [[0.0,0.8,0.2],
                     [0.1,0.7,0.2],
                     [0.1,0.3,0.6],
                     [0.0,0.2,0.8]],
                    [[0.4,0.6,0.0],
                     [0.4,0.5,0.1],
                     [0.2,0.7,0.1],
                     [0.4,0.6,0.0]]])
    # matrix of shape (5,4)
    C = np.array([  [3,1,2,4],
                    [1,1,1,1],
                    [2,2,2,2],
                    [0,0,1,1],
                    [0,0,2,2]])
    Theta= compute_Theta(C,Phi)
    assert Theta.shape == (5,3)
    Theta_true= [[0.27 ,0.37 ,0.36],
                 [0.375,0.5  ,0.125],
                 [0.075,0.25 ,0.675],
                 [0.05 ,0.25 ,0.7  ],
                 [0.3  ,0.65 ,0.05 ]]
    assert np.allclose(Theta, Theta_true, atol =0.001)
    # test with random sizes
    for _ in range(10):
        c = np.random.randint(2,8)
        v = np.random.randint(2,8)
        n = np.random.randint(2,8)
        C = np.random.randint(2,5,size=(n,v))
        Phi = np.random.rand(n,v,c) + 0.01
        Phi = Phi/Phi.sum(axis = 2,keepdims=True)
        Theta = compute_Theta(C,Phi) 
        assert Theta.shape == (n,c)
        assert np.allclose(Theta.sum(axis=1), np.ones(n))
#---------------------------------------------------
def test_compute_beta_t():
    ''' (5 points) compute_beta_t'''
    # example of  4 words, 5 documents
    phi_t = np.array([[0.5,0.2,0.3,0.5],
                      [0. ,0.1,0.1,0.1],
                      [0. ,0.1,0.1,0. ],
                      [0.4,0.4,0.2,0.4],
                      [0. ,0. ,0. ,0. ]])
    C = np.array([  [1,1,1,1],
                    [2,2,2,2],
                    [0,0,1,1],
                    [0,0,2,2],
                    [1,2,3,4]])
    beta_t= compute_beta_t(C,phi_t)
    beta_true= [0.14705882,0.11764706,0.29411765,0.44117647]
    assert np.allclose(beta_t, beta_true, atol =0.001)
    phi_t = np.array([[0.5, 0.4, 0.6, 0.5],
                      [0.5, 0.4, 0.1, 0. ],
                      [0.8, 0.7, 0.3, 0.2],
                      [0.6, 0.5, 0.7, 0.6],
                      [0. , 0. , 0. , 0. ]])
    beta_t= compute_beta_t(C,phi_t)
    beta_true= [0.21126761,0.16901408,0.35211268,0.26760563]
    assert np.allclose(beta_t, beta_true, atol =0.001)
    phi_t = np.array([[0. , 0.4, 0.1, 0. ],
                      [0.5, 0.5, 0.8, 0.9],
                      [0.2, 0.2, 0.6, 0.8],
                      [0. , 0.1, 0.1, 0. ],
                      [1. , 1. , 1. , 1. ]])
    beta_t= compute_beta_t(C,phi_t)
    beta_true= [0.11428571,0.19428571,0.31428571,0.37714286]
    assert np.allclose(beta_t, beta_true, atol =0.001)
#---------------------------------------------------
def test_compute_Beta():
    ''' (5 points) compute_Beta'''
    # example of 3 topics, 4 words, 5 documents
    # tensor of shape (5,4,3)
    Phi = np.array([[[0.5,0.5,0.0],
                     [0.2,0.4,0.4],
                     [0.3,0.6,0.1],
                     [0.5,0.5,0.0]],
                    [[0.0,0.5,0.5],
                     [0.1,0.4,0.5],
                     [0.1,0.1,0.8],
                     [0.1,0.0,0.9]],
                    [[0.0,0.8,0.2],
                     [0.1,0.7,0.2],
                     [0.1,0.3,0.6],
                     [0.0,0.2,0.8]],
                    [[0.4,0.6,0.0],
                     [0.4,0.5,0.1],
                     [0.2,0.7,0.1],
                     [0.4,0.6,0.0]],
                    [[0. ,0. ,1. ],
                     [0. ,0. ,1. ],
                     [0. ,0. ,1. ],
                     [0. ,0. ,1. ]]])
    # matrix of shape (5,4)
    C = np.array([
                    [1,1,1,1],
                    [2,2,2,2],
                    [0,0,1,1],
                    [0,0,2,2],
                    [1,2,3,4]])
    Beta= compute_Beta(C,Phi)
    assert Beta.shape == (3,4)
    Beta_true= [[0.14705882,0.11764706,0.29411765,0.44117647],
                [0.21126761,0.16901408,0.35211268,0.26760563],
                [0.11428571,0.19428571,0.31428571,0.37714286]]
    assert np.allclose(Beta, Beta_true, atol =0.001)
    # test with random sizes
    for _ in range(10):
        c = np.random.randint(2,8)
        v = np.random.randint(2,8)
        n = np.random.randint(2,8)
        C = np.random.randint(2,5,size=(n,v))
        Phi = np.random.rand(n,v,c) + 0.01
        Phi = Phi/Phi.sum(axis = 2,keepdims=True)
        Beta = compute_Beta(C,Phi) 
        assert Beta.shape == (c,v)
        assert np.allclose(Beta.sum(axis=1), np.ones(c))
#---------------------------------------------------
def test_PLSA():
    ''' (5 points) PLSA'''
    # example of 2 topics, 4 words, 5 documents
    # matrix of shape (5,4)
    C = np.array([[2,2,1,1],
                  [1,1,2,2],
                  [2,2,4,4],
                  [6,6,3,3],
                  [8,8,8,8]])
    count = 0
    for _ in range(2):
        Theta, Beta, Phi = PLSA(C,c=2)
        assert Theta.shape==(5,2)
        assert Beta.shape==(2,4)
        assert Phi.shape==(5,4,2)
        assert np.allclose(Theta.sum(axis=1),np.ones(5),atol=0.001)
        assert np.allclose(Beta.sum(axis=1),np.ones(2),atol=0.001)
        assert np.allclose(Phi.sum(axis=2),np.ones((5,4)),atol=0.001)
        d = (Theta[:,0]-Theta[:,1]).squeeze()
        # document 0 and 3 should be assigned to the same topic (with high probability)
        if d[0]*d[3]>0:
            count+=1
        # document 1 and 2 should be assigned to the same topic (with high probability)
        if d[1]*d[2]>0:
            count+=1
        # document 4 should be assigned to both topics (with similar probabilities)
        if np.abs(d[4])<0.2:
            count+=1
    assert count>2

