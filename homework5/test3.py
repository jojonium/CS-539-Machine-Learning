from problem3 import *
import sys
import math

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (40 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3.6 or above 
    assert sys.version_info[1]>=6

#---------------------------------------------------
def test_compute_digamma():
    ''' (4 points) compute_digamma'''
    dx = compute_digamma(1)
    assert np.allclose(dx,-0.577215,atol=0.001)
    dx = compute_digamma(2)
    assert np.allclose(dx,0.4227843,atol=0.001)
    dx = compute_digamma(3)
    assert np.allclose(dx,0.9227843,atol=0.001)
    x = np.array([1,2,3])
    dx = compute_digamma(x)
    assert np.allclose(dx,[-0.57721566,  0.42278434,  0.92278434],atol=0.001)
#---------------------------------------------------
def test_compute_phi_w():
    ''' (4 points) compute_phi_w'''
    w=0 # the word id =0, the first word in the vocabulary (suppose we have 4 words in the vocabulary
    # suppose we have 3 topics
    Beta = np.array([[ 0.1, 0.4, 0.3, 0.2], # word probability distribution in the 1st topic
                     [ 0.2, 0.5, 0.1, 0.2], # word probability distribution in the 2nd topic
                     [ 0.3, 0.6, 0.1, 0.0]]) # word probability distribution in the 3rd topic
    Gamma = np.array([1,2,1.5]) # gamma values for the 3 topics
    phi_w = compute_phi_w(w,Beta, Gamma)
    phi_w_true = [0.08348391,0.4538656 ,0.46265048]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
    w=1 # the word id =1, the 2nd word in the vocabulary 
    phi_w = compute_phi_w(w,Beta, Gamma)
    phi_w_true = [0.13949437,0.47398125,0.38652439]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
    w=2 # the word id =2, the 3rd word in the vocabulary 
    phi_w = compute_phi_w(w,Beta, Gamma)
    phi_w_true = [0.39653451,0.35929752,0.24416798]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
    w=3 # the word id =3, the 4th word in the vocabulary 
    phi_w = compute_phi_w(w,Beta, Gamma)
    phi_w_true = [0.26894142,0.73105858,0.        ]
    assert np.allclose(phi_w,phi_w_true,atol=0.01)
#---------------------------------------------------
def test_compute_phi_d():
    ''' (4 points) compute_phi_d'''
    # 4 words in the vocabulary, 3 topics
    Beta = np.array([[ 0.1, 0.4, 0.3, 0.2], # word probability distribution in the 1st topic
                     [ 0.2, 0.5, 0.1, 0.2], # word probability distribution in the 2nd topic
                     [ 0.3, 0.6, 0.1, 0.0]]) # word probability distribution in the 3rd topic
    gamma_d = np.array([1,2,1.5]) # gamma values for the 3 topics
    phi_d = compute_phi_d(Beta, gamma_d)
    phi_true =[[0.08348391,0.4538656 ,0.46265048],
               [0.13949437,0.47398125,0.38652439],
               [0.39653451,0.35929752,0.24416798],
               [0.26894142,0.73105858,0.        ]]
    assert np.allclose(phi_d,phi_true,atol=0.01)
    gamma_d = np.array([4,1,3]) # gamma values for the 3 topics
    phi_d = compute_phi_d(Beta, gamma_d)
    phi_true =[[0.28823815,0.09216688,0.61959497],
               [0.43962872,0.08785966,0.47251162],
               [0.77391173,0.04124427,0.184844  ],
               [0.86215834,0.13784166,0.        ]]
    assert np.allclose(phi_d,phi_true,atol=0.01)
#---------------------------------------------------
def test_compute_Phi():
    ''' (4 points) compute_Phi'''
    # suppose we have 3 topics, 2 documents, 4 words in the vocabulary
    Beta = np.array([[ 0.1, 0.4, 0.3, 0.2], # word probability distribution in the 1st topic
                     [ 0.2, 0.5, 0.1, 0.2], # word probability distribution in the 2nd topic
                     [ 0.3, 0.6, 0.1, 0.0]]) # word probability distribution in the 3rd topic
    Gamma = np.array([[1,2,1.5],  # gamma values for the 3 topics in the first document
                      [4,1,3  ]]) # gamma values for the 3 topics in the second document
    Phi = compute_Phi(Beta, Gamma)
    Phi_true = [[[0.08348391,0.4538656 ,0.46265048],
                 [0.13949437,0.47398125,0.38652439],
                 [0.39653451,0.35929752,0.24416798],
                 [0.26894142,0.73105858,0.        ]],
                [[0.28823815,0.09216688,0.61959497],
                 [0.43962872,0.08785966,0.47251162],
                 [0.77391173,0.04124427,0.184844  ],
                 [0.86215834,0.13784166,0.        ]]]
    assert np.allclose(Phi,Phi_true,atol=0.01)
#---------------------------------------------------
def test_compute_gamma_d():
    ''' (4 points) compute_gamma_d'''
    # 3 topics, 4 words in the vocabulary
    phi_d = np.array([[0.1,0.2,0.7],
                      [0.2,0.3,0.5],
                      [0.3,0.4,0.3],
                      [0.4,0.5,0.1]])
    Alpha = np.array([1,2,3])
    C_d = np.array([1,1,1,1])
    gamma_d = compute_gamma_d(phi_d,C_d,Alpha)
    assert np.allclose(gamma_d,[2.,3.4,4.6],atol=0.01)
    C_d = np.array([1,2,3,4])
    gamma_d = compute_gamma_d(phi_d,C_d,Alpha)
    assert np.allclose(gamma_d,[4,6,6],atol=0.01)
#---------------------------------------------------
def test_compute_Gamma():
    ''' (4 points) compute_Gamma'''
    # 3 topics, 4 words in each document, 2 documents
    Phi = np.array([[[0.1,0.2,0.7],
                     [0.2,0.3,0.5],
                     [0.3,0.4,0.3],
                     [0.4,0.5,0.1]],
                    [[0.8,0.1,0.1],
                     [0.3,0.3,0.4],
                     [0.4,0.5,0.1],
                     [0.1,0.4,0.5]]])
    Alpha = np.array([1,2,3])
    C = np.array([[1,2,3,4],
                  [1,1,1,1]])
    Gamma= compute_Gamma(Phi,C,Alpha)
    Gamma_true = [[4,6,6],
                  [2.6,3.3,4.1]]
    assert np.allclose(Gamma,Gamma_true,atol=0.01)
#---------------------------------------------------
def test_E_step():
    ''' (4 points) E_step'''
    # 2 documents, 4 words in the vocabulary, 3 topics
    C = np.array([[1,2,3,4], # word counts in the 1st document
                  [1,1,1,1]]) # word counts in the 2nd document
    Beta = np.array([[ 0.1, 0.4, 0.3, 0.2], # word probability distribution in the 1st topic
                     [ 0.2, 0.5, 0.1, 0.2], # word probability distribution in the 2nd topic
                     [ 0.3, 0.6, 0.1, 0.0]]) # word probability distribution in the 3rd topic
    Alpha = np.array([1,2,3])
    Phi, Gamma = E_step(C, Alpha, Beta,n_iter_var=1)
    Gamma_true = [[3.1882687 ,6.60828112,6.20345018],
                  [1.70328253,3.57655577,4.7201617 ]] 
    assert np.allclose(Gamma,Gamma_true,atol=0.01)
    Phi_true = [[[0.05029768,0.27344656,0.67625575],
                 [0.08992494,0.30555165,0.60452341],
                 [0.29411849,0.26649898,0.43938253],
                 [0.26894142,0.73105858,0.        ]],
                [[0.05029768,0.27344656,0.67625575],
                 [0.08992494,0.30555165,0.60452341],
                 [0.29411849,0.26649898,0.43938253],
                 [0.26894142,0.73105858,0.        ]]]
    assert np.allclose(Phi,Phi_true,atol=0.01)
    Phi, Gamma = E_step(C, Alpha, Beta,n_iter_var=2)
    Gamma_true = [[3.81690191,6.88364326,5.29945484],
                  [1.79102219,3.63961475,4.56936306]] 
    assert np.allclose(Gamma,Gamma_true,atol=0.01)
    Phi_true = [[[0.08430982,0.38140821,0.53428197],
                 [0.14293896,0.40414993,0.45291111],
                 [0.4068172 ,0.30673259,0.28645021],
                 [0.30656564,0.69343436,0.        ]],
                [[0.06143308,0.30737939,0.63118753],
                 [0.10794039,0.3375487 ,0.55451091],
                 [0.33607652,0.28025886,0.38366462],
                 [0.28557219,0.71442781,0.        ]]]
    assert np.allclose(Phi,Phi_true,atol=0.01)
#---------------------------------------------------
def test_compute_beta_t():
    ''' (4 points) compute_beta_t'''
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
    ''' (4 points) compute_Beta'''
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
def test_LDA():
    ''' (4 points) LDA'''
    # example of 2 topics, 4 words, 5 documents
    C = np.array([[8,8,1,1],
                  [1,1,5,5],
                  [2,2,9,9],
                  [7,7,1,1],
                  [3,3,8,8]])
    Alpha = np.array([1,1])
    count = 0
    for _ in range(2):
        Beta = LDA(C,Alpha)
        assert Beta.shape==(2,4)
        assert np.allclose(Beta.sum(axis=1),np.ones(2),atol=0.001)
        if np.allclose((Beta>0.3).astype(float).sum(axis=0),np.ones(4),atol=0.1):
            count+=1
    assert count>0
    # example of 3 topics, 6 words, 6 documents
    C = np.array([[8,8,1,1,1,1],
                  [7,7,1,1,2,2],
                  [1,1,5,5,1,1],
                  [2,2,9,9,1,1],
                  [1,1,2,2,9,9],
                  [3,3,1,1,8,8]])
    Alpha = np.array([1,1,1])
    count = 0
    for _ in range(2):
        Beta = LDA(C,Alpha)
        assert Beta.shape==(3,6)
        assert np.allclose(Beta.sum(axis=1),np.ones(3),atol=0.001)
        if np.allclose((Beta>0.2).astype(float).sum(axis=0),np.ones(6),atol=0.1):
            count+=1
    assert count>0

