import numpy as np
import matplotlib.pyplot as plt

#neuralnetwork\Scripts\Activate.bat

def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
    return params


'''
                                            Linear Hypothesis : Average of a random observation can be
                                            written as a linear combination of some observed predictor variables.

                                            Z(Linear Hypothesis) == (the weight matrix).(Input Matrix) + Bias vector
'''

'''        
        Order of Matrix = (no. of rows)x(no. columns)       
'''

'''
    Matrix Multiplication --> Between Dimensions
    Matrix Addition --> In same dimensions
'''


# Z (linear hypothesis) --> Z = W*X + b 
# W - weight matrix, b --> bias vector, X --> Input 


def sigmoid(Z):
    A = 1/(1+np.exp(np.dot(-1, Z)))
    cache = (Z)
    return A, cache

def forward_prop(X, params):
    A = X
    caches = []
    L = len(params)//2
    for i in range(1,L+1):
        A_prev = A
        
        #linear hypothesis
        Z = np.dot(params['W'+str(1)], A_prev) + params['b' + str(1)]
        
        linear_cache = (A_prev, params['W'+str(1)], params['b'+str(1)])
        
        A, activation_cache = sigmoid(Z)
        
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    return A, caches

def cost_function(A, Y):
    m = Y.shape[1]
    
    cost = (-1/m)*(np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), 1-Y.T)) 
    
    return cost