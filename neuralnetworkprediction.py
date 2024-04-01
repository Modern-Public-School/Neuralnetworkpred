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

# Z (linear hypothesis) - Z = W*X + b , 
# W - weight matrix, b- bias vector, X- Input 

def sigmoid(Z):
    A = 1/(1+np.exp(np.dot(-1, Z)))
    cache = (Z)
    return A, cache