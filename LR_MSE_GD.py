# Implementation of vanilla gradient descent to minimize MSE function

import numpy as np
from sklearn.datasets import make_regression

def mse(x, theta, y):
    """
   Defines cost MSE over a dataset x

   Parameters:
       x (2-D list or numpy array): samples (x_i)
       theta (list or numpy array) 1-D: The weights for the function
       y (list or numpy array) 1-D: Truth values
   Returns:
       summation (float): total cost over the dataset
   """
    y_prev = np.dot(x, theta)
    return np.mean((y - y_prev)**2)

def mse_gradient(x, theta, y):
    """
   Defines gradient of MSE function

   Parameters:
       x (2-D list or numpy array): samples (x_i)
       theta (list or numpy array) 1-D: The weights for the function
       y (list or numpy array) 1-D: Truth values
   Returns:
       summation (float): Gradient value in point
   """
    y_prev = np.dot(x, theta)
    return 2/len(x)*(y - y_prev) @ (-x)

q = 3 # Number of features
alpha = 0.01 # Learning rate of gradient descent(step)
itr = 300 # Number of iterations
np.random.seed(11) # Random seed
theta = np.random.rand(q+1) # Random starting values of weights (w0, w1)

def gradient_descent(x, y, theta, itr, alpha):
    """
   Minimize MSE cost and updates global theta

   Parameters:
       y (list or numpy array) 1-D: Truth values
       x (list or numpy array) 2-D: dataset.
       alpha (float): learning rate
   Returns:
       prev_thetas: List of gradient descent temporary thetas
       prev_mse: value (float) of minimal MSE
   """
    x = np.c_[np.ones(x.shape[0]), x] # Append ones feature for matrix operations
    prev_mse = [] # MSE results with different weights 
    prev_thetas = [theta] # Model coefs
    for i in range(itr):
        prev_mse.append(mse(x, theta, y)) # Add MSE value with current weights
        theta = theta - (alpha * mse_gradient(x, theta, y)) # Recalculating theta
        prev_thetas.append(theta) # Change temporary weight
    return prev_thetas, prev_mse

# Sample of getting weights and mse
x,y=make_regression(1000,n_features=3,n_targets=1) # Creating data
prev_thetas, prev_mse = gradient_descent(x, y, theta, itr, alpha)
print(prev_thetas[-1]) # Linear regression model weights
print(prev_mse[-1]) # Model MSE value
