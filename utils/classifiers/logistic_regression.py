"""
Implementations of logistic regression. 
"""

import numpy as np

def logistic_regression_loss_naive(w, X, y, reg):
    """
    Logistic regression loss function, naive implementation (with loops over N samples)

    NOTE:
    In this function, you are NOT supposed to use functions like:
    - np.dot
    - np.matmul (or operator @)
    - np.linalg.norm
    You can (not necessarily) use functions like:
    - np.sum
    - np.log
    - np.exp

    Use this linear classification method to find optimal decision boundary.

    Inputs have dimension D, there are K classes, and we operate on minibatches
    of N examples.

    Inputs:
    - w: (float) a numpy array of shape (D + 1, 1) containing weights.
    - X: (float) a numpy array of shape (N, D + 1) containing a minibatch of data.
    - y: (uint8) a numpy array of shape (N, 1) containing training labels; y[i] = k means 
        that X[i] has label k, where k can be either 0 or 1.
    - reg: (float) regularization strength. For regularization, we use L2 norm.

    Returns a tuple of:
    - loss: (float) the mean value of loss functions over N samples in minibatch.
    - gradient: (float) an array of same shape as w
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dw = np.zeros_like(w)
    # Get input size
    N = X.shape[0] # how many data samples
    D = X.shape[1] - 1 # how many dimensions (including the bias)

    # Make prediction
    temp = np.zeros((N, 1)) # initialization counter
    for i in range(N): # iterate through rows of X
        # w has only one column
            for d in range(D+1): 
                temp[i] += X[i][d] * w[d] # temp is (N, 1) matrix
    
    # Apply sigmoid
    pred = 1/(1 + np.exp(-temp)) # pred is (N, 1) matrix
    
    # Calculate loss
    for i in range(N):
        loss -= (y[i])*np.log(pred[i]) + (1-y[i])*np.log(1-pred[i])
    loss = loss.item() # the result above is a 1×1 matrix, here tramsform to scalar
    
    # Calculate gradient
    for d in range(D+1): # iterate through rows of XT
        # pred-y has only one column
            for i in range(N): 
                dw[d] += X.transpose()[d][i] * (pred[i] - y[i]) # dw is (D+1, 1) matrix
    
    # Apply regularization
    loss = loss/N + 0.5 * reg * np.sum(w * w)
    dw = dw/N + reg * w
         
    #raise NotImplementedError
    return loss, dw

def sigmoid(x):
    """
    Sigmoid function.

    Inputs:
    - x: (float) a numpy array of shape (N,)

    Returns:
    - h: (float) a numpy array of shape (N,), containing the element-wise sigmoid of x
    """
    # TODO: Implement sigmoid function.
    h = np.zeros_like(x)
    h = 1/(1 + np.exp(-x))
           
    # raise NotImplementedError
    return h 

def logistic_regression_loss_vectorized(w, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    NOTE:
    In this function, you CAN (not necessarily) use functions like:
    - np.dot (unrecommanded)
    - np.matmul
    - np.linalg.norm
    You MUST use the functions you wrote above:
    - sigmoid

    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dw = np.zeros_like(w)
    # Get input size
    N = X.shape[0] # how many data samples

    # Make prediction (each sample)
    pred = sigmoid(np.matmul(X, w)) # pred is (N, 1) matrix
    # Calculate loss and gradient
    loss = - np.matmul(y.transpose(),np.log(pred)) - np.matmul((1-y).transpose(),np.log(1-pred)) # loss is a scalar
    dw = np.matmul(X.transpose(),(pred - y)) # dw is (D+1, 1) matrix
    # Apply regularization to loss
    loss = loss/N + 0.5 * reg * np.sum(w * w)
    dw = dw/N + reg*w
      
    #raise NotImplementedError
    return loss, dw