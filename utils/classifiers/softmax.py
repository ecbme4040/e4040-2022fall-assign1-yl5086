"""
Implementation of softmax classifer.
"""

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops over N samples)

    NOTE:
    In this function, you are NOT supposed to use functions like:
    - np.dot
    - np.matmul (or operator @)
    - np.linalg.norm
    You can (not necessarily) use functions like:
    - np.sum
    - np.log
    - np.exp

    Inputs have dimension D, there are K classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D + 1, K) containing weights.
    - X: a numpy array of shape (N, D + 1) containing a minibatch of data.
    - y: a numpy array of shape (N,) containing training labels; y[i] = k means 
        that X[i] has label k, where 0 <= k < K.
    - reg: regularization strength. For regularization, we use L2 norm.

    Returns a tuple of:
    - loss: the mean value of loss functions over N examples in minibatch.
    - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = []
    # Initialize the gradient to zero
    dW = np.zeros_like(W)
    # Get input size    
    N = X.shape[0] # how many data samples
    D = X.shape[1] - 1 # how many dimensions (excluding the bias)
    K = W.shape[1] # how many classes
    # Apply onehot encoding to y
    y_onehot = np.zeros((N, K)) # initialize onehot array
    for i in range(N):
        y_onehot[i, y[i]] = 1 # put a "1" in its place (y[i]) as label
    
    # Make prediction
    temp = np.zeros((N, K)) # initialization counter
    for i in range(N): # iterate through rows of X
        for k in range(K): # iterate through columns of W
            for d in range(D+1): 
                temp[i][k] += X[i][d] * W[d][k] # temp is (N, K) matrix
    
    # Apply softmax
    pred = np.zeros((N, K)) # initialization, pred is (N, K) matrix
    pred = np.exp(temp - np.max(temp)) # subtracting the max of z for numerical stability
    for i in range(N):
        pred[i] /= np.sum(pred[i])
    
    # Calculate loss
    loss = - np.sum(y_onehot * np.log(pred), axis=1).sum() # loss is a scalar
    
    # Calculate gradient
    for d in range(D+1): # iterate through rows of XT
        for k in range(K): # iterate through columns of pred - y_onehot
            for i in range(N): 
                dW[d][k] += X.transpose()[d][i] * (pred[i][k] - y_onehot[i][k]) # dW is (D+1, K) matrix
    
    # Apply regularization
    loss = loss/N + 0.5 * reg * np.sum(W * W)
    dW = dW/N + reg * W

    # raise NotImplementedError
    return loss, dW

def softmax(x):
    """
    Softmax function, vectorized version

    Inputs
    - x: (float) a numpy array of shape (N, C), containing the data

    Return a numpy array
    - h: (float) a numpy array of shape (N, C), containing the softmax of x
    """
    # TODO: Implement softmax function.
    N = x.shape[0] # get input size
    h = np.zeros_like(x) # initialization
    h = np.exp(x - np.max(x)) # subtracting the max of z for numerical stability
    for i in range(N):
        h[i] /= np.sum(h[i])
    
    # raise NotImplementedError
    return h

def onehot(y, K):
    """
    One-hot encoding function, vectorized version.

    Inputs
    - y: (uint8) a numpy array of shape (N, 1) containing labels; y[i] = k means 
        that X[i] has label k, where 0 <= k < K.
    - K: total number of classes

    Returns a numpy array
    - y_onehot: (float) the encoded labels of shape (N, K)
    """
    # TODO: Implement the one-hot encoding function.
    N = y.shape[0] # get input size
    y_onehot = np.zeros((N, K)) # initialize onehot array
    for i in range(N):
        y_onehot[i, y[i].astype(int)] = 1 # put a "1" in its place (y[i]) as label

    # raise NotImplementedError
    return y_onehot


def cross_entropy(true, pred):
    """
    Cross entropy function, vectorized version.

    Inputs:
    - true: (float) a numpy array of shape (N, K), containing ground truth labels
    - pred: (float) a numpy array of shape (N, K), containing predictions

    Returns:
    - h: (float) a numpy array of shape (N,), containing the cross entropy of 
        each data point
    """
    # TODO: Implement cross entropy function.
    h = np.zeros(true.shape[0])
    h = - np.sum(true * np.log(pred), axis=1)
    
    # raise NotImplementedError
    return h


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    NOTE:
    In this function, you CAN (not necessarily) use functions like:
    - np.dot (unrecommanded)
    - np.matmul (or operator @)
    - np.linalg.norm
    You MUST use the functions you wrote above:
    - onehot
    - softmax
    - crossentropy

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)
    # Get input size    
    N = X.shape[0] # how many data samples
    K = W.shape[1] # how many classes
    # Apply onehot encoding to y
    y_onehot = onehot(y,K)

    # Make prediction
    pred = softmax(np.matmul(X, W)) # pred is (N, K) matrix
    # Calculate loss and gradient
    loss = cross_entropy(y_onehot, pred).sum() # loss is a scalar
    # Note that: we need to use sum here ↑ because of onehot encoding
    dW = np.matmul(X.transpose(),(pred - y_onehot)) # dw is (D+1, K) matrix
    
    # Apply regularization
    loss = loss/N + 0.5 * reg * np.sum(W * W)
    dW = dW/N + reg * W

    # raise NotImplementedError
    return loss, dW
