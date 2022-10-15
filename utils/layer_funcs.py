"""
Implementation of layer functions.
"""

import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine transformation function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ############################################################################
    # TODO:
    # Implement the affine forward pass. Store the result in 'out'.
    # You will need to reshape the input into rows.
    ############################################################################
    # START OF Yufan's CODE 

    N = x.shape[0] # how many input images
    x_reshape = np.reshape(x,[N, -1]) # x_reshape is (N. D)
    out = np.matmul(x_reshape, w) + b # w is (D. M)

    # raise NotImplementedError
    # END OF Yufan's CODE
    ############################################################################

    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine transformation function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ############################################################################
    # TODO: Implement the affine backward pass.
    ############################################################################
    # START OF Yufan's CODE
    
    N = x.shape[0] # how many input images
    x_reshape = np.reshape(x,[N, -1]) # x_reshape is (N. D)

    # Calculate dx = dout(N, M) * wT(M, D)
    dx = np.matmul(dout, w.transpose()) # dx is (N, D)
    dx = np.reshape(dx, x.shape) # reshape back to shape of x (N, d_1, ..., d_k)
    # Calculate dw = x_reshapeT(D, N) * dout(N, M)
    dw = np.dot(x_reshape.transpose(),dout) # dw is (D, M)
    # Calculate db = sum dout over each image
    db = np.sum(dout, axis=0) # db is (N, 1)

    # raise NotImplementedError
    # END OF Yufan's CODE
    ############################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs) activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ############################################################################
    # TODO: Implement the ReLU forward pass.
    ############################################################################
    # START OF Yufan's CODE

    out = x.copy() # use copy in numpy to ensure pass by value
    out[out < 0] = 0 # assign 0 to all negative x

    # raise NotImplementedError
    # END OF Yufan's CODE
    ############################################################################

    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs) activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ############################################################################
    # TODO: Implement the ReLU backward pass.
    ############################################################################
    # START OF Yufan's CODE

    dx = dout * (x >= 0) # only backprop to non-negative x elements

    # raise NotImplementedError
    # END OF Yufan's CODE 
    ############################################################################

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x))

    Inputs:
    - x: (float) a tensor of shape (N, K)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradient of loss w.r.t input x
    """

    # When calculating the cross entropy,
    # you may meet another problem about numerical stability, log(0)
    # to avoid this, you can add a small number to it, log(0+epsilon)
    epsilon = 1e-15

    from .classifiers.softmax import softmax, onehot, cross_entropy

    ############################################################################
    # TODO:
    # You can use the previous softmax loss function here.
    # Hint:
    #   * Be careful of overflow problem.
    #   * You may use the functions you wrote in task1
    ############################################################################
    # START OF Yufan's CODE
    
    N = x.shape[0] # how many input images
    K = x.shape[1] # how many classes
    y_onehot = onehot(y,K) # y onehot encoding

    # Calculate loss
    loss = cross_entropy(y_onehot, softmax(x)).sum() # loss is a scalar
    loss /= N
    # Calculate gradient
    dx = softmax(x) # dx is (N, K)
    dx[range(N),y] -= 1 # prediction - true label
    dx /= N    

    # raise NotImplementedError
    # END OF Yufan's CODE
    ############################################################################

    return loss, dx


def check_accuracy(preds, labels):
    """
    Return the classification accuracy of input data.

    Inputs:
    - preds: (float) a tensor of shape (N,)
    - y: (int) an array of length N. ground truth label 
    Returns: 
    - acc: (float) between 0 and 1
    """

    return np.mean(np.equal(preds, labels))
