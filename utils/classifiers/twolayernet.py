"""
Implementation of a two-layer network. 
"""

import numpy as np
from utils.layer_utils import AffineLayer, DenseLayer

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a Leaky_ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:
    input -> DenseLayer -> AffineLayer -> softmax -> output
    Or more detailed,
    input -> affine transform -> Leaky_ReLU -> affine transform -> softmax -> output

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """

        # instantiate a DenseLayer (layer1) object as input layer
        self.layer1 = DenseLayer(input_dim, hidden_dim, weight_scale=weight_scale)
        # instantiate a AffineLayer (layer2) object as input layer
        self.layer2 = AffineLayer(hidden_dim, num_classes, weight_scale=weight_scale)
        self.reg = reg
        self.velocities = None

    def forward(self, X):
        """
        Feed forward

        Inputs:
        - X: a numpy array of (N, D) containing input data

        Returns:
        - layer2_out: a numpy array of (N, num_classes(a.k.a. K)) containing prediction scores
         NOTE: the scores should not be softmaxed.
        """
        ############################################################################
        # TODO: Feedforward
        # NOTE: Use the methods defined for the layers in layer_utils.py
        ############################################################################
        # START OF Yufan's CODE
        
        ## Step1: input -> layer1 -> layer1_out
        layer1_out = self.layer1.feedforward(X) # given input, get layer1 output
        
        ## Step2: layer1_out -> layer2 ->layer2_out
        layer2_out = self.layer2.feedforward(layer1_out) # get layer2 output
        
        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

        return layer2_out

    def loss(self, scores, labels):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        Do regularization for better model generalization.

        Inputs:
        - scores: a numpy array of (N, num_classes(a.k.a. K)) containing prediction scores 
        - labels (int): a numpy array of (N,) containing ground truth

        Return loss value (float)
        """

        loss = 0.0

        from ..layer_funcs import softmax_loss

        ############################################################################
        # TODO:
        # Backpropogation, here is just one dense layer, it should be pretty easy
        # NOTE: Use the methods defined for each layer, and you no longer need to
        # mannually cache the parameters because it would be taken care of by the
        # functions in layer_utils.py
        ############################################################################
        # START OF Yufan's CODE
        
        # Calculate the cross-entropy loss and gradient at layer2 output
        loss, dout = softmax_loss(scores, labels) # scores is (N, K). labels is (N, )
        
        # back propagation at layer2
        dx_2 = self.layer2.backward(dout)
        # dW_2, db_2 = self.layer2.gradients['W'], self.layer2.gradients['b']
        # ↑ no need to get gradients here
        
        # back propagation at layer1
        dx_1 = self.layer1.backward(dx_2)
        # dW_1, db_1 = self.layer1.gradients['W'], self.layer1.gradients['b']
        # ↑ no need to get gradients here
        
        # END OF Yufan's CODE
        # raise NotImplementedError
        ############################################################################

        # Add L2 regularization
        squared_weights = np.power(self.layer1.params['W'],2).sum() + np.power(self.layer2.params['W'],2).sum()
        loss += 0.5 * self.reg * squared_weights

        return loss

    def step(self, learning_rate=1e-5, optim='SGD', momentum=0.5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        Default learning rate = 0.00001, use SGD with momentum, momentum = 0.5.
        """

        # fetch the parameters from layer cache and form a dictionary
        params = {
            'l1_W': self.layer1.params['W'], 
            'l1_b': self.layer1.params['b'], 
            'l2_W': self.layer2.params['W'], 
            'l2_b': self.layer2.params['b']
        }
        grads = {
            'l1_W': self.layer1.gradients['W'], 
            'l1_b': self.layer1.gradients['b'], 
            'l2_W': self.layer2.gradients['W'], 
            'l2_b': self.layer2.gradients['b']
        }

        # fetch the velocities stored from previous iteration and form a dictionary
        # if None (i.e. this is the first iteration), build velocities from scratch
        velocities = self.velocities or {name: np.zeros_like(param) for name, param in params.items()}

        # Add L2 regularization:
        reg = self.reg
        grads = {name: grad+ reg * params[name] for name, grad in grads.items()}

        ############################################################################
        # TODO:
        # Use SGD or SGD with momentum to update variables in layer1 and layer2
        # NOTE: iterate through all the parameters and do the update one by one
        ############################################################################
        # START OF Yufan's CODE

        for i in range(2):
            # l{i+1}_W
            key_W = "l" + str(i+1) + "_W" # apply this key to the following dictionaries for calculation
            params[key_W] = params[key_W] - (learning_rate * grads[key_W])
            
            # l{i+1}_b
            key_b = "l" + str(i+1) + "_b" # apply this key to the following dictionaries for calculation
            params[key_b] = params[key_b] - (learning_rate * grads[key_b])
        
        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

        # update parameters in layers (2 parameters W & b in each layer)
        # with an additional reg
        self.update_model((params, reg))
        # store the parameters for model saving
        self.params = params
        # store the velocities for the next iteration
        self.velocities = velocities

    def predict(self, X):
        """
        Return the label prediction of input data

        Inputs:
        - X: (float) a tensor of shape (N, D)

        Returns: 
        - preds: (int) an array of labels, shape (N, 1)
        """

        preds = np.zeros(X.shape[0])

        ############################################################################
        # TODO: generate predictions
        ############################################################################
        # START OF Yufan's CODE

        # Call out forward and softmax for softmaxed out
        from utils.classifiers.softmax import softmax
        temp = softmax(self.forward(X))
        
        # reverse onehot encoding (scores -> labels)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if(np.around(temp[i][j]) == 1):
                    preds[i] = j # record non-zero index as label
        
        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

        return preds

    def save_model(self):
        """
        Save model's parameters, including two layers' W and b and reg.
        """
        return self.params, self.reg

    def update_model(self, model):
        """
        Update layers and reg with new parameters.
        """

        params, reg = model

        # filter out the parameters for layer 1
        # update the weights in layer 1 by calling layer1.update()
        self.layer1.update_layer({
            name.split('_')[1]: param for name, param in params.items()
            if name.startswith('l1')
        })
        # likewise
        self.layer2.update_layer({
            name.split('_')[1]: param for name, param in params.items()
            if name.startswith('l2')
        })
        self.reg = reg
