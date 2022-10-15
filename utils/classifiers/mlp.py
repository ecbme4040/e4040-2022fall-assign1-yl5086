"""
Implementation of MLP.
"""

import numpy as np
from utils.layer_utils import DenseLayer, AffineLayer

class MLP:
    """
    MLP (Multilayer Perceptrons) with an arbitrary number of dense hidden layers, and a softmax loss function. 
    For a network with L layers, the architecture will be

    input >> (L - 1) DenseLayers  >> AffineLayer >> softmax_loss >> output

    Here "(L - 1)" indicate to repeat L - 1 times. 
    """

    def __init__(self, input_dim=3072, hidden_dims=[200, 200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """

        self.num_layers = len(hidden_dims) + 1
        dims = [input_dim] + hidden_dims

        # first dim - 1 layers are dense
        self.layers = [DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale) for i in range(len(dims) - 1)]
        # last layer (output) is affine
        self.layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        self.reg = reg
        self.velocities = None

    def forward(self, X):
        """
        Feed forward

        Inputs:
        - X: a numpy array of (N, D) containing input data

        Returns a numpy array of (N, K) containing prediction scores
        """
        ############################################################################
        # TODO: Feedforward
        ############################################################################
        # START OF Yufan's CODE
          
        # Input layer
        layer_out = self.layers[0].feedforward(X)
        # Following layers
        for i in range(1, self.num_layers):
            layer_out = self.layers[i].feedforward(layer_out)
            
        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

        return layer_out

    def loss(self, scores, labels):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        Do regularization for better model generalization.

        Inputs:
        - scores: a numpy array of (N, num_classes(a.k.a. K)) containing prediction scores 
        - labels (int): a numpy array of (N,) containing ground truth

        Return loss value(float)
        """

        loss = 0.0

        from ..layer_funcs import softmax_loss

        ############################################################################
        # TODO: Backpropogation
        ############################################################################
        # START OF Yufan's CODE
        
        # Compute cross-entropy loss and gradient for output
        loss, dout = softmax_loss(scores, labels) # scores is (N, K). labels is (N, )
        
        # Output layer
        dx = self.layers[self.num_layers - 1].backward(dout)
        # Backward layers
        for i in range(self.num_layers - 2, -1, -1):
            dx = self.layers[i].backward(dx)
        
        # raise NotImplementedError
        ############################################################################
        # TODO: Add L2 regularization
        ############################################################################

        squared_weights = 0.0 # initialize squared_weights
        for i in range(self.num_layers):
            squared_weights += np.power(self.layers[i].params['W'],2).sum()
        loss += 0.5 * self.reg * squared_weights
        
        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

        return loss

    def step(self, learning_rate=1e-5, optim='SGD', momentum=0.5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        Default learning rate = 0.00001, use SGD with momentum, momentum = 0.5.
        """

        # creates new dictionary with all parameters and gradients
        # naming rule l{i}_[W/b]: layer i, weights / bias
        params = {
            'l{}_'.format(i + 1) + name: param
            for i, layer in enumerate(self.layers)
            for name, param in layer.params.items()
        }
        grads = {
            'l{}_'.format(i + 1) + name: grad
            for i, layer in enumerate(self.layers)
            for name, grad in layer.gradients.items()
        }
        # final layout: 
        # params = {
        #     'l1_W': xxx, 'l1_b': xxx, 
        #     'l2_W': xxx, 'l2_b': xxx, 
        #     ..., 
        #     'lN_W': xxx, 'lN_b': xxx, 
        # }
        # grads likewise

        ############################################################################
        # TODO: Use SGD with momentum to update variables in layers
        # NOTE: Recall what we did for the TwoLayerNet
        ############################################################################
        # START OF Yufan's CODE

        # fetch the velocities stored from previous iteration and form a dictionary
        # if None (i.e. this is the first iteration), build velocities from scratch
        velocities = self.velocities or {name: np.zeros_like(param) for name, param in params.items()}
      
        # Add L2 regularization:
        reg = self.reg
        grads = {name: grad+ reg * params[name] for name, grad in grads.items()}
        
        for i in range(self.num_layers):
            # l{i+1}_W
            key_W = "l" + str(i+1) + "_W" # apply this key to the following dictionaries for calculation
            velocities[key_W] *= momentum
            velocities[key_W] += (learning_rate * grads[key_W])
            params[key_W] -= velocities[key_W]
            
            # l{i+1}_b
            key_b = "l" + str(i+1) + "_b" # apply this key to the following dictionaries for calculation
            velocities[key_b] *= momentum
            velocities[key_b] += (learning_rate * grads[key_b])
            params[key_b] -= velocities[key_b]
        
        # update parameters in layers (2 parameters W & b in each layer)
        # with an additional reg
        self.update_model((params, reg))
        # store the parameters for model saving
        self.params = params
        # store the velocities for the next iteration
        self.velocities = velocities

        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

    def predict(self, X):
        """
        Return the label prediction of input data

        Inputs:
        - X: (float) a tensor of shape (N, D)

        Returns: 
        - predictions: (int) an array of length N
        """
    
        predictions = np.zeros(X.shape[0])

        ############################################################################
        # TODO:
        # Implement the prediction function.
        # Think about how the model decides which class to choose.
        ############################################################################
        # START OF Yufan's CODE

        # Call out forward and softmax for softmaxed out
        from utils.classifiers.softmax import softmax
        temp = softmax(self.forward(X))

        # reverse onehot encoding (scores -> labels)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if(np.around(temp[i][j]) == 1):
                    predictions[i] = j # record non-zero index as label
        
        # raise NotImplementedError
        # END OF Yufan's CODE
        ############################################################################

        return predictions

    def update_model(self, model):
        """
        Update layers and reg with new parameters.
        """

        params, reg = model

        for i, layer in enumerate(self.layers):
            layer.update_layer({
                name.split('_')[1]: param for name, param in params.items()
                if name.startswith('l{}'.format(i + 1))
            })

        self.reg = reg

