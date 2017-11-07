"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class Layer(object):
    """
    A layer object that handles feed-forward and back propagation ops."""

    def __init__(self, W, b, activation, k):
        self.W = W
        self.b = b
        self.activation = activation
        self.k = k

    def forward(self, Z):
        assert Z.shape[0] == self.W.shape[1]

        return np.dot(self.W, Z) + self.b

    def backward(self, delta):
        # compute gradients

        # apply update

        raise NotImplementedError

    def _update(self, grads, eta):
        assert grads.shape == self.W.shape, 'Gradient shape {} must match W shape {}'.format(
            grads.shape, self.W.shape)
        self.W -= eta * grads

    # keras style printing
    def __repr__(self):
        return "W_{} : {} x {}".format(self.k, *self.W.shape)


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform inference, training and it
    can also be used for evaluating prediction performance.
    """

    def __init__(self,
                 n_hidden,
                 n_classes,
                 weight_decay=0.0,
                 weight_scale=0.0001):
        """
        Constructor for an MLP object. Default values should be used as hints for
        the usage of each parameter. Weights of the linear layers should be initialized
        using normal distribution with mean = 0 and std = weight_scale. Biases should be
        initialized with constant 0. All activation functions are ReLUs.

        Args:
          n_hidden: list of ints, specifies the number of units
                         in each hidden layer. If the list is empty, the MLP
                         will not have any hidden units, and the model
                         will simply perform a multinomial logistic regression.
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the MLP.
          weight_decay: L2 regularization parameter for the weights of linear layers.
          weight_scale: scale of normal distribution to initialize weights.

        """
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.weight_scale = weight_scale
        self.input_dim = 3 * 32 * 32

        # infer shapes of each layer: dimension-first notation W.shape=(D_k+1, D_k)
        W_shapes = [self.input_dim] + n_hidden + [self.n_classes]
        W_shapes = [(W_shapes[i + 1], W_shapes[i]) for i in range(len(W_shapes) - 1)]
        bias_shapes = [(shape[0],1) for shape in W_shapes]

        # initialize weights
        self.weights = [
            self._get_init_weight(shape, self.weight_scale) for shape in W_shapes
        ]

        # initialize biases
        self.biases = [self._get_bias(dim) for dim in bias_shapes]

        def relu(x):
            return x * (x > 0)

        def linear(x):
            return x

        self.activations = [relu for _ in n_hidden] + [linear]

        # layer wrapper objects
        self.layers = [
            Layer(W=W, b=b, activation=a, k=i)
            for i, (W, b, a) in enumerate(
                list(zip(self.weights, self.biases, self.activations)))
        ]

    def _get_init_weight(self, shape, weight_scale):
        return np.random.normal(0, weight_scale, shape)

    def _get_bias(self, shape, epsilon=1e-3):
        return np.zeros(shape=shape) + epsilon

    def inference(self, x):
        """
        Performs inference given an input array. This is the central portion
        of the network. Here an input array is transformed through application
        of several hidden layer transformations (as defined in the constructor).
        We recommend you to iterate through the list self.n_hidden in order to
        perform the sequential transformations in the MLP. Do not forget to
        add a linear output layer (without non-linearity) as the last transformation.

        It can be useful to save some intermediate results for easier computation of
        gradients for backpropagation during training.
        Args:
          x: 2D float array of size [batch_size, input_dimensions]

        Returns:
          logits: 2D float array of size [batch_size, self.n_classes]. Returns
                 the logits outputs (before softmax transformation) of the
                 network. These logits can then be used with loss and accuracy
                 to evaluate the model.
        """

        self.results = []

        # dim-first convention
        Z = x.T

        # feed-forward through each layer and keep results
        for layer in self.layers:
            Z = layer.forward(Z)
            self.results.append(Z)

        logits = self.results[-1]

        return logits

    def loss(self, logits, labels):
        """
        Computes the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.

        It can be useful to compute gradients of the loss for an easier computation of
        gradients for backpropagation during training.

        Args:
          logits: 2D float array of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int array of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       sample in the batch.
        Returns:
          loss: scalar float, full loss = cross_entropy + reg_loss
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return loss

    def train_step(self, loss, flags):
        """
        Implements a training step using a parameters in flags.
        Use mini-batch Stochastic Gradient Descent to update the parameters of the MLP.

        Args:
          loss: scalar float.
          flags: contains necessary parameters for optimization.
        Returns:

        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return

    def accuracy(self, logits, labels):
        """
        Computes the prediction accuracy, i.e. the average of correct predictions
        of the network.

        Args:
          logits: 2D float array of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int array of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each sample in the batch.
        Returns:
          accuracy: scalar float, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return accuracy

    def __repr__(self):
        sep = '\n|' + '--' * 15 + '\n|\t'

        defs = ['X : {} x batch_size'.format(3 * 32 * 32)] + [str(layer) for layer in self.layers] + [
            'y:{}'.format(self.n_classes)]

        return sep.join(['|\tNetwork Overview'] + defs) + '\n' + '--'*15


def test():
    net = MLP(n_hidden=[100, 200], n_classes=10)
    print(net)

    X = np.random.rand(256, 3*32*32)
    logits = net.inference(X)
    print(logits.shape)

if __name__ == '__main__':
    test()
