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

        S_k = np.dot(self.W, Z) + self.b
        Z_k = self.activation(S_k)
        return Z_k

    def backward(self, delta, Z_prev, flags):
        # compute gradients
        dL_dW = delta.dot(Z_prev.T)
        dL_db = delta.sum(axis=1, keepdims=True)

        # apply updates
        self.W -= flags['learning_rate'] * dL_dW
        self.b -= flags['learning_rate'] * dL_db

    def _update(self, grads, eta):
        assert grads.shape == self.W.shape, 'Gradient shape {} must match W shape {}'.format(
            grads.shape, self.W.shape)
        self.W -= eta * grads

    def nlog_prior(self):
        """
        Log prior over each weight scaler N(0,1)
        :return: log( prod_i[ p(w_i) ]) = sum_i[ log N(w_i | 0,1)] = -0.5 * sum_i [w_i ** 2]
        """
        return 0.5 * (self.W ** 2).sum()

    # keras style printing
    def __repr__(self):
        return "W_{} : {} x {}, f: {}\n|\tZ_{} : {} x batch_size".format(self.k, *self.W.shape, self.activation.__name__,
                                                                      self.k, self.W.shape[0])


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
                 weight_scale=0.0001,
                 input_dim=3 * 32 * 32):
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
        self.input_dim = input_dim

        # infer shapes of each layer: dimension-first notation W.shape=(D_k+1, D_k)
        W_shapes = [self.input_dim] + n_hidden + [self.n_classes]
        W_shapes = [(W_shapes[i + 1], W_shapes[i]) for i in range(len(W_shapes) - 1)]
        bias_shapes = [(shape[0], 1) for shape in W_shapes]

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

        # relu activations for all hidden layers, linear for final layer
        self.activations = [relu for _ in n_hidden] + [linear]

        # layer wrapper objects
        self.layers = [
            Layer(W=W, b=b, activation=a, k=i)
            for i, (W, b, a) in enumerate(
                list(zip(self.weights, self.biases, self.activations)))
        ]

        # For caching
        self.ff_cache = []
        self.delta_out = None

    def _relu_gradient(self, X):
        """
        Computes the gradient of the relu activation

        dY/dXij =
                1 if Xij > 0
                0 else

        :param X:
        :return:
        """
        return np.where(X > 0, 1, 0)

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

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.ff_cache.clear()

        # dim-first convention
        Z = x.T
        self.ff_cache += [Z]

        # feed-forward through each layer and keep results
        for layer in self.layers:
            Z = layer.forward(Z)
            self.ff_cache.append(Z)

            if not np.isfinite(Z).all():
                print('WARNING: NaN encountered in feed forward')

        logits = Z.T

        if not np.isfinite(logits).all():
            print('WARNING: NaN encountered in logits')

        ########################
        # END OF YOUR CODE    #
        #######################

        return logits

    def _softmax2D(self, logits):
        """
        Performs a softmax transformation over logits. Maximum normalization is used for numerical stability (equivalent to log-sum-exp)

        :param logits: output of final (hidden) layer [n_classes, batch_size]

        :return: class probabilities [n_classes, batch_size]
        """

        # subtract maximum logit per mini-batch for numerical stability
        max_per_class = np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits - max_per_class)

        normalizer = e.sum(axis=1, keepdims=True)
        probs = e / normalizer
        return probs

    def _weight_complexity_cost(self):
        """
        Computes the complexity cost of the network parameters.
        We define a prior over each parameter as w_i ~ N(0,1), for all parameters/weights indexed by i and are assumed to be mutually-independent.
        This complexity cost is the negative log prior over the weights, and optimizing it corresponds to obtaining a MAP solution for the network weights.

        :return: scalar complexity cost
        """
        return sum([layer.nlog_prior() for layer in self.layers])

    def _cross_entropy_loss(self, pred_class_probs, labels):
        """
        Computes the cross-entropy loss between predictions and labels (one-hot vectors)

        For each datapoint x with true class label c among K classes, we obtain a predicted class probability vector y

        p: dirac's delta function centered on c (here, a one-hot encoding of K classes)
        q: softmax probability over K classes

        cross-entropy loss, computed on one-hot encodings
        H[p,q] = - sum_k p(y_k) * log q(y_k) = - log q(y_c)

        Note that since we use the max-trick to compute the softmax, we are essentially applying the log-sum-exp trick

        :param pred_class_probs: predicted class probabilities. 2D float array of size [batch_size, self.n_classes].
        :param labels: true class probabilities as 1-hot vectors. 2D int array [batch_size, n_classes]

        :return: cross-entropy loss, scalar float
        """

        true_class_probs = pred_class_probs * labels
        true_class_probs = np.sum(true_class_probs, axis=1) + 1e-6 # reduce
        loss = - np.log(true_class_probs).sum()

        return loss

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

        batch_size = logits.shape[0]
        class_probs = self._softmax2D(logits)

        # regularization cost
        nl_prior = self._weight_complexity_cost()

        nl_likelihood = self._cross_entropy_loss(class_probs, labels)

        loss = nl_likelihood + self.weight_decay * nl_prior
        loss *= 1/batch_size

        # delta_out = dL/dY_out * dY_out/dS_out
        self.delta_out = (1/batch_size) * (class_probs - labels).T

        if not np.isfinite(loss).all():
            print('WARNING: NaN encountered in loss')

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

        # delta_out computes in loss function
        delta_k = self.delta_out

        # loop backward through layers K --> 1
        for k in list(range(len(self.weights)))[::-1]:

            Z_prev = self.ff_cache[k]  # Z_{k-1}

            # backward pass through layer k
            self.layers[k].backward(delta_k, Z_prev, flags)

            if k > 0:
                # delta_{k-1} = [W_{k}.dot(delta_{k})] * dZ/dS_k
                delta_k = (self.weights[k].T.dot(delta_k)) * self._relu_gradient(Z_prev)

        # clear
        self.delta_out = None

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

        batch_size = logits.shape[0]
        class_preds = np.zeros_like(logits)

        # top predicted class per datapoint in minibatch
        top_class = np.argmax(logits, axis=1)

        # create one-hot matrix of predicted class
        class_preds[np.arange(batch_size), top_class] = 1.

        # correct predictions
        correct_preds = class_preds * labels

        # total number of correct preds: sum of values in matrix
        accuracy = correct_preds.sum() / batch_size

        ########################
        # END OF YOUR CODE    #
        #######################

        return accuracy

    def __repr__(self):
        sep = '\n|' + '--' * 15 + '\n|\t'

        defs = ['X : {} x batch_size'.format(3 * 32 * 32)] + [str(layer) for layer in self.layers] + [
            'Y : {} x batch_size'.format(self.n_classes)]

        return sep.join(['|\tNetwork Overview'] + defs) + '\n' + '--' * 15


def test():


    batch_size = 256
    n_classes = 10
    input_dim = 3 * 32 * 32

    net = MLP(n_hidden=[100], n_classes=10, input_dim=input_dim)
    print(net)

    X = np.random.standard_normal((batch_size, input_dim))
    Y = np.zeros((batch_size, n_classes))
    Y[np.arange(batch_size), np.random.choice(10, size=batch_size)] = 1

    logits = net.inference(X)
    print('logits shape = ', logits.shape)

    complexity_cost = net._weight_complexity_cost()
    print('complexity cost', complexity_cost)

    pred_probs = net._softmax2D(logits)
    crossent_loss = net._cross_entropy_loss(pred_class_probs=pred_probs, labels=Y)
    print('X entropy loss', crossent_loss)

    loss = net.loss(logits, Y)
    print('Total loss', loss)

    accuracy = net.accuracy(logits, Y)
    print('Accuracy', accuracy)

    train_flags = {'learning_rate': 1e-3}
    net.train_step(loss=loss, flags=train_flags)


if __name__ == '__main__':
    test()
