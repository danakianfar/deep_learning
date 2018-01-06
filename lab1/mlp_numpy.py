"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


# import seaborn as sns


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
        weights = [
            self._get_init_weight(shape, self.weight_scale) for shape in W_shapes
        ]

        # initialize biases
        biases = [self._get_init_bias(dim) for dim in bias_shapes]

        def relu(x):
            return x * (x > 0)

        def linear(x):
            return x

        # relu activations for all hidden layers, linear for final layer
        activations = [relu for _ in n_hidden] + [linear]

        # layer wrapper objects
        self.layers = [
            Layer(W=W, b=b, activation=a, k=i, parent=self)
            for i, (W, b, a) in enumerate(
                list(zip(weights, biases, activations)))
        ]

        # For caching and debugging
        self.activation_cache = []
        self.preactivation_cache = []
        self.delta_out = None
        self.debug_stats = defaultdict(list)
        self.training_mode = True

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

        self.activation_cache.clear()
        self.preactivation_cache.clear()

        # dim-first convention: z = WX+b
        Z = x.T
        self.activation_cache += [Z]

        # feed-forward and caching
        for layer in self.layers:
            Z, S = layer.forward(Z)

            self.activation_cache += [Z]
            self.preactivation_cache += [S]

        logits = Z.T  # to match label size for computing accuracy/loss

        if not np.isfinite(logits).all():
            print('WARNING: NaN encountered in logits')

        # Collect debug stats
        self._dump_training_stats('logits_norm', np.linalg.norm(logits))

        ########################
        # END OF YOUR CODE    #
        #######################

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

        batch_size = logits.shape[0]
        class_probs = self._softmax2D(logits)
        self.logits = logits
        self.class_probs = class_probs

        # individual losses
        nl_prior = (1. / batch_size) * self._weight_complexity_cost()
        nl_likelihood = (1. / batch_size) * self._cross_entropy_loss(class_probs, labels)

        # full loss
        loss = nl_likelihood + self.weight_decay * nl_prior

        # Caching
        # delta_out = dL/dY_out * dY_out/dS_out
        self.delta_out = (class_probs - labels).T
        self.delta_out *= (1. / batch_size)

        # Debugging
        if not np.isfinite(loss).all():
            print('WARNING: NaN encountered in loss')

        self._dump_training_stats('nl_likelihood', nl_likelihood)
        self._dump_training_stats('nl_prior', nl_prior)

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
        deltas = [self.delta_out]
        flags['weight_decay'] = self.weight_decay

        # Compute deltas
        # loop backward through layers K --> 1 (not input layer 0)
        for k in list(range(len(self.layers)))[:0:-1]:
            # for lower layers
            # delta_{k-1} = [W_{k}.dot(delta_{k})] * dZ/dS_k
            delta_k = (self.layers[k].W.T.dot(deltas[-1])) * self.layers[k - 1].activation_grad()
            deltas = [delta_k] + deltas

            self._dump_training_stats('delta_{}_norm'.format(k), np.linalg.norm(delta_k))

        # Apply updates
        [self.layers[k].backward(deltas[k], flags) for k in range(len(self.layers))]

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

        self._dump_training_stats('accuracy', accuracy)

        ########################
        # END OF YOUR CODE    #
        #######################

        return accuracy

    def _get_init_weight(self, shape, weight_scale):
        return np.random.normal(scale=weight_scale, size=shape)

    def _get_init_bias(self, shape, epsilon=0.):
        return np.zeros(shape=shape) + epsilon

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
        true_class_probs = np.sum(true_class_probs, axis=1) + 1e-6  # reduce
        loss = - np.log(true_class_probs).sum()

        return loss

    def _dump_training_stats(self, name, value):
        if self.training_mode:
            self.debug_stats[name] += [value]
        elif name in ['accuracy', 'nl_likelihood', 'nl_prior']:
            self.debug_stats['test_' + name] += [value]

    def __repr__(self):
        sep = '\n|' + '--' * 15 + '\n|\t'

        defs = ['X : input_dim x batch_size'] + [str(layer) for layer in self.layers] + [
            'Y : {} x batch_size'.format(self.n_classes)]

        return sep.join(['|\tNetwork Overview'] + defs) + '\n' + '--' * 15

    def plot_stats(self):
        """
        Generates plots
        :return:
        """
        # sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 2.5})
        # sns.set_style("whitegrid")

        plt.figure(figsize=(10, 10))
        plt.title('Delta Norms')
        for i in range(len(self.layers)):
            name = 'delta_{}_norm'.format(i)
            plt.plot(self.debug_stats[name], label=name)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./figs/mlp_delta_norms.pdf')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.title('Grad Norms')
        for i in range(len(self.layers)):
            name = 'dW_{}_norm'.format(i)
            plt.plot(self.debug_stats[name], label=name)

            name = 'db_{}_norm'.format(i)
            plt.plot(self.debug_stats[name], label=name)
        plt.legend()

        plt.tight_layout()
        plt.savefig('./figs/mlp_grad_norms.pdf')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.title('Logit norms')
        plt.plot(self.debug_stats['logits_norm'], label='logits_norm')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./figs/mlp_logit_norms.pdf')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.title('Train and test negative log likelihood')
        plt.plot(self.debug_stats['nl_likelihood'], label='Train NLL')
        num_eps = len(self.debug_stats['nl_prior'])
        num_reports = len(self.debug_stats['test_nl_prior'])
        x_interp = np.arange(num_reports) * num_eps / num_reports
        plt.plot(x_interp, self.debug_stats['test_nl_likelihood'], label='Test NLL')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./figs/mlp_losses_nll.pdf')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.title('Train and test negative log prior')
        plt.plot(self.debug_stats['nl_prior'], label='Train NLP')
        num_eps = len(self.debug_stats['nl_prior'])
        num_reports = len(self.debug_stats['test_nl_prior'])
        x_interp = np.arange(num_reports) * num_eps / num_reports
        plt.plot(x_interp, self.debug_stats['test_nl_prior'], label='Test NLP')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./figs/mlp_losses_nlp.pdf')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.title('Train and test accuracy')
        plt.plot(self.debug_stats['accuracy'], label='Train Accuracy')

        num_eps = len(self.debug_stats['accuracy'])
        num_reports = len(self.debug_stats['test_accuracy'])
        x_interp = np.arange(num_reports) * num_eps / num_reports

        plt.plot(x_interp, self.debug_stats['test_accuracy'], label='Test Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./figs/mlp_accuracy.pdf')
        plt.close()


class Layer(object):
    """
    A layer object that handles feed-forward and back propagation ops."""

    def __init__(self, W, b, activation, k, parent):
        self.W = W
        self.b = b
        self.activation_fn = activation
        self.k = k

        self.S_k = None
        self.Z_k = None
        self.Z_in = None
        self.parent = parent

    def forward(self, Z):
        assert Z.shape[0] == self.W.shape[1]

        self.Z_in = Z
        self.S_k = np.dot(self.W, Z) + self.b
        self.Z_k = self.activation_fn(self.S_k)

        return self.Z_k, self.S_k

    def backward(self, delta, flags):
        # compute gradients
        dL_dW = (delta.dot(self.Z_in.T) + flags['weight_decay'] * self.W)
        dL_db = delta.sum(axis=1, keepdims=True)

        # Gradient clipping
        # dL_dW = np.clip(dL_dW, -1., 1.)
        # dL_db = np.clip(dL_db, -1., 1.)

        # apply updates
        self.W -= flags['learning_rate'] * dL_dW
        self.b -= flags['learning_rate'] * dL_db

        # Debugging
        dL_dW_norm = np.linalg.norm(dL_dW)
        dL_db_norm = np.linalg.norm(dL_db)
        self.parent._dump_training_stats('dW_{}_norm'.format(self.k), dL_dW_norm)
        self.parent._dump_training_stats('db_{}_norm'.format(self.k), dL_db_norm)

        if dL_dW_norm > 100:
            print('dW exploding at layer {}'.format(self.k))
        if dL_db_norm > 100:
            print('db exploding at layer {}'.format(self.k))

    def activation_grad(self):
        """
        Computes the gradient of the relu activation

        dY/dXij =
                1 if Xij > 0
                0 else

        :param X: a
        :return:
        """
        return np.where(self.S_k > 0, 1, 0)

    def nlog_prior(self):
        """
        Log prior over each weight scalar N(0,1)
        :return: log( prod_i[ p(w_i) ]) = sum_i[ log N(w_i | 0,1)] = -0.5 * sum_i [w_i ** 2]
        """
        return 0.5 * (self.W ** 2).sum()

    # keras style printing
    def __repr__(self):
        return "W_{} : {} x {}, f: {}\n|\tZ_{} : {} x batch_size".format(self.k, *self.W.shape,
                                                                         self.activation_fn.__name__,
                                                                         self.k, self.W.shape[0])
