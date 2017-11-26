"""
This module implements a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in Tensorflow.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform inference, training and it
    can also be used for evaluating prediction performance.
    """

    def __init__(self,
                 n_hidden,
                 n_classes,
                 is_training,
                 activation_fn=tf.nn.relu,
                 dropout_rate=0.,
                 weight_initializer=xavier_initializer(),
                 weight_regularizer=l2_regularizer(0.001)):
        """
        Constructor for an MLP object. Default values should be used as hints for
        the usage of each parameter.

        Args:
          n_hidden: list of ints, specifies the number of units
                         in each hidden layer. If the list is empty, the MLP
                         will not have any hidden units, and the model
                         will simply perform a multinomial logistic regression.
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the MLP.
          is_training: Bool Tensor, it indicates whether the model is in training
                            mode or not. This will be relevant for methods that perform
                            differently during training and testing (such as dropout).
                            Have look at how to use conditionals in TensorFlow with
                            tf.cond.
          activation_fn: callable, takes a Tensor and returns a transformed tensor.
                              Activation function specifies which type of non-linearity
                              to use in every hidden layer.
          dropout_rate: float in range [0,1], presents the fraction of hidden units
                             that are randomly dropped for regularization.
          weight_initializer: callable, a weight initializer that generates tensors
                                   of a chosen distribution.
          weight_regularizer: callable, returns a scalar regularization loss given
                                   a weight variable. The returned loss will be added to
                                   the total loss for training purposes.
        """
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        # self.is_training = is_training
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.input_dim = 3 * 32 * 32

        # Bias initialization
        self.eps = 1e-5
        self.bias_initializer = tf.constant_initializer(value=self.eps, dtype=tf.float32)

        self.training_mode = tf.placeholder(bool, name='training_mode')

    def _construct_summary(self):
        pass

    # Convention: Y = XW + b
    def _dense_layer(self, inputs, scope_name, W_shape):
        input_dim, output_dim = W_shape

        with tf.variable_scope(scope_name):
            W = tf.get_variable(name='weights', shape=W_shape, dtype=tf.float32,
                                initializer=self.weight_initializer, regularizer=self.weight_regularizer)

            b = tf.get_variable(name='bias', dtype=tf.float32, shape=[output_dim],
                                initializer=self.bias_initializer)

            S = tf.add(tf.matmul(inputs, W), b, name='preactivation')
            Z = self.activation_fn(S, name='activations')

            outputs = tf.cond(self.training_mode,
                              lambda: tf.nn.dropout(Z, keep_prob=1. - self.dropout_rate, name='dropout_activations'),
                              lambda: Z)

            tf.summary.histogram('weights_{}'.format(scope_name), W)
            tf.summary.histogram('biases_{}'.format(scope_name), b)
            tf.summary.histogram('preactivation_{}'.format(scope_name), S)
            tf.summary.histogram('activations_{}'.format(scope_name), Z)
            tf.summary.histogram('dropout_activations_{}'.format(scope_name), outputs)

        return outputs

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network. Here an input tensor is transformed through application
        of several hidden layer transformations (as defined in the constructor).
        We recommend you to iterate through the list self.n_hidden in order to
        perform the sequential transformations in the MLP. Do not forget to
        add a linear output layer (without non-linearity) as the last transformation.

        In order to keep things uncluttered we recommend you (though it's not required)
        to implement a separate function that is used to define a fully connected
        layer of the MLP.

        In order to make your code more structured you can use variable scopes and name
        scopes. You can define a name scope for the whole model, for each hidden
        layer and for output. Variable scopes are an essential component in TensorFlow
        design for parameter sharing.

        You can use tf.summary.histogram to save summaries of the fully connected layer weights,
        biases, pre-activations, post-activations, and dropped-out activations
        for each layer. It is very useful for introspection of the network using TensorBoard.

        Args:
          x: 2D float Tensor of size [batch_size, input_dimensions]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                 the logits outputs (before softmax transformation) of the
                 network. These logits can then be used with loss and accuracy
                 to evaluate the model.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # Shapes of layers
        W_shapes = [self.input_dim] + self.n_hidden + [self.n_classes]
        W_shapes = [(W_shapes[i], W_shapes[i + 1]) for i in range(len(W_shapes) - 1)]

        Z = x
        for layer_num, shape in enumerate(W_shapes):
            layer_name = 'dense_{}'.format(layer_num)
            Z = self._dense_layer(inputs=Z, W_shape=shape, scope_name=layer_name)

        logits = Z

        ########################
        # END OF YOUR CODE    #
        #######################

        return logits

    def _complexity_cost(self):

        sum_reg_cost = None
        regularization_costs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_costs:
            sum_reg_cost = tf.add_n(regularization_costs, name='sum_regularization_loss')
            tf.summary.scalar('sum regularization loss', sum_reg_cost)
        return sum_reg_cost

    def loss(self, logits, labels):
        """
        Computes the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.

        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.

        You can use tf.summary.scalar to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       sample in the batch.
        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name='softmax_cross_entropy_loss'
        )
        loss = tf.reduce_mean(loss, name='mean_softmax_cross_entropy_loss')

        tf.summary.scalar('mean cross entropy loss', loss)

        complexity_cost = self._complexity_cost()
        if complexity_cost is not None:
            loss = tf.add(loss, complexity_cost, name='total_loss')
            tf.summary.scalar('total loss', loss)

        ########################
        # END OF YOUR CODE    #
        #######################

        return loss

    def _gradient_summary(self, variable, gradient, tag):
        tf.summary.histogram('{}_{}'.format(variable.op.name, tag), gradient)

    def train_step(self, loss, flags):
        """
        Implements a training step using a parameters in flags.

        Args:
          loss: scalar float Tensor.
          flags: contains necessary parameters for optimization.
        Returns:
          train_step: TensorFlow operation to perform one training step
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # Define global step counter

        optimizer = flags['optimizer']
        global_step = flags['global_step']

        # For batch-norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Gradient clipping
            grads = optimizer.compute_gradients(loss)
            [self._gradient_summary(var, grad, 'grad') for var, grad in grads]

            if flags['grad_clipping']:
                grads = [(tf.clip_by_value(grad, -1., 1.), tvar) for grad, tvar in
                         grads if grad is not None]

                [self._gradient_summary(var, grad, 'clipped_grad') for var, grad in grads]

            train_step = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)
        ########################
        # END OF YOUR CODE    #
        #######################

        return train_step

    def accuracy(self, logits, labels):
        """
        Computes the prediction accuracy, i.e. the average of correct predictions
        of the network.

        As in self.loss above, you can use tf.summary.scalar to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each sample in the batch.
        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        predictions = tf.argmax(input=logits, axis=1, name='label_predictions')
        class_labels = tf.argmax(input=labels, axis=1)

        accuracy = tf.to_float(tf.equal(predictions, class_labels))
        accuracy = tf.reduce_mean(accuracy, name='accuracy')

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('label predictions', predictions)

        ########################
        # END OF YOUR CODE    #
        #######################

        return accuracy

    def confusion_matrix(self, logits, labels):

        predictions = tf.argmax(input=logits, axis=1)
        class_labels = tf.argmax(input=labels, axis=1)

        confusion_matrix = tf.contrib.metrics.confusion_matrix(
            labels=class_labels,
            predictions=predictions,
            num_classes=10,
            dtype=tf.int32,
            name='confusion_matrix')

        tf.summary.image('confusion_matrix', tf.reshape(tf.cast(confusion_matrix, dtype=tf.float32),
                                                        [1, self.n_classes, self.n_classes, 1]))

        return confusion_matrix
