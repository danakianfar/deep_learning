"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ConvNet(object):
    """
    This class implements a convolutional neural network in TensorFlow.
    It incorporates a certain graph model to be trained and to be used
    in inference.
    """

    def __init__(self, n_classes=10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """

        ########################
        # PUT YOUR CODE HERE  #
        ########################

        with tf.variable_scope('conv1') as scope:
            conv1 = tf.layers.conv2d(inputs=x,
                                     filters=3,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='same',
                                     data_format='channels_last',
                                     dilation_rate=(1, 1),
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=1e-4),
                                     bias_initializer=tf.constant_initializer(1e-5),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     name='{}_conv'.format(scope.name))

            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=(3, 3),
                                            strides=1,
                                            padding='valid',
                                            data_format='channels_last',
                                            name='{}_maxpool'.format(scope.name))

        ########################
        # END OF YOUR CODE    #
        ########################
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
        Calculates the multiclass cross-entropy loss from the logits predictions and
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

        tf.summary.scalar('cross entropy loss', loss)

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

        optimizer = flags['optimizer']
        global_step = flags['global_step']

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
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
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

        accuracy = tf.metrics.accuracy(predictions=predictions, labels=class_labels)
        accuracy = tf.reduce_mean(accuracy, name='accuracy')

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('label predictions', predictions)

        ########################
        # END OF YOUR CODE    #
        #######################
        return accuracy
