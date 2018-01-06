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
        self.training_mode = tf.placeholder(tf.bool, name='training_mode')
        self.batch_norm = tf.placeholder(tf.bool, name='batch_norm')
        self.batch_norm_bool = False  # batch norm makes the graph too large
        self.dropout_rate = 0

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

        with tf.variable_scope('layer1') as scope:
            conv1 = tf.layers.conv2d(inputs=x,
                                     filters=64,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='same',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=1e-4),
                                     bias_initializer=tf.constant_initializer(1e-5),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     name='{}_conv'.format(scope.name))

            if self.batch_norm_bool:  # use this flag to avoid huge graphs
                conv1 = tf.cond(self.batch_norm,
                                lambda: tf.contrib.layers.batch_norm(conv1,
                                                                     center=True,
                                                                     scale=True,
                                                                     is_training=self.training_mode,
                                                                     activation_fn=None,
                                                                     scope=scope),
                                lambda: conv1)

            conv1 = tf.nn.relu(conv1, name='{}_relu'.format(scope.name))

            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=(3, 3),
                                            strides=2,
                                            padding='valid',
                                            data_format='channels_last',
                                            name='{}_maxpool'.format(scope.name))

        with tf.variable_scope('layer2') as scope:
            conv2 = tf.layers.conv2d(inputs=pool1,
                                     filters=64,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='same',
                                     data_format='channels_last',
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.random_normal_initializer(stddev=1e-4),
                                     bias_initializer=tf.constant_initializer(1e-5),
                                     kernel_regularizer=None,
                                     bias_regularizer=None,
                                     name='{}_conv'.format(scope.name))

            if self.batch_norm_bool:
                conv2 = tf.cond(self.batch_norm,
                                lambda: tf.contrib.layers.batch_norm(conv2,
                                                                     center=True,
                                                                     scale=True,
                                                                     is_training=self.training_mode,
                                                                     activation_fn=None,
                                                                     scope=scope),
                                lambda: conv2)

            conv2 = tf.nn.relu(conv2, name='{}_relu'.format(scope.name))

            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                            pool_size=(3, 3),
                                            strides=2,
                                            padding='valid',
                                            data_format='channels_last',
                                            name='{}_maxpool'.format(scope.name))

        with tf.name_scope('flatten') as scope:
            flattened = tf.contrib.layers.flatten(pool2, scope=scope)

        with tf.variable_scope('fc1') as scope:
            fc1 = tf.layers.dense(inputs=flattened,
                                  units=384,
                                  activation=None,
                                  use_bias=True,
                                  bias_initializer=tf.constant_initializer(1e-5),
                                  trainable=True,
                                  name=scope.name)

            if self.batch_norm_bool:
                fc1 = tf.cond(self.batch_norm,
                              lambda: tf.contrib.layers.batch_norm(fc1,
                                                                   center=True,
                                                                   scale=True,
                                                                   is_training=self.training_mode,
                                                                   activation_fn=None,
                                                                   scope=scope),
                              lambda: fc1)

            fc1 = tf.nn.relu(fc1, name='{}_relu'.format(scope.name))

            fc1 = tf.cond(self.training_mode,
                          lambda: tf.nn.dropout(fc1, keep_prob=1. - self.dropout_rate, name='dropout_activations'),
                          lambda: fc1)

        with tf.variable_scope('fc2') as scope:
            fc2 = tf.layers.dense(inputs=fc1,
                                  units=192,
                                  activation=None,
                                  use_bias=True,
                                  bias_initializer=tf.constant_initializer(1e-5),
                                  name=scope.name)

            if self.batch_norm_bool:
                fc2 = tf.cond(self.batch_norm,
                              lambda: tf.contrib.layers.batch_norm(fc2,
                                                                   center=True,
                                                                   scale=True,
                                                                   is_training=self.training_mode,
                                                                   activation_fn=None,
                                                                   scope=scope),
                              lambda: fc2)

            fc2 = tf.nn.relu(fc2, name='{}_relu'.format(scope.name))
            fc2 = tf.cond(self.training_mode,
                          lambda: tf.nn.dropout(fc2, keep_prob=1. - self.dropout_rate, name='dropout_activations'),
                          lambda: fc2)

        with tf.variable_scope('fc3') as scope:
            fc3 = tf.layers.dense(inputs=fc2,
                                  units=10,
                                  activation=None,  # linear
                                  use_bias=True,
                                  bias_initializer=tf.constant_initializer(1e-5),
                                  trainable=True,
                                  name=scope.name)

        logits = fc3

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
