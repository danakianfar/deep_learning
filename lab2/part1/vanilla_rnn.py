# MIT License
# 
# Copyright (c) 2017 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


################################################################################

class VanillaRNN(object):
    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):
        self._input_length = input_length
        self._input_dim = input_dim
        self._num_hidden = num_hidden
        self._num_classes = num_classes  # data is already in one-hot encoding
        self._batch_size = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases = tf.constant_initializer(0.0)

        # Input data [time, batch_size, input_dim]
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self._input_length, self._batch_size, self._input_dim],
                                     name='inputs')

        # Targets [batch_size, output_dim]
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[self._batch_size, self._num_classes],
                                     name='labels')

        # RNN cell
        with tf.variable_scope('rnn_cell'):
            # x{t} => h{t}
            self._Wxh = tf.get_variable(name='W_xh', shape=(self._input_dim, self._num_hidden), dtype=tf.float32,
                                        initializer=initializer_weights)
            # h{t-1} => h{t}
            self._Whh = tf.get_variable(name='W_hh', shape=(self._num_hidden, self._num_hidden), dtype=tf.float32,
                                        initializer=initializer_weights)
            self._bh = tf.get_variable(name='b_h', shape=(self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_biases)

            # h{t} => y{t}
            self._Woh = tf.get_variable(name='W_oh', shape=(self._num_hidden, self._num_classes), dtype=tf.float32,
                                        initializer=initializer_weights)
            self._bo = tf.get_variable(name='b_o', shape=(self._num_classes), dtype=tf.float32,
                                       initializer=initializer_biases)

            # tf.summary.histogram('W_xh', self._Wxh)
            # tf.summary.histogram('W_hh', self._Whh)
            # tf.summary.histogram('W_oh', self._Woh)
            # tf.summary.histogram('b_h', self._bh)
            # tf.summary.histogram('b_o', self._bo)

        self.logits_op = self.compute_logits()
        self.loss_op = self.compute_loss()
        self.accuracy_op = self.accuracy()
        self.confusion_matrix_op = self.confusion_matrix()

    def _rnn_step(self, h_prev, x):
        """
        Performs a single RNN step with a hyperbolic tangent non-linearity.
        Use this function with tf.scan to unroll the RNN and perform inference over a sequence of inputs.

        :param h_prev: hidden state from previous step. Float [batch_size, hidden_dim]
        :param x: input for current step from previous (input) layer. Float [batch_size, input_dim]
        :return: hidden state for current step. Float [batch_size, hidden_dim]
        """
        # x{t} => h{t}
        input_projection = tf.matmul(x, self._Wxh)

        # h{t-1} => h{t}
        hidden_recurrence = tf.matmul(h_prev, self._Whh)

        # h{t}
        h_current = tf.tanh(input_projection + hidden_recurrence + self._bh)

        return h_current

    @staticmethod
    def _zero_state(hidden_dim, batch_size, dtype=tf.float32):
        """
        Returns an empty (zero) state for the hidden state of the RNN
        :param hidden_dim: number of hidden units, int
        :param batch_size: batch_size, int
        :param dtype: data type, float32 by default
        :return: a zero vector [batch_size, hidden_dim]
        """
        return tf.zeros(shape=(batch_size, hidden_dim), dtype=dtype)

    def _get_hidden_states(self):
        """
        Unrolls the RNN and computes hidden states for each timestep in self.inputs placeholder
        :return: hidden states for each time step. Float [time, batch_size, hidden_dim]
        """
        return tf.scan(fn=lambda h_prev, x: self._rnn_step(h_prev=h_prev, x=x),
                       elems=self.inputs,
                       initializer=self._zero_state(hidden_dim=self._num_hidden,
                                                    batch_size=self._batch_size,
                                                    dtype=tf.float32),
                       parallel_iterations=10,
                       name='hidden_states')

    def compute_logits(self):
        """
        Forward propagates inputs, computes hidden states and then computes the outputs (logits) from the last hidden state
        :return: logits. Float [batch_size, output_dim]
        """

        # [time, batch_size, hidden_dim]
        hidden_states = self._get_hidden_states()
        last_hidden_state = hidden_states[-1, :, :]

        # h{T} => p{T}
        logits = tf.add(tf.matmul(last_hidden_state, self._Woh), self._bo, name='logits')
        # tf.summary.histogram('logits', logits)

        return logits

    def compute_loss(self):
        """
        Computes the cross-entropy loss using the internal variable _logits
        :return: loss, scalar float
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits_op,
            name='softmax_cross_entropy_loss'
        )
        loss = tf.reduce_mean(loss, name='mean_cross_entropy_loss')

        tf.summary.scalar('mean cross-entropy loss', loss)

        return loss

    def accuracy(self):
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
        # Implement the accuracy of predicting the
        # last digit over the current batch ...

        predictions = tf.argmax(input=self.logits_op, axis=1, name='label_predictions')
        class_labels = tf.argmax(input=self.labels, axis=1)

        accuracy = tf.to_float(tf.equal(predictions, class_labels))
        accuracy = tf.reduce_mean(accuracy, name='accuracy')

        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.histogram('label predictions', predictions)

        return accuracy

    def confusion_matrix(self):
        predictions = tf.argmax(input=self.logits_op, axis=1)
        class_labels = tf.argmax(input=self.labels, axis=1)

        confusion_matrix = tf.contrib.metrics.confusion_matrix(
            labels=class_labels,
            predictions=predictions,
            num_classes=10,
            dtype=tf.int32,
            name='confusion_matrix')

        tf.summary.image('confusion_matrix', tf.reshape(tf.cast(confusion_matrix, dtype=tf.float32),
                                                        [1, self._num_classes, self._num_classes, 1]))

        return confusion_matrix
