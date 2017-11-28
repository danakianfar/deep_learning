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

import tensorflow as tf


class LSTM(object):
    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):
        self._input_length = input_length
        self._input_dim = input_dim
        self._num_hidden = num_hidden
        self._num_classes = num_classes
        self._batch_size = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases = tf.constant_initializer(0.0)

        # Dim of [h_{t-1}, x_t]
        self._gate_inputs_dim = self._input_dim + self._num_hidden

        # Input data [time, batch_size, input_dim]
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[self._input_length, self._batch_size, self._input_dim],
                                     name='inputs')

        # Targets [batch_size, output_dim]
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[self._batch_size, self._num_classes],
                                     name='labels')

        with tf.variable_scope('lstm_cell'):
            # Forget gate
            self._Wf = tf.get_variable(name='W_f', shape=(self._gate_inputs_dim, self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_weights)
            self._bf = tf.get_variable(name='b_f', shape=(self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_biases)

            # Input gate
            self._Wi = tf.get_variable(name='W_i', shape=(self._gate_inputs_dim, self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_weights)
            self._bi = tf.get_variable(name='b_i', shape=(self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_biases)

            self._Wg = tf.get_variable(name='W_g', shape=(self._gate_inputs_dim, self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_weights)
            self._bg = tf.get_variable(name='b_g', shape=(self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_biases)

            # Output gate
            self._Wo = tf.get_variable(name='W_o', shape=(self._gate_inputs_dim, self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_weights)
            self._bo = tf.get_variable(name='b_o', shape=(self._num_hidden), dtype=tf.float32,
                                       initializer=initializer_biases)

            # inputs (h_{t-1}, x_t): [batch_size, self.input_dim + self.num_hidden)

            # Use less matmul ops as specified by Zaremba et. al 2014: https://arxiv.org/pdf/1409.2329.pdf
            # Order: input gate (sigmoid), new candidates (tanh), forget gate (sigmoid), output gate (sigmoid)
            # dim: [input_dim + num_hidden, 4 * num_hidden]
            self._weights = tf.concat([self._Wi, self._Wg, self._Wf, self._Wo], axis=1)

            # dim: [4 * num_hidden]
            self._biases = tf.concat([self._bi, self._bg, self._bf, self._bo], axis=0)

        # Logits
        with tf.variable_scope('logits'):
            self._Wout = tf.get_variable(name='W_out', shape=(self._num_hidden, self._num_classes), dtype=tf.float32,
                                         initializer=initializer_weights)
            self._bout = tf.get_variable(name='b_out', shape=(self._num_classes), dtype=tf.float32,
                                         initializer=initializer_biases)

        self.logits_op = self.compute_logits()
        self.loss_op = self.compute_loss()
        self.accuracy_op = self.accuracy()
        # self.confusion_matrix_op = self.confusion_matrix()

    def _lstm_step(self, lstm_state_tuple, x_t):
        """
        Performs a single LSTM step
        Use this function with a tf.scan to unroll the network and perform inference over a sequence of inputs
        Follows the convention of Zaremba et. al 2014: https://arxiv.org/pdf/1409.2329.pdf

        :param lstm_state_tuple: previous LSTM state tuple (c_{t-1}, h_{t-1})
        :param x_t: input for current step from previous (input) layer. [batch_size, input_dim]
        :return: LSTM state tuple for current step. (c_{t-1}, h_{t-1})
        """
        # unstack LSTM state (c, h) from prev time step
        c_prev, h_prev = tf.unstack(lstm_state_tuple, axis=0)

        # forward pass
        _inpt = tf.concat([h_prev, x_t], axis=1)

        # preactivations: input gate, new candidates, forget gate, output gate
        _gates = tf.matmul(_inpt, self._weights) + self._biases
        i, g, f, o = tf.split(value=_gates, num_or_size_splits=4, axis=1)

        # Update cell state and hidden state
        next_c = tf.sigmoid(i) * tf.tanh(g) + tf.sigmoid(f) * c_prev
        next_h = tf.tanh(next_c) * tf.sigmoid(o)

        next_state = tf.stack((next_c, next_h), axis=0)

        return next_state

    @staticmethod
    def _zero_state(hidden_dim, batch_size, dtype=tf.float32):
        """
        Returns an empty (zero) state for the hidden state of the RNN
        :param hidden_dim: number of hidden units, int
        :param batch_size: batch_size, int
        :param dtype: data type, float32 by default
        :return: a zero vector [batch_size, hidden_dim]
        """
        return tf.stack(values=(tf.zeros(shape=(batch_size, hidden_dim), dtype=dtype),
                                tf.zeros(shape=(batch_size, hidden_dim), dtype=dtype)), axis=0)

    def _get_hidden_states(self):
        """
        Unrolls the RNN and computes hidden states for each timestep in self.inputs placeholder
        :return: hidden states for each time step. Float [time, batch_size, hidden_dim]
        """
        return tf.scan(fn=lambda lstm_state_tuple, x: self._lstm_step(lstm_state_tuple=lstm_state_tuple, x_t=x),
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
        last_hidden_state = hidden_states[-1]

        c, h = tf.unstack(last_hidden_state, axis=0)

        # h{T} => p{T}
        logits = tf.add(tf.matmul(h, self._Wout), self._bout, name='logits')
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

        # tf.summary.image('confusion_matrix', tf.reshape(tf.cast(confusion_matrix, dtype=tf.float32), [1, self._num_classes, self._num_classes, 1]))

        return confusion_matrix
