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
from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell
from tensorflow.contrib.seq2seq import sequence_loss, GreedyEmbeddingHelper
from tensorflow.contrib.layers import fully_connected
import numpy as np


class TextGenerationModel(object):
    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers, reuse_hidden_state_prob=1.0):
        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size
        self._reuse_hidden_state_prob = reuse_hidden_state_prob
        self.state_is_tuple = False  # slower but less buggy

        # Input, label placeholders
        # Word indices: [seq_length, batch_size]
        self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, self._batch_size), name='inputs')
        self.labels = tf.placeholder(dtype=tf.int32, shape=(self._batch_size, None), name='labels')
        self.random_initial_decoding_inputs = tf.placeholder(dtype=tf.int32, shape=(self._batch_size),
                                                             name='random_initial_decoding_inputs')
        self.decode_length = tf.placeholder(dtype=tf.int32, shape=(), name='decode_length')
        self.initial_lstm_states = tf.placeholder(dtype=tf.float32, name='initial_hidden_state',
                                                  shape=(None,
                                                         self._lstm_num_layers * 2 * self._lstm_num_hidden))

        # Encode to one-hot
        self._onehot_inputs = tf.one_hot(self.inputs, depth=self._vocab_size, name='onehot_inputs')

        # Create networks
        self._lstm_layers = MultiRNNCell([self._init_lstm_cell() for _ in range(self._lstm_num_layers)],
                                         state_is_tuple=self.state_is_tuple)
        with tf.variable_scope('logits'):
            self._Wout = tf.get_variable(name='W_out', shape=(self._lstm_num_hidden, self._vocab_size),
                                         dtype=tf.float32,
                                         initializer=tf.variance_scaling_initializer())
            self._bout = tf.get_variable(name='b_out', shape=(self._vocab_size), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.))

        # Decode params
        self._random_initial_decoding_inputs_onehot = tf.one_hot(self.random_initial_decoding_inputs,
                                                                 depth=self._vocab_size,
                                                                 name='random_initial_decoding_inputs_onehot')
        self.logit_fn = lambda x: tf.matmul(x, self._Wout) + self._bout
        self.logits = self._build_model()
        self.decoded_sequence = self.inference()

        # Loss
        self.loss = self._compute_loss()

    def _init_lstm_cell(self):
        return BasicLSTMCell(num_units=self._lstm_num_hidden, state_is_tuple=self.state_is_tuple)

    def zero_state_numpy(self):
        return tuple(np.zeros(shape=(self._batch_size, 2 * self._lstm_num_hidden)) for _ in
                     range(self._lstm_num_layers)) if self.state_is_tuple else np.zeros(
            shape=(self._batch_size, self._lstm_num_layers * 2 * self._lstm_num_hidden))

    def _build_model(self):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # outputs: [timesteps, batch_size, num_hidden]
        outputs, _ = tf.nn.dynamic_rnn(cell=self._lstm_layers,
                                       initial_state=self.initial_lstm_states,
                                       inputs=self._onehot_inputs,
                                       time_major=True)

        # logits: [timesteps, batch_size, vocab size]
        logits_per_step = self._get_logits_per_step(outputs)

        return logits_per_step

    def _compute_loss(self):
        # Cross-entropy loss, averaged over timestep and batch
        loss = sequence_loss(logits=tf.transpose(self.logits, perm=(1, 0, 2)),
                             targets=self.labels,
                             average_across_timesteps=True,
                             average_across_batch=True,
                             weights=tf.ones(shape=(self._batch_size, self._seq_length)))

        tf.summary.scalar('cross entropy loss', loss)
        return loss

    def probabilities(self):
        # Returns the normalized per-step probabilities
        probabilities = tf.nn.softmax(self.logits, dim=-1)
        return probabilities

    def predictions(self):
        # Returns the per-step predictions
        predictions = tf.argmax(self.logits)
        return predictions

    def _get_logits_per_step(self, outputs_per_step):
        """
        Returns logits over a sequences of hidden states
        :param outputs_per_step: network outputs. Float [time_steps, batch_size, lstm_hidden_num]
        :return: [time_steps, batch_size, vocab_size]
        """
        outputs_flat = tf.reshape(outputs_per_step, (-1, self._lstm_num_hidden))  # flatten to do matmul
        logits = self.logit_fn(outputs_flat)
        logits_per_step = tf.reshape(logits, (self._seq_length, self._batch_size, self._vocab_size))
        return logits_per_step

    def inference(self):
        """
        Performs decoding (inference) using the LSTM cell.
        The decoding length, initial state and initial inputs to the network are passed as placeholders.
        :return: Decoded sequence [time_step, batch_size]
        """

        def _greedy_decoding(output):
            """
            Greedy decoding for the output at a single timestep.
            Greedy decoding takes the argmax of the logits.

            :param output: network outputs, float [batch_size, num_hidden]
            :return: one-hot representation of decoded tokens. int [batch_size, vocab_size]
            """
            token_ids = tf.argmax(self.logit_fn(output), axis=-1)
            return tf.one_hot(token_ids, depth=self._vocab_size)

        def loop_fn(time, previous_output, previous_state, loop_state):
            emit_output = previous_output

            if previous_output is None:
                next_cell_state = self.initial_lstm_states
                next_input = self._random_initial_decoding_inputs_onehot
            else:
                next_cell_state = previous_state
                next_input = _greedy_decoding(previous_output)  # TODO replace by sampling

            next_loop_state = None  # ignore
            finished = time >= self.decode_length

            return finished, next_input, next_cell_state, emit_output, next_loop_state

        decoded_outputs_ta, _, _ = tf.nn.raw_rnn(cell=self._lstm_layers, loop_fn=loop_fn, parallel_iterations=10)
        decoded_outputs = decoded_outputs_ta.stack()  # [time_step, batch_size, num_hidden]
        decoded_logits = self._get_logits_per_step(decoded_outputs)  # [time_step, batch_size, vocab_size]
        decoded_tokens = tf.argmax(decoded_logits, axis=-1)  # [time_step, batch_size]

        return decoded_tokens
