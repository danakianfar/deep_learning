#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from tensorflow.contrib.seq2seq import sequence_loss
import numpy as np


class TextGenerationModel(object):
    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers, decoding_model='greedy', embed_dim=40):

        self._train_seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size
        self.state_is_tuple = True
        self.decoding_mode = decoding_model
        self.embed_dim = embed_dim

        # Input, label placeholders
        # Word indices: [seq_length, batch_size]
        self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, self._batch_size), name='inputs')
        self.labels = tf.placeholder(dtype=tf.int32, shape=(self._batch_size, None), name='labels')
        self.decode_length = tf.placeholder(dtype=tf.int32, shape=(), name='decode_length')

        # Embeddings
        self._embeddings = tf.get_variable('embeddings', [self._vocab_size, self.embed_dim], dtype=tf.float32)

        # Encode to one-hot
        self._inputs_embed = tf.nn.embedding_lookup(self._embeddings, self.inputs)

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
        self.logit_fn = lambda x: tf.matmul(x, self._Wout) + self._bout
        self.logits = self._build_model()
        # self.decoded_sequence_train = self.predictions(self.logits)

        # Loss
        self.loss = self._compute_loss()

    def _init_lstm_cell(self):
        return BasicLSTMCell(num_units=self._lstm_num_hidden, state_is_tuple=self.state_is_tuple, activation=tf.nn.relu)

    def _build_model(self):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        # outputs: [timesteps, batch_size, num_hidden]
        outputs, _ = tf.nn.dynamic_rnn(cell=self._lstm_layers,
                                       initial_state=self._lstm_layers.zero_state(batch_size=self._batch_size,
                                                                                  dtype=tf.float32),
                                       inputs=self._inputs_embed,
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
                             weights=tf.ones(shape=(self._batch_size, self._train_seq_length)))

        tf.summary.scalar('cross entropy loss', loss)
        return loss

    def probabilities(self):  # not used
        # Returns the normalized per-step probabilities
        pass

    def _get_logits_per_step(self, outputs_per_step):
        """
        Returns logits over a sequences of hidden states
        :param outputs_per_step: network outputs. Float [time_steps, batch_size, lstm_hidden_num]
        :return: [time_steps, batch_size, vocab_size]
        """
        # outputs_flat = tf.reshape(outputs_per_step, (-1, self._lstm_num_hidden))  # flatten to do matmul
        # logits = self.logit_fn(outputs_flat)
        # logits_per_step = tf.reshape(logits, (seq_len, self._batch_size, self._vocab_size))
        logits_per_step = tf.tensordot(outputs_per_step, self._Wout, [[-1], [0]]) + self._bout
        return logits_per_step

    def decode_warmup(self, warmup_seq, decode_length=30):
        """
            Performs decoding (inference).
            :param warmup_seq: warm-up tokens. Int [seq_len, num_of_warmup_tokens]
            :param decode_length: int, number of steps to decode
            :return: output of decode fn
                """

        batch_size = warmup_seq.shape[1]
        outputs, init_state = tf.nn.dynamic_rnn(cell=self._lstm_layers,
                                                initial_state=self._lstm_layers.zero_state(batch_size=batch_size,
                                                                                           dtype=tf.float32),
                                                inputs=tf.nn.embedding_lookup(self._embeddings, warmup_seq),
                                                time_major=True)

        logits = self._get_logits_per_step(outputs_per_step=outputs)
        last_char = self.predictions(logits=logits[-1, :, :])
        return self.decode(batch_size, last_char, decode_length, init_state=init_state)

    def decode(self, decode_batch_size, init_input, decode_length=30, init_state=None):
        """
        A manual sampling function
        """

        if init_state is None:
            init_state = self._lstm_layers.zero_state(batch_size=decode_batch_size, dtype=tf.float32)
        else:
            init_state = init_state

        last_decode = init_input
        results = [init_input]
        for i in range(decode_length):
            embeds = tf.nn.embedding_lookup(self._embeddings, last_decode)
            preds, init_state = self._lstm_layers(inputs=embeds, state=init_state)
            last_decode = self.predictions(self.logit_fn(preds))
            results.append(last_decode)
        return tf.stack(results)

    def _greedy_decoding(self, logits):
        """
        Greedy decoding for the output at a single timestep.
        Greedy decoding takes the argmax of the logits.

        :param output: network outputs, float [batch_size, num_hidden]
        :return: decoded tokens. int [output_batch_size]
        """
        token_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return token_ids

    def _sample_decoding(self, logits):
        """
        Decoding by sampling from softmax probabilities for the output at a single timestep.

        :param output: network outputs, float [batch_size, num_hidden]
        :return: Decoded tokens. int [output_batch_size]
        """
        token_ids = tf.distributions.Categorical(logits=logits).sample()
        return token_ids

    def predictions(self, logits):
        if self.decoding_mode == 'greedy':
            return self._greedy_decoding(logits)
        else:
            return self._sample_decoding(logits)

    def decode_rawrnn(self):
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
            token_ids = tf.argmax(self.logit_fn(output), axis=-1, output_type=tf.int32)
            return token_ids

        def _sample_decoding(output):
            """
            Decoding by sampling from softmax probabilities for the output at a single timestep.

            :param output: network outputs, float [batch_size, num_hidden]
            :return: one-hot representation of decoded tokens. int [batch_size, vocab_size]
            """

            # output = tf.Print(output, [output.shape], 'output_shape in sampling = ')
            logits = self.logit_fn(output)
            # logits = tf.Print(logits, [logits.shape], 'logits_shape in sampling = ')
            token_ids = tf.distributions.Categorical(logits=logits).sample()
            # token_ids = tf.Print(token_ids, [token_ids.shape], 'token_ids_shape in sampling = ')
            return token_ids

        def loop_fn(time, previous_output, previous_state, loop_state):
            emit_output = previous_output

            if previous_output is None:  # only at time =0
                next_cell_state = self.initial_lstm_states
                next_input = self.random_initial_decoding_inputs
            else:
                next_cell_state = previous_state

                if self.decoding_mode == 'greedy':
                    next_input = _greedy_decoding(previous_output)
                elif self.decoding_mode == 'sampling':
                    next_input = _sample_decoding(previous_output)

            next_input = tf.nn.embedding_lookup(self._embeddings, next_input)
            next_loop_state = None  # ignore
            finished = time >= self.decode_length

            return finished, next_input, next_cell_state, emit_output, next_loop_state

        decoded_outputs_ta, _, _ = tf.nn.raw_rnn(cell=self._lstm_layers, loop_fn=loop_fn)
        decoded_outputs = decoded_outputs_ta.stack()  # [time_step, batch_size, num_hidden]
        decoded_logits = self._get_logits_per_step(decoded_outputs,
                                                   self.decode_length)  # [time_step, batch_size, vocab_size]
        decoded_tokens = tf.argmax(decoded_logits, axis=-1)  # [time_step, batch_size]

        return decoded_tokens
