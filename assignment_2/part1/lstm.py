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
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        # ...

    def _lstm_step(self, lstm_state_tuple, x):
        # Single step through LSTM cell ...
        raise NotImplementedError()

    def compute_logits(self):
        # Implement the logits for predicting the last digit in the palindrome
        logits = None
        return logits

    def compute_loss(self):
        # Implement the cross-entropy loss for classification of the last digit
        loss = None
        return loss

    def accuracy(self):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        accuracy = None
        return accuracy