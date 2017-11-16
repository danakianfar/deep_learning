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

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size

        # Initialization:
        # ...

    def _build_model(self):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]
        logits_per_step = None
        return logits_per_step

    def _compute_loss(self):
        # Cross-entropy loss, averaged over timestep and batch
        loss = None
        return loss

    def probabilities(self):
        # Returns the normalized per-step probabilities
        probabilities = None
        return probabilities

    def predictions(self):
        # Returns the per-step predictions
        predictions = None
        return predictions