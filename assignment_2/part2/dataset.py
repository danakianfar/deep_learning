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

import os
import re
import math
import numpy as np
import tensorflow as tf


class TextDataset(object):
    def __init__(self, filename, clean_data=False):
        assert os.path.splitext(filename)[1] == ".txt"
        self._data = open(filename, 'r', encoding='utf8').read()
        if clean_data:
            self._data = self.clean_str(self._data)
        self._chars = list(set(self._data))
        self._chars.sort()
        self._data_size, self._vocab_size = len(self._data), len(self._chars)
        print("Initialize dataset with {} characters, {} unique.".format(
            self._data_size, self._vocab_size))
        self._char_to_ix = {ch: i for i, ch in enumerate(self._chars)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self._chars)}
        self._offset = 0

    def example(self, seq_length):
        offset = np.random.randint(0, len(self._data) - seq_length - 2)
        inputs = [self._char_to_ix[ch] for ch in self._data[offset:offset + seq_length]]
        targets = [self._char_to_ix[ch] for ch in self._data[offset + 1:offset + seq_length + 1]]
        return inputs, targets

    def batch(self, batch_size, seq_length):
        batch_inputs = np.zeros((batch_size, seq_length), np.int32)
        batch_targets = np.zeros((batch_size, seq_length), np.int32)
        for i in range(batch_size):
            batch_inputs[i], batch_targets[i] = self.example(seq_length)
        return batch_inputs, batch_targets

    def convert_to_string(self, char_ix):
        return ''.join(self._ix_to_char[ix] for ix in char_ix)

    @property
    def vocab_size(self):
        return self._vocab_size

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"\\n", " ", string)
        return string.strip().lower()

