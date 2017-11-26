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
import math
import numpy as np
import tensorflow as tf


def generate_palindrome(length):
    # Generates a single, random palindrome number of 'length' digits.
    left = [np.random.randint(0, 10) for _ in range(math.ceil(length/2))]
    left = np.asarray(left, dtype=np.int32)
    right = np.flip(left, 0) if length % 2 == 0 else np.flip(left[:-1], 0)
    return np.concatenate((left, right))

def generate_palindrome_batch(batch_size, length):
    # Generates a batch of random palindrome numbers.
    batch = [generate_palindrome(length) for _ in range(batch_size)]
    return np.asarray(batch, np.int32)

def init_summary_writer(sess, save_path):
    # Optional to use.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return tf.summary.FileWriter(save_path, sess.graph)

