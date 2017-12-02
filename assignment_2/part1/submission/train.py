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
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from vanilla_rnn import VanillaRNN
from lstm import LSTM


################################################################################

def _ensure_path_exists(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


def _gradient_summary(variable, gradient, tag):
    tf.summary.histogram('{}_{}'.format(variable.op.name, tag), gradient)


def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.
    Args:
      labels_dense: Dense labels.
      num_classes: Number of classes.

    Outputs:
      labels_one_hot: One-hot encoding for labels.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def train(config):

    # Reproducibility
    tf.set_random_seed(42)
    np.random.seed(42)

    assert config.model_type in ('RNN', 'LSTM')
    tf.reset_default_graph()

    # Setup the model that we are going to use
    if config.model_type == 'RNN':
        print("Initializing Vanilla RNN model...")
        model = VanillaRNN(
            config.input_length - 1, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size)
    else:
        print("Initializing LSTM model...")
        model = LSTM(
            config.input_length - 1, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size)

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Utility vars and ops
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # logging
    train_logdir = os.path.join(config.summary_path, '{}_train'.format(config.model_name))
    train_log_writer = utils.init_summary_writer(session, train_logdir)

    # Define the optimizer
    if config.optimizer.lower() == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
    elif config.optimizer.lower() == 'adam':
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
    ###########################################################################
    # QUESTION: what happens here and why?
    # Answer: Instead of calling optimizer.minimize(..) as usual, we compute the gradients,
    # and then clip each gradient value if they fall outside of a desirable range.
    # This avoid applying gradient updates that we either too large or too small, due
    # to the exploding/vanishing gradients problem.
    ###########################################################################
    grads_and_vars = optimizer.compute_gradients(model.loss_op)
    # [_gradient_summary(var, grad, 'raw_grad') for var, grad in grads_and_vars]

    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)

    grads_and_vars = list(zip(grads_clipped, variables))
    # [_gradient_summary(var, grad, 'clipped_grad') for var, grad in grads_and_vars]
    apply_gradients_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    ############################################################################

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Initialize variables
    summary_op = tf.summary.merge_all()
    session.run(fetches=[tf.global_variables_initializer(), tf.local_variables_initializer()])
    for train_step in range(config.train_steps):

        # Get data and convert to one-hot
        data = utils.generate_palindrome_batch(batch_size=config.batch_size, length=config.input_length)
        inputs, labels = data[:, :-1], data[:, -1]
        inputs = (np.arange(config.num_classes) == inputs[..., None]).astype(int)
        labels = (np.arange(config.num_classes) == labels[..., None]).astype(int)
        inputs = np.transpose(inputs, axes=(1, 0, 2))  # [time, batch_size, input_dim)

        # Only for time measurement of step through network
        t1 = time.time()

        train_feed = {model.inputs: inputs, model.labels: labels}

        fetches = [model.loss_op, model.accuracy_op, apply_gradients_op]
        if train_step % config.print_every == 0:
            fetches += [summary_op]
            loss, accuracy, _, summary = session.run(fetches=fetches, feed_dict=train_feed)
            train_log_writer.add_summary(summary, train_step)
        else:
            loss, accuracy, _ = session.run(fetches=fetches, feed_dict=train_feed)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        # Print the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Examples/Sec = {:.2f}, Accuracy = {:.2f}%, Loss = {:.4f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                config.train_steps, config.batch_size, examples_per_second, accuracy * 100, loss
            ))

    train_log_writer.close()


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2500, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=10.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--model_name', type=str, default='vanilla_rnn', help='Model name for saving')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'rmsprop'], default="RMSProp",
                        help='Optimizer, choose between adam and rmsprop')

    parser.add_argument('--grid_search', action='store_true', help='Performs a grid search over parameters')
    config, _ = parser.parse_known_args()

    if config.grid_search:

        for model_type in ['LSTM']:
            for input_length in [200,300]:
                for learning_rate in [1e-1]:
                    for optimizer in ['adam']:
                        model_name = '{}_({}_{})_T{}'.format(model_type, optimizer, learning_rate, input_length)
                        config.model_type = model_type
                        config.learning_rate = learning_rate
                        config.input_length = input_length
                        config.model_type = model_type
                        config.model_name = model_name

                        print('Grid Search \n {}'.format(str(config)))
                        train(config)

    # Train the model
    else:
        train(config)
