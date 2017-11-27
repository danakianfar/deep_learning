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
import time
from datetime import datetime
import argparse

import numpy as np
import pickle
import tensorflow as tf

from dataset import TextDataset
from model import TextGenerationModel


def init_summary_writer(sess, save_path):
    # Optional to use.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return tf.summary.FileWriter(save_path, sess.graph)


def train(config):
    # Initialize the text dataset
    dataset = TextDataset(config.txt_file)

    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers
    )

    ###########################################################################
    # Implement code here.
    ###########################################################################

    # Reproducibility
    tf.set_random_seed(42)
    np.random.seed(42)

    # Utility vars and ops
    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem_frac, allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # logging
    train_logdir = os.path.join(config.summary_path, '{}_train'.format(config.model_name))
    train_log_writer = init_summary_writer(session, train_logdir)

    # Define the optimizer
    if config.optimizer.lower() == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate, decay=config.learning_rate_decay)
    elif config.optimizer.lower() == 'adam':
        optimizer = tf.train.AdamOptimizer(config.learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

    ###########################################################################
    # Implement code here.
    ###########################################################################

    summary_op = tf.summary.merge_all()
    session.run(fetches=[tf.global_variables_initializer(), tf.local_variables_initializer()])
    decoded_seqs = {}

    for train_step in range(int(config.train_steps)):

        # dim: [batch_size, time_step]
        batch_inputs, batch_labels = dataset.batch(batch_size=config.batch_size, seq_length=config.seq_length)

        # Time-major: [time_step, batch_size]
        batch_inputs, batch_labels = batch_inputs.T, batch_labels.T

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################################
        # Implement code here.
        #######################################################################
        train_feed = {model.inputs: batch_inputs,
                      model.labels: batch_labels,
                      model.initial_lstm_states: model.zero_state_numpy()}
        fetches = [model.loss, apply_gradients_op]
        if train_step % config.print_every == 0:
            fetches += [summary_op]
            loss, _, summary = session.run(feed_dict=train_feed, fetches=fetches)
            train_log_writer.add_summary(summary, train_step)
        else:
            loss, _ = session.run(feed_dict=train_feed, fetches=fetches)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step + 1,
                int(config.train_steps), config.batch_size, examples_per_second, loss))

        # Decode
        if train_step % config.sample_every == 0:
            t3 = time.time()
            decode_feed = {model.decode_length: config.decode_length,
                           model.initial_lstm_states: model.zero_state_numpy(),
                           model.random_initial_decoding_inputs: np.random.choice(a=dataset.vocab_size,
                                                                                  size=(config.batch_size))}
            fetches = [model.decoded_sequence]
            decoded_tokens = session.run(fetches=fetches, feed_dict=decode_feed)
            decoded_seqs[train_step] = decoded_tokens

            print('Decoded at train step {}, Sequences/Sec {:.2f}:',
                  format(train_step, config.batch_size / float(time.time() - t3)))

    train_log_writer.close()
    with open('{}_decoded_seqs.pkl', 'wb') as f:
        pickle.dump(decoded_seqs, f)

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--decode_length', type=int, default=30,
                        help='Inference (decoding) number of steps, int default is 30')
    parser.add_argument('--model_name', type=str, default='vanilla_rnn', help='Model name for saving')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'rmsprop'], default="RMSProp",
                        help='Optimizer, choose between adam and rmsprop')

    config = parser.parse_args()

    # Train the model
    train(config)
