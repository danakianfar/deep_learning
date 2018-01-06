"""
This module implements training and evaluation of a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import os
import json

from mlp_tf import MLP
import cifar10_utils
from util import Args
from collections import defaultdict
import pickle

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'
SAVE_PATH_DEFAULT = './trained_models/'

# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
WEIGHT_INITIALIZATION_DICT = {'xavier': lambda _: tf.contrib.layers.xavier_initializer(uniform=False),
                              # Xavier initialisation
                              'normal': lambda scale: tf.random_normal_initializer(stddev=scale),
                              # Initialization from a standard normal
                              'uniform': lambda scale: tf.random_uniform_initializer(minval=-scale, maxval=scale),
                              # Initialization from a uniform distribution
                              }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None,  # No regularization
                           'l1': lambda scale: tf.contrib.layers.l1_regularizer(scale=scale),  # L1 regularization
                           'l2': lambda scale: tf.contrib.layers.l2_regularizer(scale=scale)  # L2 regularization
                           }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn
ACTIVATION_DICT = {'relu': tf.nn.relu,  # ReLU
                   'elu': tf.nn.elu,  # ELU
                   'tanh': tf.nn.tanh,  # Tanh
                   'sigmoid': tf.nn.sigmoid}  # Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer,  # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer,  # Adadelta
                  'adagrad': tf.train.AdagradOptimizer,  # Adagrad
                  'adam': tf.train.AdamOptimizer,  # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

FLAGS = None


def _parse_flags(flags):
    activation_fn = ACTIVATION_DICT[flags.activation]
    dropout_rate = flags.dropout_rate
    weight_init_scale = flags.weight_init_scale
    weight_initializer = WEIGHT_INITIALIZATION_DICT[flags.weight_init](weight_init_scale)
    weight_regularizer_scale = flags.weight_reg_strength
    weight_regularizer = WEIGHT_REGULARIZER_DICT[flags.weight_reg]
    weight_regularizer = weight_regularizer(weight_regularizer_scale) if weight_regularizer is not None else None
    n_classes = 10
    optimizer = OPTIMIZER_DICT[flags.optimizer](learning_rate=flags.learning_rate)
    batch_size = flags.batch_size
    max_steps = flags.max_steps
    log_dir = flags.log_dir
    data_dir = flags.data_dir

    return activation_fn, dropout_rate, weight_initializer, weight_regularizer, n_classes, optimizer, batch_size, \
           max_steps, log_dir, data_dir


def _update_stats(stats, train_loss=None, train_accuracy=None, test_loss=None, test_accuracy=None,
                  test_confusion_matrix=None):
    """
    Utility function for collecting stats
    """
    if train_loss:
        stats['train_loss'].append(train_loss)
    if train_accuracy:
        stats['train_accuracy'].append(train_accuracy)
    if test_loss:
        stats['test_loss'].append(test_loss)
    if test_accuracy:
        stats['test_accuracy'].append(test_accuracy)
    if test_confusion_matrix is not None:
        stats['test_confusion_matrix'].append(test_confusion_matrix)

    return stats


def _ensure_path_exists(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


def train():
    """
    Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
    as you did in the task 1 of this assignment.
    """
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    tf.set_random_seed(42)
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Parameters
    input_dim = 3 * 32 * 32
    activation_fn, dropout_rate, weight_initializer, weight_regularizer, n_classes, optimizer, batch_size, max_steps, \
    log_dir, data_dir = _parse_flags(
        FLAGS)

    # dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir=data_dir)

    # Session
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Placeholders for images, labels input.
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='inputs')
    y = tf.placeholder(dtype=tf.int32, shape=[None, n_classes], name='labels')

    # init network
    net = MLP(n_hidden=dnn_hidden_units, n_classes=n_classes, is_training=True,
              activation_fn=activation_fn, dropout_rate=dropout_rate,
              weight_initializer=weight_initializer,
              weight_regularizer=weight_regularizer)

    # Trainings ops
    global_step = tf.Variable(0, trainable=False, name='global_step')
    logits_op = net.inference(X)
    train_flags = {'optimizer': optimizer, 'global_step': global_step, 'grad_clipping': FLAGS.grad_clipping}
    loss_op = net.loss(logits_op, y)
    accuracy_op = net.accuracy(logits_op, y)
    train_op = net.train_step(loss_op, train_flags)
    confusion_matrix_op = net.confusion_matrix(logits=logits_op, labels=y)
    train_loss = train_accuracy = test_accuracy = test_loss = 0.

    # utility ops
    summary_op = tf.summary.merge_all()
    write_logs = FLAGS.log_dir is not None
    save_model = True

    if write_logs:
        train_log_path = os.path.join(log_dir, '{}_train'.format(FLAGS.model_name))
        test_log_path = os.path.join(log_dir, '{}_test'.format(FLAGS.model_name))
        _ensure_path_exists(train_log_path)
        _ensure_path_exists(test_log_path)
        train_log_writer = tf.summary.FileWriter('{}_train/'.format(train_log_path), graph=session.graph)
        test_log_writer = tf.summary.FileWriter('{}_test/'.format(test_log_path), graph=session.graph)

    # Initialize variables
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    session.run(fetches=[init_op, local_init_op])

    # track losses
    stats = defaultdict(list)

    # loop over steps
    for _step in range(FLAGS.max_steps):

        # get batch of data
        X_train, y_train = cifar10.train.next_batch(batch_size)
        X_train = np.reshape(X_train, (batch_size, -1))
        # feed to model
        train_feed = {X: X_train, y: y_train, net.training_mode: True}
        fetches = [train_op, loss_op, accuracy_op]

        # Training set
        if _step % 13 == 0 and write_logs:  # write summary
            fetches += [summary_op]
            _, train_loss, train_accuracy, train_summary = session.run(fetches=fetches, feed_dict=train_feed)
            train_log_writer.add_summary(train_summary, _step)
        else:
            _, train_loss, train_accuracy = session.run(fetches=fetches, feed_dict=train_feed)

        if _step % 10 == 0:
            print('Ep.{}: train_loss:{:+.4f}, train_accuracy:{:+.4f}'.format(_step, train_loss, train_accuracy))
            stats = _update_stats(stats, train_loss=train_loss, train_accuracy=train_accuracy)

        # Sanity check
        if np.isnan(train_loss):
            print('Warning: training loss is NaN.. ')
            break

        # Test set evaluation
        if (_step + 1) % 100 == 0:
            X_test, y_test = cifar10.test.images, cifar10.test.labels
            X_test = np.reshape(X_test, [X_test.shape[0], -1])
            test_feed = {X: X_test, y: y_test, net.training_mode: False}
            test_loss, test_accuracy, test_logits, test_confusion_matrix, test_summary = session.run(
                fetches=[loss_op, accuracy_op, logits_op,
                         confusion_matrix_op, summary_op],
                feed_dict=test_feed)

            if write_logs:
                test_log_writer.add_summary(test_summary, _step)

            stats = _update_stats(stats, test_loss=test_loss, test_accuracy=test_accuracy,
                                  test_confusion_matrix=test_confusion_matrix)
            print('==> Ep.{}: test_loss:{:+.4f}, test_accuracy:{:+.4f}'.format(_step, test_loss, test_accuracy))
            print('==> Confusion Matrix on test set \n {} \n'.format(test_confusion_matrix))

        if _step > 1000 and test_accuracy < 0.25:  # hopeless trials
            save_model = False
            break

        # Early stopping: if the last test accuracy is not above the mean of prev 10 epochs, stop
        delta = 1e-4  # accuracy is in decimals
        if _step > 1000:
            window = stats['test_accuracy'][-10:]
            window_accuracy = sum(window) / len(window)

            if abs(test_accuracy - window_accuracy) < delta:
                print(
                    '\n==> EARLY STOPPING with accuracy {} and moving-window mean accuracy {} \n'.format(test_accuracy,
                                                                                                         window_accuracy))

    # save model
    if write_logs:
        train_log_writer.close()
        test_log_writer.close()

    if save_model:
        save_dir = os.path.join(FLAGS.save_path, FLAGS.model_name)
        saver = tf.train.Saver()
        _ensure_path_exists(save_dir)
        saver.save(session, save_path=os.path.join(save_dir, 'model.ckpt'))

        # save results for easy plotting
        results_dir = os.path.relpath('./results')
        _ensure_path_exists(results_dir)
        with open(os.path.join(results_dir, '{}.pkl'.format(FLAGS.model_name)), 'wb') as f:
            pickle.dump(stats, f)


            #######################
            # END OF YOUR CODE    #
            #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main(_):
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    # Make directories if they do not exists yet
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--weight_init', type=str, default=WEIGHT_INITIALIZATION_DEFAULT,
                        choices=['xavier', 'normal', 'uniform'],
                        help='Weight initialization type [xavier, normal, uniform].')
    parser.add_argument('--weight_init_scale', type=float, default=WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                        help='Float, Weight initialization scale (e.g. std of a Gaussian).')
    parser.add_argument('--weight_reg', type=str, default=WEIGHT_REGULARIZER_DEFAULT,
                        choices=['none', 'l1', 'l2'],
                        help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
    parser.add_argument('--weight_reg_strength', type=float, default=WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                        help='Float, Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE_DEFAULT,
                        help='Dropout rate.')
    parser.add_argument('--activation', type=str, default=ACTIVATION_DEFAULT,
                        choices=['relu', 'elu', 'tanh', 'sigmoid'],
                        help='Activation function [relu, elu, tanh, sigmoid].')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                        choices=['sgd', 'adadelta', 'adagrad', 'adam', 'rmsprop'],
                        help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')

    # Custom args
    parser.add_argument('--grad_clipping', action='store_true',
                        help='Performs gradient clipping')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH_DEFAULT,
                        help='save path directory')
    parser.add_argument('--model_name', type=str, default='mlp_tf',
                        help='model_name')
    parser.add_argument('--train_settings_path', type=str, default=None,
                        help='Path to a file with training settings that will override the CLI args.')
    parser.add_argument('--grid_search', action='store_true')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
