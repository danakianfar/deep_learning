"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import cifar10_utils
from mlp_numpy import MLP

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DNN_HIDDEN_UNITS_DEFAULT = '100'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def train():
    """
    Performs training and evaluation of MLP model. Evaluate your model on the whole test set each 100 iterations.
    """
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
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

    # dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)

    learning_rate = FLAGS.learning_rate
    weight_init_scale = FLAGS.weight_init_scale
    weight_reg_strength = FLAGS.weight_reg_strength
    batch_size = FLAGS.batch_size
    n_classes = 10
    input_dim = 3 * 32 * 32

    net = MLP(n_hidden=dnn_hidden_units, n_classes=n_classes, input_dim=input_dim, weight_decay=weight_reg_strength,
              weight_scale=weight_init_scale)
    print(net)

    for _step in range(FLAGS.max_steps):

        net.training_mode = True

        X_train, y_train = cifar10.train.next_batch(batch_size)
        X_train = np.reshape(X_train, (batch_size, -1))

        # Feed forward
        logits_train = net.inference(X_train)

        # Obtain loss and accuracy
        train_loss = net.loss(logits_train, y_train)
        train_accuracy = net.accuracy(logits_train, y_train)

        print('Ep.{}: train_loss:{:.4f}, train_accuracy:{:.4f}'.format(_step, train_loss, train_accuracy))

        train_flags = {'learning_rate': learning_rate, 'batch_size': batch_size}
        net.train_step(loss=train_loss, flags=train_flags)

        if _step % 50 == 0:
            net.training_mode = False
            X_test, y_test = cifar10.test.images, cifar10.test.labels
            X_test = np.reshape(X_test, [X_test.shape[0], -1])

            # Feed forward
            logits_test = net.inference(X_test)

            # Obtain loss and accuracy
            test_loss = net.loss(logits_test, y_test)
            test_accuracy = net.accuracy(logits_test, y_test)

            print('\t\ttest_loss:{:.4f}, test_accuracy:{:.4f}'.format(test_loss, test_accuracy))

    # Print stats
    # net.plot_stats()
    print('Done training.')
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

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
    parser.add_argument('--weight_init_scale', type=float, default=WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                        help='Weight initialization scale (e.g. std of a Gaussian).')
    parser.add_argument('--weight_reg_strength', type=float, default=WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                        help='Regularizer strength for weights of fully-connected layers.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
