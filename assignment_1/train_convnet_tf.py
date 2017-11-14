from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet_tf import ConvNet
from collections import defaultdict
import pickle

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'


def _update_stats(stats, train_loss=None, train_accuracy=None, test_loss=None, test_accuracy=None,
                  test_confusion_matrix=None):
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


def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as savers and summarizers. Finally, initialize
    your model within a tf.Session and do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    # Parameters
    input_dim = (32, 32, 3)
    # activation_fn, dropout_rate, weight_initializer, weight_regularizer, n_classes, optimizer, batch_size, max_steps, \
    # log_dir, data_dir = _parse_flags(
    #     FLAGS)
    n_classes = 10
    eta = FLAGS.learning_rate
    optimizer = tf.train.AdamOptimizer(learning_rate=eta)
    batch_size = FLAGS.batch_size

    # dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir)

    # Session
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Placeholders for images, labels input.
    X = tf.placeholder(dtype=tf.float32, shape=[None, *input_dim])
    y = tf.placeholder(dtype=tf.int32, shape=[None, n_classes])

    # init network
    net = ConvNet(n_classes=n_classes)

    # Trainings ops
    net.is_training = True
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
    train_log_path = os.path.join(FLAGS.log_dir, 'train')
    test_log_path = os.path.join(FLAGS.log_dir, 'test')
    if not tf.gfile.Exists(train_log_path):
        tf.gfile.MakeDirs(train_log_path)
    if not tf.gfile.Exists(test_log_path):
        tf.gfile.MakeDirs(train_log_path)
    train_log_writer = tf.summary.FileWriter(train_log_path, graph=session.graph)
    test_log_writer = tf.summary.FileWriter(test_log_path, graph=session.graph)

    # Initialize variables
    global_init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    session.run(fetches=[global_init_op, local_init_op])
    saver = tf.train.Saver(max_to_keep=20)

    # track losses
    stats = defaultdict(list)

    # loop over steps
    for _step in range(FLAGS.max_steps):

        # get batch of data
        inputs, labels = cifar10.train.next_batch(batch_size)

        # feed to model
        train_feed = {X: inputs, y: labels}
        fetches = [train_op]

        if _step % FLAGS.print_freq == 0:  # write summary and eval on train set
            fetches += [loss_op, accuracy_op, summary_op]
            _, train_loss, train_accuracy, train_summary = session.run(fetches=fetches, feed_dict=train_feed)
            train_log_writer.add_summary(train_summary, _step)

            print('Ep.{}: train_loss:{:+.4f}, train_accuracy:{:+.4f}'.format(_step, train_loss, train_accuracy))
            stats = _update_stats(stats, train_loss=train_loss, train_accuracy=train_accuracy)
        else:
            _ = session.run(fetches=fetches, feed_dict=train_feed)

        # Sanity check
        if np.isnan(train_loss):
            print('\n\n==> WARNING: training loss is NaN\n\n')
            break

        # eval on test set every 100 steps
        if _step % FLAGS.eval_freq == 0:
            X_test, y_test = cifar10.test.images, cifar10.test.labels
            test_feed = {X: X_test, y: y_test}
            test_loss, test_accuracy, test_confusion_matrix, test_summary = session.run(
                fetches=[loss_op, accuracy_op, confusion_matrix_op, summary_op],
                feed_dict=test_feed)
            test_log_writer.add_summary(test_summary, _step)
            stats = _update_stats(stats, test_loss=test_loss, test_accuracy=test_accuracy,
                                  test_confusion_matrix=test_confusion_matrix)
            print('==> Ep.{}: test_loss:{:+.4f}, test_accuracy:{:+.4f}'.format(_step, test_loss, test_accuracy))
            print('==> Confusion Matrix on test set \n {} \n'.format(test_confusion_matrix))

        if _step % FLAGS.checkpoint_freq == 0:
            saver.save(session, save_path=os.path.join(FLAGS.checkpoint_dir, 'model.ckpt'))

    # save model
    train_log_writer.close()
    test_log_writer.close()

    # save results for easy plotting
    results_dir = os.path.relpath('./results')
    if not tf.gfile.Exists(results_dir):
        tf.gfile.MakeDirs(results_dir)

    with open(os.path.join(results_dir, '{}.pkl'.format(FLAGS.model_name)), 'wb') as f:
        pickle.dump(stats, f)

        ########################
        # END OF YOUR CODE    #
        ########################


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main(_):
    print_flags()

    initialize_folders()

    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    parser.add_argument('--model_name', type=str, default='convnet_default',
                        help='model_name')
    parser.add_argument('--grad_clipping', type=bool, default=True,
                        help='gradient clipping to [-1.,1.]')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
