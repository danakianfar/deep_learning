from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils

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
  input_dim = 3 * 32 * 32
  # activation_fn, dropout_rate, weight_initializer, weight_regularizer, n_classes, optimizer, batch_size, max_steps, \
  # log_dir, data_dir = _parse_flags(
  #     FLAGS)
  data_dir = DATA_DIR_DEFAULT


  # dataset
  cifar10 = cifar10_utils.get_cifar10(data_dir=data_dir)

  # Session
  tf.reset_default_graph()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
  session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  # Placeholders for images, labels input.
  X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
  y = tf.placeholder(dtype=tf.int32, shape=[None, n_classes])

  # init network
  net = MLP(n_hidden=dnn_hidden_units, n_classes=n_classes, is_training=True,
            activation_fn=activation_fn, dropout_rate=dropout_rate,
            weight_initializer=weight_initializer,
            weight_regularizer=weight_regularizer)

  # Trainings ops
  net.is_training = True
  global_step = tf.Variable(0, trainable=False, name='global_step')
  logits_op = net.inference(X)
  train_flags = {'optimizer': optimizer, 'global_step': global_step, 'grad_clipping': FLAGS.grad_clipping}
  loss_op = net.loss(logits_op, y)
  accuracy_op = net.accuracy(logits_op, y)
  train_op = net.train_step(loss_op, train_flags)
  confusion_matrix_op = net.confusion_matrix(logits=logits_op, labels=y)

  # Inference ops
  net.is_training = False
  logits_deterministic_op = net.inference(X)
  loss_deterministic_op = net.loss(logits_deterministic_op, y)
  accuracy_deterministic_op = net.accuracy(logits_deterministic_op, y)
  confusion_matrix_deterministic_op = net.confusion_matrix(logits=logits_deterministic_op, labels=y)
  net.is_training = True  # revert back
  train_loss = train_accuracy = test_accuracy = test_loss = 0.

  # utility ops
  summary_op = tf.summary.merge_all()
  write_logs = FLAGS.log_dir is not None
  save_model = True

  if write_logs:
      log_path = os.path.join(log_dir, FLAGS.model_name)
      if not tf.gfile.Exists(log_path):
          tf.gfile.MakeDirs(log_path)
      log_writer = tf.summary.FileWriter(log_path, graph=session.graph)

  # Initialize variables
  init_op = tf.global_variables_initializer()
  local_init_op = tf.local_variables_initializer()
  session.run(fetches=[init_op, local_init_op])

  # track losses
  stats = defaultdict(list)

  # loop over steps
  for _step in range(FLAGS.max_steps):

      # get batch of data
      inputs, labels = cifar10.train.next_batch(batch_size)
      inputs = np.reshape(inputs, (batch_size, -1))
      # feed to model
      train_feed = {X: inputs, y: labels}
      fetches = [train_op, loss_op, accuracy_op]

      if _step % 13 == 0 and write_logs:  # write summary
          fetches += [summary_op]
          _, train_loss, train_accuracy, train_summary = session.run(fetches=fetches, feed_dict=train_feed)
          log_writer.add_summary(train_summary, _step)
      else:
          _, train_loss, train_accuracy = session.run(fetches=fetches, feed_dict=train_feed)

      if _step % 10 == 0:
          print('Ep.{}: train_loss:{:+.4f}, train_accuracy:{:+.4f}'.format(_step, train_loss, train_accuracy))
          stats = _update_stats(stats, test_loss=train_loss, test_accuracy=train_accuracy)

      # Sanity check
      if np.isnan(train_loss):
          print('Warning: training loss is NaN.. ')
          break

      # eval on test set every 100 steps
      if (_step + 1) % 100 == 0:
          X_test, y_test = cifar10.test.images, cifar10.test.labels
          X_test = np.reshape(X_test, [X_test.shape[0], -1])
          test_feed = {X: X_test, y: y_test}
          test_loss, test_accuracy, test_logits, confusion_matrix = session.run(
              fetches=[loss_deterministic_op, accuracy_deterministic_op, logits_deterministic_op,
                       confusion_matrix_deterministic_op],
              feed_dict=test_feed)

          stats = _update_stats(stats, train_loss=train_loss, train_accuracy=train_accuracy,
                                test_confusion_matrix=confusion_matrix)
          print('==> Ep.{}: test_loss:{:+.4f}, test_accuracy:{:+.4f}'.format(_step, test_loss, test_accuracy))
          print('==> Confusion Matrix on test set \n {} \n'.format(confusion_matrix))

      # Early stopping: if the last test accuracy is not above the mean of prev 10 epochs, stop
      delta = 1e-4  # accuracy is in decimals
      if _step > 1000:
          window = stats['test_accuracy'][-10:-5]
          window_accuracy = sum(window) / len(window)

          if abs(test_accuracy - window_accuracy) < delta:
              print('\n==> EARLY STOPPING with accuracy {} and moving-window mean accuracy {} \n'.format(test_accuracy,
                                                                                                         window_accuracy))
              if test_accuracy < 0.3:
                  save_model = False
              break

      if _step > 1000 and test_accuracy < 0.2:  # hopeless
          save_model = False
          break

  # save model
  if write_logs:
      log_writer.close()

  if save_model:
      save_dir = os.path.join(FLAGS.save_path, FLAGS.model_name)
      saver = tf.train.Saver(var_list=None)
      if not tf.gfile.Exists(save_dir):
          tf.gfile.MakeDirs(save_dir)
      saver.save(session, save_path=os.path.join(save_dir, 'model.ckpt'))

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
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
