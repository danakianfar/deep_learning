import tensorflow as tf
import numpy as np


def load_mnist_images(binarize=True):
    """
    :param binarize: Turn the images into binary vectors
    :return: x_train, x_test  Where
        x_train is a (55000 x 784) tensor of training images
        x_test is a  (10000 x 784) tensor of test images
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    x_train = mnist.train.images
    x_test = mnist.test.images
    if binarize:
        x_train = (x_train>0.5).astype(x_train.dtype)
        x_test = (x_test>0.5).astype(x_test.dtype)
    return x_train, x_test


class NaiveBayesModel(object):

    def __init__(self, w_init, b_init = None, c_init = None):
        """
        :param w_init: An (n_categories, n_dim) array, where w[i, j] represents log p(X[j]=1 | Z[i]=1)
        :param b_init: A (n_categories, ) vector where b[i] represents log p(Z[i]=1), or None to fill with zeros
        :param c_init: A (n_dim, ) vector where b[j] represents log p(X[j]=1), or None to fill with zeros
        """
        raise NotImplementedError()

    def log_p_x_given_z(self, x, z):
        """
        :param x: An (n_samples, n_dims) tensor
        :param z: An (n_samples, n_labels) tensor of integer class labels
        :return: An (n_samples, n_labels) tensor  p_x_given_z where result[i, j] indicates p(X=x[i] | Z=z[j])
        """
        raise NotImplementedError()

    def log_p_x(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of log-probabilities assigned to each point
        """
        raise NotImplementedError()

    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        raise NotImplementedError()


def train_simple_generative_model_on_mnist(n_categories=20, initial_mag = 0.01, optimizer='rmsprop', learning_rate=.01, n_epochs=20, test_every=100,
                                           minibatch_size=100, plot_n_samples=16):
    """
    Train a simple Generative model on MNIST and plot the results.

    :param n_categories: Number of latent categories (K in assignment)
    :param initial_mag: Initial weight magnitude
    :param optimizer: The name of the optimizer to use
    :param learning_rate: Learning rate for the optimization
    :param n_epochs: Number of epochs to train for
    :param test_every: Test every X iterations
    :param minibatch_size: Number of samples in a minibatch
    :param plot_n_samples: Number of samples to plot
    """

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    # Build the model
    raise NotImplementedError('Insert Model Building Code')

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        n_steps = (n_epochs * n_samples)/minibatch_size
        for i in xrange(n_steps):
            if i%test_every==0:
                raise NotImplementedError('INSERT CODE TO RUN TEST AND RECORD LOG-PROB PERIODICALLY')

            raise NotImplementedError('COMPUTE TRAINING UPDATES HERE')


if __name__ == '__main__':
    train_simple_generative_model_on_mnist()
