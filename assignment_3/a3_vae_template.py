import numpy as np
import tensorflow as tf


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


class VariationalAutoencoder(object):

    def lower_bound(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of the lower-bound on the log-probability of each data point
        """
        raise NotImplementedError()


    def mean_x_given_z(self, z):
        """
        :param z: A (n_samples, n_dim_z) tensor containing a set of latent data points (n_samples, n_dim_z)
        :return: A (n_samples, n_dim_x) tensor containing the mean of p(X|Z=z) for each of the given points
        """
        raise NotImplementedError()

    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        raise NotImplementedError()


def train_vae_on_mnist(z_dim=2, kernel_initializer='glorot_uniform', optimizer = 'adam',  learning_rate=0.001, n_epochs=4000,
        test_every=100, minibatch_size=100, encoder_hidden_sizes=[200, 200], decoder_hidden_sizes=[200, 200],
        hidden_activation='relu', plot_grid_size=10, plot_n_samples = 20):
    """
    Train a variational autoencoder on MNIST and plot the results.

    :param z_dim: The dimensionality of the latent space.
    :param kernel_initializer: How to initialize the weight matrices (see tf.keras.layers.Dense)
    :param optimizer: The optimizer to use
    :param learning_rate: The learning rate for the optimizer
    :param n_epochs: Number of epochs to train
    :param test_every: Test every X training iterations
    :param minibatch_size: Number of samples per minibatch
    :param encoder_hidden_sizes: Sizes of hidden layers in encoder
    :param decoder_hidden_sizes: Sizes of hidden layers in decoder
    :param hidden_activation: Activation to use for hidden layers of encoder/decoder.
    :param plot_grid_size: Number of rows, columns to use to make grid-plot of images corresponding to latent Z-points
    :param plot_n_samples: Number of samples to draw when plotting samples from model.
    """

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    # Build Model
    raise NotImplementedError("Build the model here")


    with tf.Session() as sess:
        sess.run(train_iterator.initializer)  # Initialize the variables of the data-loader.
        sess.run(tf.global_variables_initializer())  # Initialize the model parameters.
        n_steps = (n_epochs * n_samples)/minibatch_size
        for i in xrange(n_steps):
            if i%test_every==0:
                raise NotImplementedError('INSERT CODE TO RUN TEST AND RECORD LOG-PROB PERIODICALLY')

            raise NotImplementedError('CALL TRAINING FUNCTION HERE')


if __name__ == '__main__':
    train_vae_on_mnist()
