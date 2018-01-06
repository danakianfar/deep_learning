import numpy as np
import tensorflow as tf
import matplotlib
import pickle

matplotlib.use('agg')
from matplotlib import pyplot as plt


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
        x_train = (x_train > 0.5).astype(x_train.dtype)
        x_test = (x_test > 0.5).astype(x_test.dtype)
    return x_train, x_test


def plot(samples, save_path='figs/vae', fname=None, probs=None):
    assert fname is not None

    N = samples.shape[0]
    _ensure_path_exists(save_path)

    fig, axs = plt.subplots(2, N // 2, figsize=(2 * (N // 2), 2 * 2),
                            gridspec_kw={'wspace': 0.0, 'hspace': 1.0})  # , squeeze=True)
    axs = axs.ravel()

    for i in range(N):
        axs[i].imshow(samples[i], cmap='gray', aspect='auto')
        axs[i].axis('off')

        if probs is not None:
            axs[i].set_title('$lp=({:.1f})$'.format(probs[i]))

    # Store figure.
    plt.show()
    fig.savefig(save_path + '/{}.png'.format(fname))
    # plt.tight_layout()
    plt.close()


def init_summary_writer(sess, save_path):
    # Optional to use.
    _ensure_path_exists(save_path)
    return tf.summary.FileWriter(save_path, sess.graph)


def _ensure_path_exists(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


class VariationalAutoencoder(object):
    def __init__(self, encoder_hidden_sizes, decoder_hidden_sizes, input_dim, z_dim, activation_fn, initializer):
        self._encoder_hidden_sizes = encoder_hidden_sizes
        self._decoder_hidden_sizes = decoder_hidden_sizes
        self._activation_fn = activation_fn
        self.kernel_initializer = initializer
        self._input_dim = self._output_dim = input_dim
        self._z_dim = z_dim

    def Q(self, x):
        """
        Encodes data input x to hidden states
        :param x: float tensor, [batch_size, input_dim]
        :return: tuple of encoded variational params, (mu [batch_size, z_dim], sigma^2 [batch_size, z_dim])
        """
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            for i, dim in enumerate(self._encoder_hidden_sizes):
                x = tf.layers.dense(inputs=x, units=dim, activation=self._activation_fn,
                                    kernel_initializer=self.kernel_initializer, name='enc_fc_{}'.format(i))

            # Encode
            mu = tf.layers.dense(inputs=x, units=self._z_dim, activation=tf.identity,
                                 kernel_initializer=self.kernel_initializer, name='enc_mu')

            # softplus activation: ensure positivity
            var = tf.layers.dense(inputs=x, units=self._z_dim, activation=tf.exp,
                                  kernel_initializer=self.kernel_initializer, name='enc_sigma')
        return mu, var

    def P(self, z):
        """
        Decodes the data given latent codes
        :param z: latennt code, float tensor [batch_size, z_dim]
        :return: log p(X=1|Z) [batch_size, input_dim]
        """
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            for i, dim in enumerate(self._decoder_hidden_sizes):
                z = tf.layers.dense(inputs=z, units=dim, activation=self._activation_fn,
                                    kernel_initializer=self.kernel_initializer, name='dec_fc_{}'.format(i))

            # Decode
            log_p_x = tf.layers.dense(inputs=z, units=self._input_dim, activation=tf.log_sigmoid,
                                      kernel_initializer=self.kernel_initializer, name='decoder')
        return log_p_x

    def lower_bound(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of the lower-bound on the log-probability of each data point
        """
        mu, var = self.Q(x)  # [batch_size, z_dim]
        sigma = tf.sqrt(var)

        # Reparameterization trick, num samples = 1
        epsilon = tf.random_normal(shape=tf.shape(var), dtype=tf.float32)
        z = mu + epsilon * sigma  # [batch_size, z_dim]

        # Complexity cost
        kl = tf.reduce_sum(tf.log(sigma) + 0.5 * (- var - tf.pow(mu, 2) + 1), axis=-1)  # [N]
        tf.summary.scalar('Mean KL Divergence', tf.reduce_mean(kl))

        # Reconstruction
        log_x_pred = self.P(z)  # [batch_size, input_dim]
        thetas = tf.exp(log_x_pred)
        thetas = tf.clip_by_value(thetas, 1e-6, 1 - 1e-6)

        recon_loss = tf.reduce_sum(x * log_x_pred + (1 - x) * tf.log(1 - thetas), axis=-1)  # [N]
        tf.summary.scalar('Mean Reconstruction Error', tf.reduce_mean(recon_loss))

        # ELBO
        elbo = kl + recon_loss
        mean_eblo = tf.reduce_mean(elbo)
        tf.summary.scalar('Mean ELBO', mean_eblo)

        return mean_eblo

    def mean_x_given_z(self, z):
        """
        :param z: A (n_samples, n_dim_z) tensor containing a set of latent data points (n_samples, n_dim_z)
        :return: A (n_samples, n_dim_x) tensor containing the mean of p(X|Z=z) for each of the given points
        """
        return tf.exp(self.P(z))

    def sample(self, n_samples, sample_x=False):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        z_samples = tf.random_normal(shape=(n_samples, self._z_dim), dtype=tf.float32)  # [n_samples, z_dim]
        thetas = self.mean_x_given_z(z_samples)  # [n_samples, input_dim]

        # Evaluates p(X=1|z)
        if not sample_x:
            return tf.reshape(thetas, shape=(-1, 28, 28))

        return tf.reshape(tf.distributions.Bernoulli(probs=thetas).sample(), shape=(-1, 28, 28))


def train_vae_on_mnist(z_dim=2, kernel_initializer='glorot_uniform', optimizer='adam', learning_rate=0.001,
                       n_epochs=50,
                       test_every=100, minibatch_size=100, encoder_hidden_sizes=[200, 200],
                       decoder_hidden_sizes=[200, 200],
                       hidden_activation='relu', plot_grid_size=10, plot_n_samples=20):
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
    x_train, x_test = 1 - x_train, 1 - x_test[:1000]
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(
        minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    # Build Model
    vae = VariationalAutoencoder(input_dim=n_dims, z_dim=z_dim,
                                 initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                 activation_fn=tf.nn.relu,
                                 decoder_hidden_sizes=decoder_hidden_sizes, encoder_hidden_sizes=encoder_hidden_sizes)

    # Placeholder
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, n_dims])
    manifold_input = tf.placeholder(dtype=tf.float32, shape=[None, z_dim])
    std3 = np.linspace(-3, 3, plot_grid_size)
    grid1, grid2 = np.meshgrid(std3, std3)
    manifold = np.empty((plot_grid_size * 28, plot_grid_size * 28))

    # Build the model
    elbo_op = vae.lower_bound(input_data)
    loss_op = - elbo_op

    # Utility vars and ops
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_op)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=10.)
    train_op = optimizer.apply_gradients(zip(grads_clipped, variables),
                                         global_step=global_step)
    # train_op = optimizer.minimize(loss_op, global_step=global_step)
    summary_op = tf.summary.merge_all()

    # Sampling
    samples_op = vae.sample(n_samples=plot_n_samples, sample_x=True)
    mean_x_given_z_op = vae.sample(n_samples=plot_n_samples, sample_x=False)

    train_elbos = []
    test_elbos = []

    with tf.Session() as sess:

        train_log_writer = init_summary_writer(sess, './summaries/vae/train')
        test_log_writer = init_summary_writer(sess, './summaries/vae/test')

        sess.run(train_iterator.initializer)  # Initialize the variables of the data-loader.
        sess.run(tf.global_variables_initializer())  # Initialize the model parameters.
        n_steps = (n_epochs * n_samples) // minibatch_size

        for i in range(n_steps):
            # Test
            if i % test_every == 0:
                summary, test_loss, samples, mean_samples = sess.run(feed_dict={input_data: x_test},
                                                                     fetches=[summary_op, loss_op, samples_op,
                                                                              mean_x_given_z_op])
                print('{}/{}: Test Loss: {}'.format(i, n_steps, test_loss))
                plot(samples, fname='samples_{}'.format(i))
                plot(mean_samples, fname='mean_x_cond_z_{}'.format(i))
                test_log_writer.add_summary(summary, i)
                test_elbos.append((i, -test_loss))

            # Train
            summary, train_loss, _ = sess.run(feed_dict={input_data: sess.run(x_minibatch)},
                                              fetches=[summary_op, loss_op, train_op])
            print('{}/{}: Train Loss: {}'.format(i, n_steps, train_loss))
            train_log_writer.add_summary(summary, i)
            train_elbos.append((i, -train_loss))

        # Manifold
        for i in range(plot_grid_size):
            for j in range(plot_grid_size):
                # Get linspaced values in grid at i,j
                z = np.array([grid1[i, j], grid2[i, j]], dtype=np.float32)[None, :]

                # Decode image from z sample
                img = sess.run(feed_dict={manifold_input: z}, fetches=[vae.mean_x_given_z(manifold_input)])

                # save
                manifold[28 * (i):28 * (i + 1), 28 * (j):28 * (j + 1)] = np.reshape(img[0].squeeze(), (28, 28))

        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(manifold, cmap='gray')
        plt.title('Manifold')
        plt.savefig('./figs/vae/manifold.png')
        plt.close()

        with open('elbos.pkl', 'wb') as f:
            pickle.dump((train_elbos, test_elbos), f)


if __name__ == '__main__':
    train_vae_on_mnist()
