import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

class NaiveBayesModel(object):
    def __init__(self, w_init, b_init=None, c_init=None):
        """
        :param w_init: An (n_categories, n_dim) array, where w[i, j] represents log p(X[j]=1 | Z[i]=1)
        :param b_init: A (n_categories, ) vector where b[i] represents log p(Z[i]=1), or None to fill with zeros
        :param c_init: A (n_dim, ) vector where b[j] represents log p(X[j]=1), or None to fill with zeros
        """
        self._K, self._D = w_init.shape
        self._W = tf.Variable(initial_value=w_init, name='W', dtype=tf.float32)

        if b_init is None:
            self._b = tf.get_variable(name='b', dtype=tf.float32, shape=(self._K),
                                      initializer=tf.zeros_initializer())
        else:
            self._b = tf.Variable(initial_value=b_init, name='b', dtype=tf.float32)

        if c_init is None:
            self._c = tf.get_variable(name='c', dtype=tf.float32, shape=(self._D),
                                      initializer=tf.zeros_initializer())
        else:
            self._c = tf.Variable(initial_value=c_init, name='c', dtype=tf.float32)

        self._logits_x = tf.add(self._W, self._c, name='logits_p_x')  # [K,D]
        self._logits_z = tf.nn.softmax(self._b, dim=-1, name='logits_p_z')  # [K]

        # Distributions
        # P(Z)
        self._categorical = tf.distributions.Categorical(probs=self._logits_z,
                                                         dtype=tf.int32,
                                                         validate_args=False,
                                                         allow_nan_stats=True,
                                                         name='p_z')

        # P(X|Z)
        self._bernoulli = tf.distributions.Bernoulli(logits=self._logits_x,
                                                     dtype=tf.int32,
                                                     validate_args=False,
                                                     allow_nan_stats=True,
                                                     name='p_x_cond_z')

    def log_p_x_given_z(self, x):
        """
        :param x: An (n_samples, n_dims) tensor
        :return: An (n_samples, n_categories) tensor  p_x_given_z where result[i, j] indicates p(X=x[i] | Z=z[j])
        """
        X = tf.tile(tf.expand_dims(x, axis=1), multiples=(1, self._K, 1))  # [N,K,D]
        nkd_logp = self._bernoulli.log_prob(value=X, name='logp_x_cond_z')
        return tf.reduce_sum(nkd_logp, axis=-1)  # [N, K]

    def log_p_z(self):
        """
        Returns log p(Z) = log softmax(b)
        :return: [K] float tensor
        """
        return tf.nn.log_softmax(logits=self._b, dim=-1, name='log_p_z')  # [K]

    def log_p_x(self, x):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of log-probabilities assigned to each point
        """
        return tf.reduce_logsumexp(self.log_p_z() + self.log_p_x_given_z(x), axis=-1, name='log_p_x')  # [N]

    def nll(self, x):
        """
        Computes the average negative log-likehood of X given the model.
        :param x: input tensor [N,D]
        :return: scalar
        """
        return - tf.reduce_mean(self.log_p_x(x), name='nll')

    def sample_all_z(self):
        """
        Problem 6: for all k
        :return: float [K, 28,28]
        """
        samples = self._bernoulli.mean()  # p(X=1|Z) float, [K,D]
        return tf.reshape(tensor=samples, shape=[self._K, 28, 28])

    def sample(self, n_samples, ):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        z_samples = self._categorical.sample(sample_shape=n_samples)  # integer, [n_samples]
        x_samples = self._bernoulli.sample(sample_shape=1)  # binary, [1,K,D]

        samples = tf.gather(x_samples, z_samples, axis=1)  # [n_samples,D]
        return tf.reshape(tensor=samples, shape=[-1, 28, 28])


def get_frankenstein_images(x_test, labels_test):
    """
    Retrieve n images per class of the mnist dataset. Stored
    in a dictionary.
    """
    idx = 0
    imgs = {}
    while True:
        imgs[labels_test[idx]] = x_test[idx].reshape((28, 28))
        idx += 1

        if len(imgs) == 10:
            break

    normal = [imgs[i] for i in range(10)]
    frank = []
    indices = set(list(range(10)))
    for idx, img in enumerate(normal):
        left = img.copy()

        rand_idx = np.random.choice(list(indices - {idx}))
        left[..., 28 // 2:] = normal[rand_idx][..., 28 // 2:]
        frank.append(left)

    return np.array([x.reshape((-1)) for x in frank]), np.array([x.reshape((-1)) for x in normal])


def init_summary_writer(sess, save_path):
    # Optional to use.
    _ensure_path_exists(save_path)
    return tf.summary.FileWriter(save_path, sess.graph)


def _ensure_path_exists(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MakeDirs(path)


def load_mnist_images(binarize=True, return_labels=False):
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

    if return_labels:
        return (x_train, mnist.train.labels), (x_test, mnist.test.labels)
    else:
        return x_train, x_test


def plot(samples, title, save_path='figs/naivebayes', fname=None, probs=None):
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


def train_simple_generative_model_on_mnist(n_categories=20, initial_mag=0.01, optimizer='rmsprop', learning_rate=.01,
                                           n_epochs=20, test_every=100,
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
    tf.reset_default_graph()

    # Get Data
    (x_train, _), (x_test, labels_test) = load_mnist_images(binarize=True, return_labels=True)
    x_train, x_test = 1 - x_train, 1 - x_test
    x_test = x_test[:1000]  # doesn't fit on my gpu

    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(
        minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors
    frankenstein, normal = get_frankenstein_images(x_test, labels_test)

    # Utility vars and ops
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # Build the model
    w_init = np.random.standard_normal(size=(n_categories, n_dims)) * initial_mag
    model = NaiveBayesModel(w_init=w_init)
    train_nll_op = model.nll(x_minibatch)
    test_nll_op = model.nll(x_test)
    normal_lp_op = model.log_p_x(normal)
    frankenstein_lp_op = model.log_p_x(frankenstein)

    train_op = optimizer.minimize(train_nll_op, global_step=global_step)
    train_nll_summary_op = tf.summary.scalar('train_nll', - train_nll_op)
    test_nll_summary_op = tf.summary.scalar('test_nll', - test_nll_op)

    # Sampling
    samples_op = model.sample(plot_n_samples)
    all_samples_op = model.sample_all_z()

    with tf.Session() as session:

        # logging
        train_log_writer = init_summary_writer(session, './summaries/naivebayes/train')
        test_log_writer = init_summary_writer(session, './summaries/naivebayes/test')

        session.run(train_iterator.initializer)
        session.run(tf.global_variables_initializer())

        n_steps = (n_epochs * n_samples) // minibatch_size
        for i in range(n_steps):

            # Test and plot
            if i % test_every == 0:
                samples, all_samples, test_loss, test_summary = session.run(
                    [samples_op, all_samples_op, test_nll_op, test_nll_summary_op])
                print('{}/{}: Test NLL: {}'.format(i, n_steps, test_loss))

                plot(samples, 'Samples at {}.'.format(i), fname='most_likely_samples_{}'.format(i))
                plot(all_samples, 'Each Z at {}'.format(i), fname='all_z_samples_{}'.format(i))

                test_log_writer.add_summary(test_summary, i)

            # Train
            t = time.time()
            _, train_loss, train_summary = session.run([train_op, train_nll_op, train_nll_summary_op])
            print('{}/{}: Train NLL: {} in {}s'.format(i, n_steps, train_loss, time.time() - t))
            train_log_writer.add_summary(train_summary, i)

        train_log_writer.close()
        test_log_writer.close()

        # Perform analysis on normal and Frankenstein digits.
        lp_normal, lp_frankenstein = session.run([normal_lp_op, frankenstein_lp_op])
        normal = np.reshape(normal, (-1, 28, 28))
        frankenstein = np.reshape(frankenstein, (-1, 28, 28))
        plot(normal, 'Normal MNIST Evaluation', fname='normal_imgs', probs=lp_normal)
        plot(frankenstein, 'Frankenstein Evaluation', fname='frankenstein_imgs', probs=lp_frankenstein)


if __name__ == '__main__':
    train_simple_generative_model_on_mnist()
