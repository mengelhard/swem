
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from vis_data import TrainingSummary


class SWEM:

    def __init__(self, embedding_dimension=300, num_outputs=1,
                 embedding_mlp_depth=3, prediction_mlp_layers=(300, 300, 300),
                 classifier=False, alpha=0., learning_rate=1e-4,
                 max_sentence_length=200, activation_fn=tf.nn.relu):

        self.embedding_dimension = embedding_dimension
        self.num_outputs = num_outputs
        self.embedding_mlp_depth = embedding_mlp_depth
        self.prediction_mlp_layers = prediction_mlp_layers
        self.classifier = classifier
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_sentence_length = max_sentence_length
        self.activation_fn = activation_fn

        self._build_model()
        self._get_train_step()

    def train(self, X, Y, val_prc=.2, batch_size=100, epochs=10,
              shuffle=True, truncate_epochs=True,
              progress_bar=True, plot_training=True,
              plotfile=None):

        assert np.shape(X[0])[1] == self.embedding_dimension
        assert len(X) == len(Y)

        X, X_mask = process_sentences(X, self.max_sentence_length)

        if np.ndim(Y) == 1:
            Y = Y[:, np.newaxis]

        tX, tX_mask, tY, vX, vX_mask, vY = split_data(
            X, X_mask, Y, split_prc=val_prc)

        train_len = len(tX)
        num_batches = train_len // batch_size

        if not truncate_epochs:
            num_batches += int(train_len % batch_size > 0)

        ts = TrainingSummary(['train_loss', 'val_loss'])

        progress = range(epochs * num_batches)

        if progress_bar:
            progress = tqdm(progress)

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())

            for train_idx in progress:

                batch_idx = train_idx % num_batches

                if shuffle and (batch_idx == 0):

                    tX, tX_mask, tY = shuffle_data(tX, tX_mask, tY)

                first = batch_idx * batch_size
                last = min((batch_idx + 1) * batch_size, train_len)

                loss_, _ = s.run(
                    [self._loss, self._train_step],
                    feed_dict={self._input_sentences: tX[first:last],
                               self._input_masks: tX_mask[first:last],
                               self._labels: tY[first:last]})

                ts.add_point('train_loss', [train_idx, loss_])

                if progress_bar:
                    progress.set_description("loss=%.3f" % (loss_))

                if batch_idx == num_batches - 1:

                    vloss_ = s.run(
                        self._loss,
                        feed_dict={self._input_sentences: vX,
                                   self._input_masks: vX_mask,
                                   self._labels: vY})

                    ts.add_point('val_loss', [train_idx, vloss_])

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ts.plot_metrics(ax, ['train_loss', 'val_loss'])

        if plotfile is not None:
            plt.savefig(plotfile)

    def predict(self, X):

        return None

    def _build_model(self):

        self._labels = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.num_outputs],
            name='labels')

        self._input_sentences = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.max_sentence_length, self.embedding_dimension],
            name='input_sentences')

        self._input_masks = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.max_sentence_length, self.embedding_dimension],
            name='input_masks')

        x = self._input_sentences

        for i in range(self.embedding_mlp_depth):

            x = tf.contrib.layers.fully_connected(
                x, self.embedding_dimension,
                activation_fn=self.activation_fn)

        m = self._input_masks

        x_avg = tf.reduce_sum(x * m, axis=1) / tf.reduce_sum(m, axis=1)
        x_max = tf.reduce_max(x * m, axis=1)

        x = tf.concat([x_avg, x_max], 1)

        for layer in self.prediction_mlp_layers:

            x = tf.contrib.layers.fully_connected(
                x, layer,
                activation_fn=self.activation_fn)

        if self.classifier:

            self._output_tensor = tf.contrib.layers.fully_connected(
                x, self.num_outputs,
                activation_fn=self.activation_fn)

        else:

            self._output_tensor = tf.contrib.layers.fully_connected(
                x, self.num_outputs,
                activation_fn=None)

    def _get_train_step(self):

        if self.classifier:

            self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                self._labels, self._output_tensor)

        else:

            self._loss = tf.losses.mean_squared_error(
                self._labels, self._output_tensor)

        self._train_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self._loss)


def split_data(*args, split_prc=.2):

    if split_prc > 0.:

        dlen = len(args[0])

        v_indices = list(np.random.permutation(
            np.arange(dlen)) < int(split_prc * dlen))

        v = []
        t = []

        for arr in args:

            v.append(np.array(
                [elt for elt, inc in zip(arr, v_indices) if inc]))
            t.append(np.array(
                [elt for elt, inc in zip(arr, v_indices) if not inc]))

    else:

        v = [None] * len(args)
        t = args

    return t + v


def shuffle_data(*args):

    z = list(zip(*args))
    random.shuffle(z)
    return zip(*z)


def process_sentences(X_list, max_length=200):

    nd = np.shape(X_list[0])[-1]
    X, X_mask = zip(*[pad(X, (max_length, nd)) for X in X_list])
    return np.array(X), np.array(X_mask)


def pad(X, shape):

    X = np.array(X)

    if (X.ndim < 2) or (X.shape[0] == 0):
        print('Warning: Empty sentence encountered')
        X = np.zeros((1, shape[1]))

    if X.shape[0] > shape[0]:
        print('Warning: Max sentence length exceeded')
        X = X[:shape[0], :]

    Z = np.zeros(shape)
    X_mask = np.zeros(shape)

    # print(Z.shape, X.shape)

    Z[:X.shape[0], :X.shape[1]] = X
    X_mask[:X.shape[0], :X.shape[1]] = 1.

    return Z, X_mask
