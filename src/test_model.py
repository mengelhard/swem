import numpy as np
import pandas as pd
from swem import SWEM
import tensorflow as tf


def random_noise_test():

    embedding_dim = 300
    data_size = 1000

    X = [np.random.randn(np.random.randint(10, 100), embedding_dim)
         for i in range(data_size)]

    Y = .2 * np.random.randn(data_size) + .5

    m = SWEM(embedding_dimension=embedding_dim)

    m.train(X, Y, plotfile='../img/test_training.png')


def mturk_test():

    embedding_dim = 300

    Y = get_mturk_outcomes()
    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    X = np.load('../data/mturk_embedded.npz')

    # X0 = X['arr_0']
    # X1 = X['arr_1']
    X2 = X['arr_2']
    # X3 = X['arr_3']

    m = SWEM(
        embedding_dimension=embedding_dim,
        num_outputs=2,
        learning_rate=1e-4,
        activation_fn=tf.nn.elu,
        embedding_mlp_depth=2,
        prediction_mlp_layers=(120, 24))

    # m.train(X0, Y, plotfile='../img/X0_training.png')
    m.train(X2, Y[:, :2], plotfile='../img/X2_Y01_training.png',
            batch_size=100,
            epochs=20)  # m.train(X2, Y, plotfile='../img/X2_training.png')
    # m.train(X3, Y, plotfile='../img/X3_training.png',
    #         batch_size=500,
    #         epochs=10)


def get_mturk_outcomes():

    df = pd.read_csv('../data/cleaned_data.csv')

    phq_cols = ['phq:1', 'phq:2', 'phq:3', 'phq:4', 'phq:5',
                'phq:6', 'phq:7', 'phq:8', 'phq:9']

    gad_cols = ['gad:1', 'gad:2', 'gad:3', 'gad:4', 'gad:5',
                'gad:6', 'gad:7']

    ina_cols = ['swan:1', 'swan:2', 'swan:3', 'swan:4', 'swan:5',
                'swan:6', 'swan:7', 'swan:8', 'swan:9']

    hyp_cols = ['swan:10', 'swan:11', 'swan:12', 'swan:13', 'swan:14',
                'swan:15', 'swan:16', 'swan:17', 'swan:18']

    df['phq'] = df[phq_cols].sum(axis=1)
    df['gad'] = df[gad_cols].sum(axis=1)
    df['ina'] = df[ina_cols].sum(axis=1)
    df['hyp'] = df[hyp_cols].sum(axis=1)

    Y = df[['phq', 'gad', 'ina', 'hyp']].values

    assert Y.ndim == 2
    assert Y.shape[1] == 4

    return Y


if __name__ == '__main__':
    mturk_test()

# better way to do this: pass session to train and predict methods
