import numpy as np
from swem import SWEM

embedding_dim = 300
data_size = 1000

X = [np.random.randn(np.random.randint(10, 100), embedding_dim)
     for i in range(data_size)]

Y = .2 * np.random.randn(data_size) + .5

m = SWEM()

m.train(X, Y, plotfile='../img/test_training.png')
