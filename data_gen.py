import numpy as np


def data_gen(Nk=10, K=4, p=2000, d=4, seed=None):
    if seed != None:
        np.random.seed(seed % (2 ** 31))
    X = np.random.randn(Nk * K, p)
    Y = np.arange(K).repeat(Nk)
    scale = np.sqrt(1 + 0.5 * (d ** 2))  # scale for first two dimensions
    for i in range(K):
        X[i * Nk:(i + 1) * Nk, 0] += d * np.cos((i / K - 0.125) * np.pi * 2)
        X[i * Nk:(i + 1) * Nk, 1] += d * np.sin((i / K - 0.125) * np.pi * 2)
    X[:, 0] /= scale
    X[:, 1] /= scale
    # print(X)
    return X, Y


MNIST_LOADED = False
MNIST = {}


def prepare_MNIST():
    import tensorflow as tf
    (MNIST['X_TR'], MNIST['Y_TR']), (MNIST['X_TE'], MNIST['Y_TE']) = tf.keras.datasets.mnist.load_data()
    MNIST['X_TR'] = MNIST['X_TR'].reshape((-1, 28 * 28)) / 127.5 - 1
    MNIST['X_TE'] = MNIST['X_TE'].reshape((-1, 28 * 28)) / 127.5 - 1


def data_gen_MNIST(N, train=True):
    if MNIST_LOADED == False:
        prepare_MNIST()
    if train:
        idx = np.random.choice(MNIST['X_TR'].shape[0], size=N, replace=False, p=None)
        return MNIST['X_TR'][idx], MNIST['Y_TR'][idx],
    else:
        idx = np.random.choice(MNIST['X_TE'].shape[0], size=N, replace=False, p=None)
        return MNIST['X_TE'][idx], MNIST['Y_TE'][idx],

# print(data_gen_MNIST(1))
