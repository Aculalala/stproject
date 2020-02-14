import numpy as np


def data_gen(Nk=10, K=4, p=2000, d=3):
    X = np.random.randn(Nk * K, p)
    Y = np.arange(K).repeat(Nk)
    for i in range(K):
        X[i * Nk:(i + 1) * Nk, 0] += d * np.cos((i / K - 0.125) * np.pi * 2)
        X[i * Nk:(i + 1) * Nk, 1] += d * np.sin((i / K - 0.125) * np.pi * 2)
    return X, Y
