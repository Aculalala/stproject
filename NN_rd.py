from sklearn.neighbors import NearestNeighbors

from data_gen import data_gen_REAL


def nearest_neighbors(values, all_values, nbr_neighbors=1):
    nn = NearestNeighbors(nbr_neighbors, algorithm='brute').fit(all_values)
    dists, idxs = nn.kneighbors(values)
    return idxs


c = 0
for i in range(40):
    X_TR, Y_TR, X_TE, Y_TE = data_gen_REAL(i)
    if (Y_TR[nearest_neighbors(X_TE, X_TR)[0][0]] == Y_TE[0]):
        c = c + 1

print(c)
