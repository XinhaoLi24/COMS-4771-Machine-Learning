import random
import numpy as np

import scipy
from scipy.spatial.distance import cdist

from sklearn.neighbors import kneighbors_graph


def kmeans(X, k, iters=10):
    """

    :param X:
    :param k:
    :param iters:
    :return:
    """

    # Number of samples
    m = X.shape[0]

    if k < m + 1:
        # Without replacement
        idx = random.sample(population=range(m), k=k)
    else:
        # With replacement
        idx = random.choice(population=range(m), k=k)

    means = X[idx]

    # Keep track of number of iterations passed
    n_iters = 0

    while n_iters < iters:
        # Euclidean distances between data points with current means
        distances = cdist(X, means, 'euclidean')

        # Assign each data point to the closest cluster
        clusters = np.array([np.argmin(i) for i in distances])

        # Update the cluster means
        new_means = []
        for c in range(k):
            new_means.append(X[clusters == c].mean(axis=0))
        new_means = np.vstack(new_means)

        if np.array_equal(new_means, means):
            means = new_means
            break
        else:
            means = new_means
            n_iters += 1

        if n_iters == iters:
            print('Maximum number of iterations reached')

    distances = cdist(X, means, 'euclidean')
    clusters = np.array([np.argmin(i) for i in distances])

    return clusters, means


def flexible_kmeans(X, k, n_neighbors, iters=10):
    """

    :param X:
    :param k:
    :param n_neighbors:
    :param iters:
    :return:
    """
    n_samples = X.shape[0]

    G = kneighbors_graph(X=X, n_neighbors=n_neighbors, metric='euclidean',
                         include_self=True).toarray()

    W = np.eye(n_samples)

    for i in range(n_samples):
        # Not include ii
        for j in range(i + 1, n_samples):
            if G[i, j] == 1 or G[j, i] == 1:
                W[i, j] = 1
                W[j, i] = 1

    D = np.diag(W.sum(axis=1))

    L = D - W

    assert ((L == L.T).all())

    eigenvalues, eigenvectors = scipy.linalg.eigh(L)

    # Bottom k eigenvalues
    idx = range(n_samples)
    sortedIdx = list(
        list(zip(*sorted(zip(idx, eigenvalues.tolist()),
                         key=lambda x: x[1])))[0])

    V = eigenvectors[:, sortedIdx][:, 0:k]

    clusters, means = kmeans(X=V, k=k, iters=iters)

    return clusters, means
