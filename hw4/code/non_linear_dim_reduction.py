import numpy as np

from tqdm import tqdm

from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph import graph_shortest_path


# %%
def derivative(y, pi):
    y_i = np.tile(y, (y.shape[0], 1, 1))
    y_j = np.swapaxes(y_i, 0, 1)
    y_delta = y_j - y_i
    norm_y_delta = np.linalg.norm(y_delta, axis=2)
    M = np.ones_like(pi) - np.divide(pi, norm_y_delta,
                                     out=np.zeros_like(pi),
                                     where=norm_y_delta != 0)
    out = np.zeros(y.shape)
    for i in range(y.shape[0]):
        out[i] = np.matmul(M[i], y_delta[i])

    return out


def non_linear_dim_reduction(data, n_neighbors=100, alpha=0.0001, dim=2):
    G = kneighbors_graph(X=data, n_neighbors=n_neighbors,
                         mode='distance',
                         metric='euclidean',
                         include_self=True).toarray()
    pi = graph_shortest_path(G)
    y = np.random.rand(data.shape[0], dim)

    for _ in tqdm(range(200), leave=False, desc='Progress'):
        y -= alpha * derivative(y, pi)

    return y


