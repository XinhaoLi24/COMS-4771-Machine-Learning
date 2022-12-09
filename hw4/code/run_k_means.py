import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import *

from k_means import *


# %%
rcparams = {"font.size": 16,
            "legend.frameon": True,
            "xtick.top": False,
            "xtick.direction": "in",
            "xtick.minor.visible": True,
            "xtick.major.size": 10,
            "xtick.minor.size": 6,
            "ytick.right": False,
            "ytick.direction": "in",
            "ytick.minor.visible": True,
            "ytick.major.size": 10,
            "ytick.minor.size": 6}


def plot(X, Y, k_means_clusters, k_means_centers, save=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, ax=ax[0], alpha=0.5)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=k_means_clusters, ax=ax[1],
                    alpha=0.5)

    sns.scatterplot(x=k_means_centers[:, 0], y=k_means_centers[:, 1],
                    color='red', alpha=1, ax=ax[1], s=200, marker='x',
                    label='K means centers')

    ax[0].set_title('Ture Clusters')
    ax[1].set_title('Predicted Clusters using k means')

    # ax[0].legend(loc="upper right")
    # ax[1].legend(loc="upper right")

    plt.rcParams.update(rcparams)
    plt.figure(figsize=(6, 4))
    plt.tight_layout()
    # if save is not None:
    fig.savefig('{}.pdf'.format(str(save)), dpi=300)
    plt.show()


# %%
# load data
X1, Y1 = make_circles(n_samples=500, random_state=64)
#
# run k-means
k_means_clusters, k_means_centers = kmeans(X=X1, k=2, iters=50)

plot(X1, Y1, k_means_clusters, k_means_centers, save='circle-1')

# %%
# load data
X2, Y2 = make_moons(n_samples=500, random_state=64)

# run k-means
k_means_clusters, k_means_centers = kmeans(X=X2, k=2, iters=50)

plot(X2, Y2, k_means_clusters, k_means_centers, save='moon-1')

# %%
# load data
x_small, y_small = make_circles(n_samples=500, random_state=3, factor=0.7)
x_large, y_large = make_circles(n_samples=500, random_state=3, factor=0.4)

y_large[y_large == 1] = 2

df = pd.DataFrame(np.vstack([x_small,x_large]),columns=['x1','x2'])
df['label'] = np.hstack([y_small, y_large])

X3 = np.array(df[['x1', 'x2']])
Y3 = df.label

# run k-means
k_means_clusters, k_means_centers = kmeans(X=X3, k=3, iters=100)

# Plot
plot(X3, Y3, k_means_clusters, k_means_centers, save='circle-2')

# %%
n_neighbors = 2

# load data
X1, Y1 = make_circles(n_samples=500, random_state=64)
#
# run k-means
k_means_clusters, k_means_centers = flexible_kmeans(X=X1, k=2, iters=50,
                                                    n_neighbors=n_neighbors)

plot(X1, Y1, k_means_clusters, k_means_centers, save='n-circle-1')

# %%
# load data
X2, Y2 = make_moons(n_samples=500, random_state=64)

# run k-means
k_means_clusters, k_means_centers = flexible_kmeans(X=X2, k=2, iters=50,
                                                    n_neighbors=n_neighbors)

plot(X2, Y2, k_means_clusters, k_means_centers, save='n3-moon-1')

# %%
# load data
x_small, y_small = make_circles(n_samples=500, random_state=3, factor=0.7)
x_large, y_large = make_circles(n_samples=500, random_state=3, factor=0.4)

y_large[y_large == 1] = 2

df = pd.DataFrame(np.vstack([x_small, x_large]), columns=['x1','x2'])
df['label'] = np.hstack([y_small, y_large])

X3 = np.array(df[['x1', 'x2']])
Y3 = df.label

# run k-means
k_means_clusters, k_means_centers = flexible_kmeans(X=X3, k=3, iters=100,
                                                    n_neighbors=n_neighbors)

# Plot
plot(X3, Y3, k_means_clusters, k_means_centers, save='n3-circle-2')