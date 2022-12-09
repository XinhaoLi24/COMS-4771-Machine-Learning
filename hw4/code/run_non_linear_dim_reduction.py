import matplotlib.pyplot as plt

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

from non_linear_dim_reduction import *

# %%
srh = np.loadtxt('swiss_roll_hole.txt')
sr = np.loadtxt('swiss_roll.txt')

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot3D(sr[:, 0], sr[:, 1], sr[:, 2], '.')

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot3D(srh[:, 0], srh[:, 1], srh[:, 2], '.')
plt.savefig('3D-swiss-role.pdf', dpi=300)
plt.show()

# %%
sr_embed = non_linear_dim_reduction(sr, n_neighbors=1000)

#%%
plt.scatter(sr_embed[:, 0], sr_embed[:, 1])
plt.savefig('nldr-1000-200.pdf', dpi=300)
plt.show()

# %% Compare with sklearn
# %%
isomap = Isomap(n_neighbors=200, n_components=2)
pca = PCA(n_components=2)
sr_pca = pca.fit_transform(sr)
sr_isomap = isomap.fit_transform(sr)

# %%
figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[0].scatter(-sr_pca[:, 0], sr_pca[:, 1])
ax[1].scatter(sr_isomap[:, 0], sr_isomap[:, 1])

ax[0].set_title('PCA-swiss-roll')
ax[1].set_title('Isomap-swiss-roll')
plt.tight_layout()
plt.savefig('sklearn-swiss-role-test-200.pdf', dpi=300)
plt.show()

# %%
isomap = Isomap(n_neighbors=200, n_components=2)
pca = PCA(n_components=2)
sr_pca = pca.fit_transform(srh)
sr_isomap = isomap.fit_transform(srh)

# %%
figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[0].scatter(-sr_pca[:, 0], sr_pca[:, 1])
ax[1].scatter(sr_isomap[:, 0], sr_isomap[:, 1])

ax[0].set_title('PCA-swiss-roll-hole')
ax[1].set_title('Isomap-swiss-roll-hole')
plt.tight_layout()
plt.savefig('sklearn-swiss-role-hole-test-200.pdf', dpi=300)
plt.show()
