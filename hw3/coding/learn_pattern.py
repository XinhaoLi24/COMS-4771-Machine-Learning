import time

from scipy.io import loadmat
import matplotlib.pyplot as plt

from neural_network_gd_adam import Network, Adam, Layer

# %% Read Data
data = loadmat('nn_data.mat')
X1, X2, Y1, Y2 = data['X1'], data['X2'], data['Y1'], data['Y2']

# %% Data 1
# Show
img1 = Y1.reshape(100, 76)
plt.imshow(img1)
plt.savefig('img1.pdf', dpi=300)
plt.show()

# Standardize
x_train = X1
y_train = Y1
x_train /= max(X1[:, 0])
y_train /= max(Y1[:, 0])

# Train
epochs = 500
lr = 0.01
batch_size = 64

model1 = Network()
model1.add(Layer(2, activation='tanh'))
model1.add(Layer(128, activation='tanh'))
model1.add(Layer(128, activation='tanh'))
model1.add(Layer(256, activation='tanh'))
# model.add(Layer(64, activation='tanh'))
model1.add(Layer(1, activation='tanh'))

# If optimizer = None, then the standard gradient descent will be used.
adam = Adam(lr)
model1.set_config(epochs=epochs, learning_rate=lr, optimizer=adam)

# Predict
start_time = time.time()
y_pred = model1.fit(x_train, y_train, x_train, y_train, batch_size=batch_size)
img1 = y_pred.reshape(100, 76)
plt.imshow(img1)
plt.savefig('5-128-128-256-500-64.pdf', dpi=300)
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
# %% Data 2
# # Show
img2 = Y2.reshape(133, 140, 3) / 256
plt.imshow(img2)
plt.savefig('img2.pdf', dpi=300)
plt.show()
#
# Standardize
x_train = X2
y_train = Y2
x_train /= max(X2[:, 0])
y_train /= max(Y2[:, 0])

# Train
epochs = 1000
lr = 0.01
batch_size = 64

model2 = Network()
model2.add(Layer(2, activation='tanh'))
model2.add(Layer(128, activation='tanh'))
model2.add(Layer(128, activation='tanh'))
# model2.add(Layer(128, activation='tanh'))
# model2.add(Layer(256, activation='tanh'))
model2.add(Layer(3, activation='tanh'))

adam = Adam(lr)
model2.set_config(epochs=epochs, learning_rate=lr, optimizer=adam)

# Predict
start_time = time.time()
y_pred = model2.fit(x_train, y_train, x_train, y_train, batch_size=batch_size)

print("--- %s seconds ---" % (time.time() - start_time))

img2 = y_pred.reshape(133, 140, 3) * max(Y2[:, 0])
plt.imshow(img2)
# plt.savefig('4-128-128-1000-64.pdf', dpi=300)
plt.show()