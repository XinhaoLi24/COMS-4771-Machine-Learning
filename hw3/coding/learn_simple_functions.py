import time

import numpy as np
import matplotlib.pyplot as plt

from neural_network_gd_adam import Network, Adam, Layer
# %% XOR
x_train = np.array([[[0, 0]], [[0, 1]],
                    [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
x_train = np.squeeze(x_train)
y_train = np.squeeze(y_train)
y_train = y_train[:, np.newaxis]

# Train
epochs = 1000
lr = 0.01
batch_size = 4

model1 = Network()
model1.add(Layer(2, activation='tanh'))
model1.add(Layer(256, activation='tanh'))
model1.add(Layer(256, activation='tanh'))
# model1.add(Layer(128, activation='tanh'))
# model.add(Layer(64, activation='tanh'))
model1.add(Layer(1, activation='tanh'))

# If optimizer = None, then the standard gradient descent will be used.
adam = Adam(lr)
model1.set_config(epochs=epochs, learning_rate=lr, optimizer=adam)

# Predict
start_time = time.time()
y_pred = model1.fit(x_train, y_train, x_train, y_train, batch_size=batch_size)
print(list(y_train))
print(list(y_pred))

print("--- %s seconds ---" % (time.time() - start_time))
# %% Data 2
x_train = np.linspace(1, 10, 500)
x_train = x_train[:, np.newaxis]

y_train = x_train[:, 0] + 1 * np.sin(x_train[:, 0]) + 1 * np.random.rand(
    x_train.shape[0])
y_train = y_train[:, np.newaxis]
x_train /= 10
y_train /= 10

# Train
epochs = 1000
lr = 0.01
batch_size = 64

model1 = Network()
model1.add(Layer(2, activation='tanh'))
model1.add(Layer(128, activation='tanh'))
# model1.add(Layer(256, activation='tanh'))
model1.add(Layer(128, activation='tanh'))
# model.add(Layer(64, activation='tanh'))
model1.add(Layer(1, activation='tanh'))

# If optimizer = None, then the standard gradient descent will be used.
adam = Adam(lr)
model1.set_config(epochs=epochs, learning_rate=lr, optimizer=adam)

# Predict
start_time = time.time()
y_pred = model1.fit(x_train, y_train, x_train, y_train, batch_size=batch_size)

#%%
plt.scatter(x_train, y_train, c='b')
plt.scatter(x_train, y_pred, c='r')
plt.savefig('simple_func.pdf', dpi=300)
plt.show()