import numpy as np

from sklearn.utils import shuffle


# %%
class Layer:
    """
    Define the layer class.
    """
    def __init__(self, hidden_units: int, activation: str = None):
        """

        :param hidden_units:
        :param activation:
        """
        self.A = None
        self.E = None
        self.input = None
        self.hidden_units = hidden_units
        self.activation = activation
        self.W = None
        self.b = None

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid.
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        """
        Derivative of sigmoid.
        :param x:
        :return:
        """
        return Layer.sigmoid(x) * (1 - Layer.sigmoid(x))

    @staticmethod
    def tanh(x):
        """
        tanh
        :param x:
        :return:
        """
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        """
        Derivative of tanh.
        :param x:
        :return:
        """
        return 1 - np.tanh(x) ** 2

    def initialize_params(self, n_in, hidden_units):
        """
        Initialize W and b.
        :param n_in:
        :param hidden_units:
        :return:
        """
        np.random.seed(42)
        self.W = np.random.randn(n_in, hidden_units) * np.sqrt(2 / n_in)
        np.random.seed(42)
        self.b = np.zeros((1, hidden_units))

    def forward(self, X):
        """
        Forward propagation
        :param X:
        :return:
        """
        self.input = np.array(X, copy=True)
        if self.W is None:
            self.initialize_params(self.input.shape[-1], self.hidden_units)
        self.E = X @ self.W + self.b

        if self.activation is not None:
            self.A = self.activation_fn(self.E)
            return self.A
        return self.E

    def activation_fn(self, x, derivative=False):
        """
        Activation function.
        :param x:
        :param derivative:
        :return:
        """
        if self.activation == 'sigmoid':
            if derivative:
                return self.dsigmoid(x)
            return self.sigmoid(x)
        if self.activation == 'tanh':
            if derivative:
                return self.dtanh(x)
            return self.tanh(x)


class Network:
    """
    Define the network class.
    """
    def __init__(self):
        self.learning_rate = None
        self.optimizer = None  # if None, then gradient descent will be used
        self.epochs = None
        self.layers = dict()
        self.parameters = dict()
        self.grads = dict()

    def add(self, layer):
        """
        Add layers into the network.
        :param layer:
        :return:
        """
        self.layers[len(self.layers) + 1] = layer

    def set_config(self, epochs, learning_rate, optimizer=None):
        """
        Config the network.
        :param epochs:
        :param learning_rate:
        :param optimizer:
        :return:
        """
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if not not self.optimizer:
            self.optimizer.config(self.layers)
            self.optimizer.epochs = self.epochs
            self.optimizer.learning_rate = self.learning_rate

    def forward(self, x):
        """
        Forward propagation.
        :param x:
        :return:
        """
        for idx, layer in self.layers.items():
            x = layer.forward(x)
            self.parameters[f'W{idx}'] = layer.W
            self.parameters[f'E{idx}'] = layer.E
            self.parameters[f'A{idx}'] = layer.A
        return x

    def backward(self, y):
        """
        Backward propagation.
        :param y:
        :return:
        """
        last_layer_idx = max(self.layers.keys())
        m = y.shape[0]
        # back prop through all dZs
        for idx in reversed(range(1, last_layer_idx + 1)):
            if idx == last_layer_idx:
                # e.g. dZ3 = y_pred - y_true for a 3 layer network
                self.grads[f'dE{idx}'] = self.parameters[f'A{idx}'] - y
            else:
                # dZn = dZ(n+1) dot W(n+1) * inverse derivative of
                # activation function of Layer n, with Zn as input
                self.grads[f'dE{idx}'] = self.grads[f'dE{idx + 1}'] @ \
                                         self.parameters[f'W{idx + 1}'].T * \
                                         self.layers[idx].activation_fn(
                                             self.parameters[f'E{idx}'],
                                             derivative=True)
            self.grads[f'dW{idx}'] = 1 / m * self.layers[idx].input.T @ \
                                     self.grads[f'dE{idx}']
            self.grads[f'db{idx}'] = 1 / m * np.sum(self.grads[f'dE{idx}'],
                                                    axis=0, keepdims=True)

            assert self.grads[f'dW{idx}'].shape == self.parameters[
                f'W{idx}'].shape

    def update_params(self, steps):
        """
        Update parameters.
        :param steps:
        :return:
        """
        for idx in self.layers.keys():
            if self.optimizer is None:
                self.optimize(idx)
            else:
                self.optimizer.optimize(idx, self.layers, self.grads, steps)

    def optimize(self, idx):
        """

        :param idx:
        :return:
        """
        # Vanilla minibatch gradient descent
        self.layers[idx].W -= self.learning_rate * self.grads[f'dW{idx}']
        self.layers[idx].b -= self.learning_rate * self.grads[f'db{idx}']

    def fit(self, x_train, y_train, x_test=None, y_test=None, batch_size=32):
        global test_preds
        losses = []
        for i in range(1, self.epochs + 1):
            if i % 100 == 0:
                print(f'Epoch {i}')
            batches = self.create_batches(x_train, y_train, batch_size)
            epoch_loss = []
            steps = 0

            for x, y in batches:
                steps += 1
                preds = self.forward(x)
                loss = self.compute_loss(y, preds)
                epoch_loss.append(loss)

                # Backward propagation - calculation of gradients
                self.backward(y)

                # update weights and biases of each layer
                self.update_params(steps)

            loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(loss)

            # Predict with network on x_train
            train_preds = self.forward(x_train)

            # Predict with network on x_test
            test_preds = self.forward(x_test)

        return test_preds

    @staticmethod
    def compute_loss(Y, Y_hat):
        return np.mean((Y - Y_hat) ** 2)

    @staticmethod
    def create_batches(x, y, batch_size):
        x, y = shuffle(x, y)
        m = x.shape[0]
        num_batches = m / batch_size
        batches = []
        for i in range(int(num_batches + 1)):
            batch_x = x[i * batch_size:(i + 1) * batch_size]
            batch_y = y[i * batch_size:(i + 1) * batch_size]
            batches.append((batch_x, batch_y))

        # Check divisible
        if m % batch_size == 0:
            batches.pop(-1)

        return batches


class Adam(object):
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999,
                 epsilon=1e-8):
        """
        Define parameters.
        :param learning_rate:
        :param beta1:
        :param beta2:
        :param epsilon:
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()

    def config(self, layers):
        """
        Initialize parameters needed to be updated.
        :param layers:
        :return:
        """
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0
            self.v[f'W{i}'] = 0
            self.v[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, steps):
        """
        OPtimization.
        :param idx:
        :param layers:
        :param grads:
        :param steps:
        :return:
        """
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']

        # weights
        self.m[f'W{idx}'] = self.beta1 * self.m[f'W{idx}'] + (
                1 - self.beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (
                1 - self.beta2) * dW ** 2

        # biases
        self.m[f'b{idx}'] = self.beta1 * self.m[f'b{idx}'] + (
                1 - self.beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (
                1 - self.beta2) * db ** 2

        # take timestep into account
        mt_w = self.m[f'W{idx}'] / (1 - self.beta1 ** steps)
        vt_w = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b = self.m[f'b{idx}'] / (1 - self.beta1 ** steps)
        vt_b = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        w_update = - self.learning_rate * mt_w / (
                np.sqrt(vt_w) + self.epsilon)
        b_update = - self.learning_rate * mt_b / (
                np.sqrt(vt_b) + self.epsilon)

        layers[idx].W += w_update
        layers[idx].b += b_update
