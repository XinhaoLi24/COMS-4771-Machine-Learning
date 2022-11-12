import pandas as pd
import numpy as np


# %%
class NaiveBayesClassifier(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_set = None
        self.train_size = None
        self.target = None
        self.prior = {}
        self.features = list
        self.means = {}
        self.vars = {}
        self.count = None
        self.classes = None

    def calculate_class_prior(self):
        self.prior = ((self.train_set.groupby(self.target).count() / len(
            self.train_set)).iloc[:, 1]).to_numpy()
        return self.prior

    def calculate_mean_var(self):
        self.means = self.train_set.groupby([self.target]).mean().to_numpy()
        self.vars = self.train_set.groupby([self.target]).var().to_numpy()
        return self.means, self.vars

    def gaussian_distribution(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        probability = numerator / denominator
        return probability

    def calculate_posterior(self, x):
        posteriors = []
        for i in range(self.count):
            prior = np.log(self.prior[i])
            conditional = np.sum(np.log(self.gaussian_distribution(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.train_set = pd.concat([X_train, y_train], axis=1, join='inner')
        self.target = y_train.columns.values[0]
        self.train_size = X_train.shape[0]
        self.features = list(X_train.columns)
        self.classes = np.unique(self.y_train)
        self.count = len(self.classes)

        self.calculate_class_prior()
        self.calculate_mean_var()

    def predict(self, X_test):
        predictions = [self.calculate_posterior(X_test.iloc[i])
                       for i in range(len(X_test))]
        return predictions


class KNN(object):
    def __init__(self,
                 k,
                 distance_metric):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    @staticmethod
    def vote_common(predictions):
        return max(set(predictions), key=predictions.count)

    @staticmethod
    def manhattan(new_point, data):
        return np.sum(abs(new_point - data), axis=1)

    @staticmethod
    def euclidean(new_point, data):
        return np.sqrt(np.sum((new_point - data)**2, axis=1))

    @staticmethod
    def max_distance(new_point, data):
        return abs((new_point - data).max(axis=1))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train['spam'].values.tolist()

    def predict(self, X_test):
        if self.distance_metric == "euclidean":
            self.distance_metric = self.euclidean
        elif self.distance_metric == "manhattan":
            self.distance_metric = self.manhattan
        elif self.distance_metric == "max_distance":
            self.distance_metric = self.max_distance

        neighbors = []
        for i in range(len(X_test)):
            distances = self.distance_metric(X_test.iloc[i], self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        results = list(map(self.vote_common, neighbors))
        return results
