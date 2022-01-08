# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper code to handle/process data
"""

import numpy as np

# derived from https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/datasets/_samples_generator.py#L506
# for generating data in batches
class RegressionDataGenerator():
    """Generator for regression data"""
    def __init__(self,
                 batch_size: int,
                 n_features: int,
                 n_informative: int,
                 n_targets: int,
                 bias: float,
                 noise: float,
                 seed: int):
        """Initializes the generator"""
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_informative = min(n_features, n_informative)
        self.n_targets = n_targets
        self.bias = bias
        self.noise = noise
        self.generator = np.random.RandomState(seed)

        # Generate a ground truth model with only n_informative features being non-zeros
        self.ground_truth = np.zeros((self.n_features, self.n_targets))
        self.ground_truth[:n_informative, :] = 100 * self.generator.rand(self.n_informative, self.n_targets)

    def generate(self):
        """Generate one batch of data.

        Returns:
            X (numpy.ndarray)
            y (numpy.ndarray)
        """
        # Randomly generate a well conditioned input set
        X = self.generator.randn(self.batch_size, self.n_features)

        y = np.dot(X, self.ground_truth) + self.bias

        # Add noise
        if self.noise > 0.0:
            y += self.generator.normal(scale=self.noise, size=y.shape)

        y = np.squeeze(y)

        return X, y


class ClassificationDataGenerator():
    """Generator for regression data"""
    def __init__(self,
                 n_label_classes: int,
                 batch_size: int,
                 n_features: int,
                 n_informative: int,
                 n_targets: int,
                 bias: float,
                 noise: float,
                 seed: int):
        """Initializes the generator"""
        self.n_label_classes = n_label_classes
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_informative = min(n_features, n_informative)
        self.n_targets = n_targets
        self.bias = bias
        self.noise = noise
        self.generator = np.random.RandomState(seed)

        # Generate a ground truth model with only n_informative features being non-zeros
        self.ground_truth = np.zeros((self.n_features, self.n_targets))
        self.ground_truth[:n_informative, :] = 100 * self.generator.rand(self.n_informative, self.n_targets)

    def generate(self):
        """Generate one batch of data.

        Returns:
            X (numpy.ndarray)
            y (numpy.ndarray)
        """
        # Randomly generate a well conditioned input set
        X = self.generator.randn(self.batch_size, self.n_features)
        y = np.dot(X, self.ground_truth) + self.bias

        # Add noise
        if self.noise > 0.0:
            y += self.generator.normal(scale=self.noise, size=y.shape)

        # create n_label_classes ranking labels
        y = ((y - min(y))/(max(y)-min(y))*self.n_label_classes).astype(int)

        y = np.squeeze(y)

        return X, y


class RankingDataGenerator():
    """Generator for regression data"""
    def __init__(self,
                 docs_per_query: int,
                 n_label_classes: int,
                 batch_size: int,
                 n_features: int,
                 n_informative: int,
                 n_targets: int,
                 bias: float,
                 noise: float,
                 seed: int):
        """Initializes the generator"""
        self.docs_per_query = docs_per_query
        self.n_label_classes = n_label_classes
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_informative = min(n_features, n_informative)
        self.n_targets = n_targets
        self.bias = bias
        self.noise = noise
        self.generator = np.random.RandomState(seed)

        # Generate a ground truth model with only n_informative features being non-zeros
        self.ground_truth = np.zeros((self.n_features, self.n_targets))
        self.ground_truth[:n_informative, :] = 100 * self.generator.rand(self.n_informative, self.n_targets)

    def generate(self):
        """Generate one batch of data.

        Returns:
            X (numpy.ndarray)
            y (numpy.ndarray)
        """
        # Randomly generate a well conditioned input set
        X = self.generator.randn(self.batch_size, self.n_features)
        y = np.dot(X, self.ground_truth) + self.bias

        # add query column
        query_col = [[i // self.docs_per_query] for i in range(self.batch_size)]
        X = np.hstack((query_col, X))

        # Add noise
        if self.noise > 0.0:
            y += self.generator.normal(scale=self.noise, size=y.shape)

        # create n_label_classes ranking labels
        y = ((y - min(y))/(max(y)-min(y))*self.n_label_classes).astype(int)

        y = np.squeeze(y)

        return X, y