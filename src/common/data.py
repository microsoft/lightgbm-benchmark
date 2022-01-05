import numpy as np

# derived from https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/datasets/_samples_generator.py#L506
# for generating data in batches
class RegressionDataGenerator():
    def __init__(self,
                 batch_size: int,
                 n_features: int,
                 n_informative: int,
                 n_targets: int,
                 bias: float,
                 noise: float,
                 seed: int):
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
        # Randomly generate a well conditioned input set
        X = self.generator.randn(self.batch_size, self.n_features)

        y = np.dot(X, self.ground_truth) + self.bias

        # Add noise
        if self.noise > 0.0:
            y += self.generator.normal(scale=self.noise, size=y.shape)

        y = np.squeeze(y)

        return X, y
