import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Self


class Perceptron:
    """Classifier - perceptron
    Parameters:
    eta - learning rate, [0, 1]
    epochs - how many training iterations are performed
    random_seed - seed for random generator
    """

    ROW_IDX = 0
    COLUMNS_IDX = 1

    def __init__(self, learning_rate: float = 0.01, epochs: int = 50, random_seed: int = 1) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.w = None
        self.e = None

    def fit(self, X: NDArray, y: ArrayLike) -> Self:
        """Fitting training data"""
        random_generator = np.random.RandomState(self.random_seed)
        # Initializing weights vector (size m+1 where m is number of features); scale is standard deviation
        self.w = random_generator.normal(loc=0.0, scale=0.01, size=(1 + X.shape[self.COLUMNS_IDX]))
        self.e = []

        for _ in range(self.epochs):
            e = 0
            # Iterating through every training sample row after row
            for x_i, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(x_i))
                self.w[1:] += update * x_i  # Scaling update factor by an argument value
                self.w[0] += update
                e += 1 if update != 0.0 else 0
            self.e.append(e)
        return self

    def __net_input(self, X: NDArray) -> ArrayLike:
        """Calculating overall input z, unit bias included"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X: NDArray) -> NDArray:
        """Returning classification afer calculating Heavyside function"""
        return np.where(self.__net_input(X) >= 0.0, 1, -1)
