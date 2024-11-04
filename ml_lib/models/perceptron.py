import numpy as np
from numpy.typing import NDArray
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

    def __init__(self, eta: float = 0.01, epochs: int = 50, random_seed: int = 1) -> None:
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        self.w = None
        self.number_of_e = None

    def fit(self, X: NDArray, y: NDArray) -> Self:
        '''Fitting model using training data
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        y - Nx1 vector where there is N samples and 1 output trait
        '''
        random_generator = np.random.RandomState(self.random_seed)
        # Initializing weights vector (column type, size m+1 where m is number of features); scale is standard deviation
        self.w = random_generator.normal(loc=0.0, scale=0.01, size=(1 + X.shape[self.COLUMNS_IDX]))
        # print(f'Weights vector w shape: {{{self.w.shape}}}')
        self.number_of_e = []

        for _ in range(self.epochs):
            number_of_mismatches = 0
            # Iterating through every training sample row after row; weights are updated every sample
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w[1:] += update * x_i  # Scaling update factor by an argument value
                self.w[0] += update
                number_of_mismatches += 1 if update != 0.0 else 0
            self.number_of_e.append(number_of_mismatches)
        return self

    def _net_input(self, X: NDArray) -> NDArray:
        """Calculating overall input z, unit bias included
        Arguments:
        X - NxM vector where there is N samples and M input traits
        Return:
        z - Nx1 vector where there is N samples and 1 output trait"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X: NDArray) -> NDArray:
        """Returning classification afer calculating Heavyside function
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        Return:
        y - Nx1 vector where there is N samples and 1 output trait"""
        return np.where(self._net_input(X) >= 0.0, 1, -1)
