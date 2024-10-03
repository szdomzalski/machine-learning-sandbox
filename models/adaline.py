import numpy as np
from numpy.typing import NDArray
from typing import Self

from . import perceptron


class Adaline(perceptron.Perceptron):
    '''Classifier - ADAptive LInear NEuron
    Parameters:
    eta - learning rate, [0, 1]
    epochs - how many training iterations are performed
    random_seed - seed for random generator
    '''

    def __init__(self, learning_rate: float = 0.01, epochs: int = 50, random_seed: int = 1) -> None:
        super().__init__(learning_rate, epochs, random_seed)
        self.cost = None

    def fit(self, X: NDArray, y: NDArray) -> Self:
        '''Fitting model using training data
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        y - Nx1 vector where there is N samples and 1 output trait
        '''
        random_generator = np.random.RandomState(self.random_seed)
        # Initializing weights vector (size m+1 where m is number of features); scale is standard deviation
        self.w = random_generator.normal(loc=0.0, scale=0.01, size=(1 + X.shape[self.COLUMNS_IDX]))
        self.cost = []

        for _ in range(self.epochs):
            # Scalar product of matrix X and vector w, result is a vector of size Nx1
            net_input = self._net_input(X)
            # Activation function is just scaling, activation output and error are vectors of size Nx1
            activation_output = self._activation(net_input)
            e = np.array(y - activation_output, dtype=np.float64)  # Somehow dtype was changed if simply subtracted
            # Vector w will be column vector of size Mx1
            self.w[1:] += self.learning_rate * X.T.dot(e)
            # Sum of errors is simply used as bias is technically ones vector
            self.w[0] += self.learning_rate * e.sum()

            # This is tricky, numpy vector e to the power of 2 is element-wise operation, [e1**2 e2**2 ... eN**2]
            cost = (e**2).sum() / 2.0
            self.cost.append(cost)

        return self

    def _activation(self, x: NDArray) -> NDArray:
        '''Calculating linear activation function'''
        return x
