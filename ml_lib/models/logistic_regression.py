import numpy as np
from numpy.typing import NDArray
from typing import Self

from .adaline import AdalineGD


class LogisticRegressionGD(AdalineGD):
    '''Classifier - logistic regression (gradient descent method)
    Parameters:
    eta - learning rate, [0, 1]
    epochs - how many training iterations are performed
    random_seed - seed for random generator
    '''

    # Changed default parameters only in __init__
    def __init__(self, eta: float = 0.05, epochs: int = 100, random_seed: int = 1) -> None:
        super().__init__(eta, epochs, random_seed)

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
            self.w[1:] += self.eta * X.T.dot(e)
            # Sum of errors is simply used as bias is technically ones vector
            self.w[0] += self.eta * e.sum()

            # This time cost is a logistic cost, not a sum of squared errors
            cost = -(y.dot(np.log(activation_output)) + (1 - y).dot(np.log(1 - activation_output)))
            self.cost.append(cost)

        return self

    def _activation(self, z: NDArray) -> NDArray:
        '''Calculating logistic, sigmoid activation function'''
        # Clipping z is to limit numerical calculations - calculated activation should be close to 0 anyways
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X: NDArray) -> NDArray:
        """Returning classification afer calculating non-negative step function
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        Return:
        y - Nx1 vector where there is N samples and 1 output trait"""
        # return np.where(self._activation(self._net_input(X)) >= 0.5, 1, 0)
        # Below is equivalent which requires less calculation as z(0) = 0.5
        return np.where(self._net_input(X) >= 0.0, 1, 0)
