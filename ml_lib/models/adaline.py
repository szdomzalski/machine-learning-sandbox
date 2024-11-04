import numpy as np
from numpy.typing import NDArray
from typing import Self

from . import perceptron


class AdalineGD(perceptron.Perceptron):
    '''Classifier - ADAptive LInear NEuron (gradient descent method)
    Parameters:
    eta - learning rate, [0, 1]
    epochs - how many training iterations are performed
    random_seed - seed for random generator
    '''

    def __init__(self, eta: float = 0.01, epochs: int = 50, random_seed: int = 1) -> None:
        super().__init__(eta, epochs, random_seed)
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
            self.w[1:] += self.eta * X.T.dot(e)
            # Sum of errors is simply used as bias is technically ones vector
            self.w[0] += self.eta * e.sum()

            # This is tricky, numpy vector e to the power of 2 is element-wise operation, [e1**2 e2**2 ... eN**2]
            cost = (e**2).sum() / 2.0
            self.cost.append(cost)

        return self

    def _activation(self, z: NDArray) -> NDArray:
        '''Calculating linear activation function'''
        return z

    def predict(self, X: NDArray) -> NDArray:
        """Returning classification afer calculating Heavyside function
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        Return:
        y - Nx1 vector where there is N samples and 1 output trait"""
        return np.where(self._activation(self._net_input(X)) >= 0.0, 1, -1)


class AdalineSGD(AdalineGD):
    '''Classifier - ADAptive LInear NEuron (stochastic gradient descent method)
    Parameters:
    eta - learning rate, [0, 1]
    epochs - how many training iterations are performed
    random_seed - seed for random generator
    '''

    def __init__(self, eta: float = 0.01, epochs: int = 50, random_seed: int = 1) -> None:
        super().__init__(eta, epochs, random_seed)
        self.w_initialized = False
        self.random_generator = None

    def fit(self, X: NDArray, y: NDArray) -> Self:
        '''Fitting model using training data
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        y - Nx1 vector where there is N samples and 1 output trait
        '''
        self.random_generator = np.random.RandomState(self.random_seed)
        # Initializing weights vector (size m+1 where m is number of features); scale is standard deviation
        self.w = self.random_generator.normal(loc=0.0, scale=0.01, size=(1 + X.shape[self.COLUMNS_IDX]))
        self.w_initialized = True
        self.cost = []

        for _ in range(self.epochs):
            X, y = self.__shuffle(X, y)
            training_cost = []
            for x_i, y_i in zip(X, y):
                training_cost.append(self.__update_weights(x_i, y_i))
            avg_cost = sum(training_cost) / len(training_cost)
            self.cost.append(avg_cost)
        return self

    def partial_fit(self, X: NDArray, y: NDArray) -> Self:
        '''Fitting training data without weights initialization
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        y - Nx1 vector where there is N samples and 1 output trait
        IMPORTANT - X needs to be 2D matrix and y 1D vector, even if N is equal to 1
        '''
        if not self.w_initialized:
            raise Exception('Weights are not initialized and yet partial fit is called!')

        for x_i, y_i in zip(X, y):
            self.__update_weights(x_i, y_i)
        return self

    def __shuffle(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Shuffling training set using numpty random generator to permute
        Arguments:
        X - NxM matrix where there is N samples and M input traits
        y - Nx1 vector where there is N samples and 1 output trait
        """
        r = self.random_generator.permutation(len(y))
        return X[r], y[r]

    def __update_weights(self, x_i: NDArray, y_i: float) -> float:
        """Uses Adaline training method to update weights
        Arguments:
        x_i - 1xM vector where there's 1 sample and M input traits
        y_i - 1x1 scalar (1 sample and 1 output trait)
        Return:
        cost - half of squared error (expected value - calculated activation function output)
        """
        z = self._net_input(x_i)  # this is a scalar as x_i is row vector
        activation_output = self._activation(z)
        e = y_i - activation_output  # this is a scalar as well
        self.w[1:] += self.eta * x_i.dot(e)
        self.w[0] += self.eta * e
        cost = 0.5 * (e ** 2)
        return cost
