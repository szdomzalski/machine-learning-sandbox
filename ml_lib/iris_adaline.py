import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import adaline
from utils.plots import plot_decision_regions


DEBUG = False
ROWS_CONSIDERED = 100


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    if DEBUG:
        print('Read dataframe:')
        print(df)
    # Choose 4th column, first 100 rows
    y = df[0:ROWS_CONSIDERED][4].values
    if DEBUG:
        print('Correct output vector:')
        print(f'Type {{{type(y)}}}, shape {{{y.shape}}}')
        print(y)
    # Replace string labels with numerical values
    y = np.where(y == 'Iris-setosa', -1, y)
    y = np.where(y == 'Iris-versicolor', 1, y)
    if DEBUG:
        print('Correct output vector with integers instead of labels:')
        print(f'Type {{{type(y)}}}, shape {{{y.shape}}}')
        print(y)

    # Prepare training set - for now choosing only 2 features (columns with indexes 0 and 2)
    X = df[0:ROWS_CONSIDERED][[0, 2]].values
    if DEBUG:
        print('Training samples:')
        print(f'Type {{{type(X)}}}, shape {{{X.shape}}}')
        print(X)

    # Training data simple visualisation
    plt.figure()
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
    plt.xlabel('Calyx sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show(block=False)

    # Training AdalineGD model on 2-level iris dataset
    etas = (0.01, 0.0001,)
    scaling_functions = (np.log10, lambda x: x)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    for i, (eta, scaling_function) in enumerate(zip(etas, scaling_functions)):
        model = adaline.AdalineGD(eta, 50).fit(X, y)
        ax[i].plot(range(1, len(model.cost) + 1), scaling_function(model.cost), marker='o')
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel('Scaled errors sum')
        ax[i].set_title(f'AdalineGD - learning rate {eta:.5f}')
    plt.show(block=False)

    # Standardization of matrix X
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    model = adaline.AdalineGD(eta=0.01, epochs=15)
    model.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=model)
    plt.title('AdalineGD - gradient descent')
    plt.xlabel('Calyx sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.plot(range(1, len(model.cost) + 1), model.cost, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum of squared errors')
    plt.show(block=False)

    model = adaline.AdalineSGD(eta=0.01, epochs=15)
    model.fit(X_std, y)
    model.partial_fit(X_std[0:1, :], y[0:1])

    plot_decision_regions(X_std, y, classifier=model)
    plt.title('AdalineSGD - stochastic gradient descent')
    plt.xlabel('Calyx sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show(block=False)

    plt.figure()
    plt.plot(range(1, len(model.cost) + 1), model.cost, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average cost')
    plt.show()
