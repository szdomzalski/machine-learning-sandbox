import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import plots
from models import perceptron


DEBUG = True
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

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
    plt.xlabel('Calyx sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show(block=False)

    # Training perceptron on 2-level iris dataset
    ppn = perceptron.Perceptron(eta=0.1, epochs=10)
    ppn.fit(X, y)

    plt.figure()
    plt.plot(range(1, len(ppn.number_of_e) + 1), ppn.number_of_e, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show(block=False)

    plots.plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('Calyx sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
