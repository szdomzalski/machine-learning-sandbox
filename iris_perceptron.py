import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEBUG = False

ROWS_CONSIDERED = 100

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
if DEBUG:
    print('Read dataframe:')
    print(df)
# Choose 4th column, first 100 rows
y = df[0:ROWS_CONSIDERED][4].values
if DEBUG:
    print('Correct output vector:')
    print(f'Type {{{type(y)}}}')
    print(y)
# Replace string labels with numerical values
y = np.where(y == 'Iris-setosa', -1, y)
y = np.where(y == 'Iris-versicolor', 1, y)
if DEBUG:
    print('Correct output vector with integers instead of labels:')
    print(f'Type {{{type(y)}}}')
    print(y)

# Prepare training set - for now choosing only 2 features (columns with indexes 0 and 2)
X = df[0:ROWS_CONSIDERED][[0, 2]].values
if DEBUG:
    print('Training samples:')
    print(f'Type {{{type(y)}}}')
    print(X)

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('Calyx sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
