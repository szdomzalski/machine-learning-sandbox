import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, model_selection

from .models import logistic_regression
from .utils.plots import plot_decision_regions


def main():
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Stratification check
    print(f'Number of each label occurences in dataset y: {np.bincount(y)}')
    print(f'Number of each label occurences in training subset: {np.bincount(y_train)}')
    print(f'Number of each label occurences in testing subset: {np.bincount(y_test)}')

    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    # Scaling features (zeroing mean and setting unital variance)
    # std_scaler = preprocessing.StandardScaler()
    # std_scaler.fit(X_train)  # we assume train data only to simulate real life environment
    # X_train_std = std_scaler.transform(X_train)
    # X_test_std = std_scaler.transform(X_test)

    model = logistic_regression.LogisticRegressionGD(epochs=1000)
    model.fit(X_train_01_subset, y_train_01_subset)

    # y_predict = model.predict(X_test_std)

    # print(f'Number of falsely classified test samples: {(y_test != y_predict).sum()} / {len(y_test)}')
    # print(f'Ratio of falsely classified test samples: {(y_test != y_predict).sum()/len(y_test):.2f}')
    # print(f'Accuracy score: {metrics.accuracy_score(y_test, y_predict):.2f}')

    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_train_01_subset, np.where(y_train_01_subset == 0, 'Iris setosa', 'Iris versicolor'), model)

    plt.ylabel('Petal width [cm]')
    plt.xlabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()
