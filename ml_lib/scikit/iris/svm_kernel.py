import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets, model_selection, preprocessing
from sklearn.svm import SVC

from ml_lib.utils.plots import plot_decision_regions


def main():
    # XOR experiment
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    plt.figure()
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.show(block=False)

    # Test lower gamma
    model = SVC(kernel='rbf', C=10.0, random_state=1, gamma=0.1)
    model.fit(X_xor, y_xor)

    plot_decision_regions(X_xor, y_xor, classifier=model)

    plt.title('gamma = 0.1')
    plt.legend(loc='upper left')
    plt.show(block=False)

    # Test greater gamma
    model = SVC(kernel='rbf', C=10.0, random_state=1, gamma=10.0)
    model.fit(X_xor, y_xor)

    plot_decision_regions(X_xor, y_xor, classifier=model)
    plt.title('gamma = 10')
    plt.legend(loc='upper left')
    plt.show(block=False)

    # Iris dataset classification using rbf kernel
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # Scaling features (zeroing mean and setting unital variance)
    std_scaler = preprocessing.StandardScaler()
    std_scaler.fit(X_train)  # we assume train data only to simulate real life environment
    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # Low gamma
    model = SVC(kernel='rbf', C=1.0, random_state=1, gamma=0.2)
    model.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=model, test_idx=range(105, 150))

    plt.title('Gamma 0.2')
    plt.ylabel('Petal width [standardized]')
    plt.xlabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show(block=False)

    # Greater gamma
    model = SVC(kernel='rbf', C=1.0, random_state=1, gamma=100.0)
    model.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=model, test_idx=range(105, 150))

    plt.title('Gamma 100')
    plt.ylabel('Petal width [standardized]')
    plt.xlabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
