from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, model_selection, preprocessing
from sklearn.svm import SVC

from ml_lib.utils.plots import plot_decision_regions


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

    # Scaling features (zeroing mean and setting unital variance)
    std_scaler = preprocessing.StandardScaler()
    std_scaler.fit(X_train)  # we assume train data only to simulate real life environment
    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    model = SVC(kernel='linear', C=1.0, random_state=1)
    model.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std, y_combined, classifier=model, test_idx=range(105, 150))

    plt.ylabel('Petal width [standardized]')
    plt.xlabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
