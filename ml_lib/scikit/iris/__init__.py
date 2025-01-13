import numpy as np
from numpy.typing import NDArray
from sklearn import datasets, model_selection


def provide_data() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    return X_train, y_train, X_combined, y_combined
