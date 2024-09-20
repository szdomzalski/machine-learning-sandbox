import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from numpy.typing import NDArray


# For 2 dimensional plots
def plot_decision_regions(X: NDArray, y: NDArray, classifier: any, resolution: float = 0.02) -> None:
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_map = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # xx1 and xx2 are matrices and need to be flattened to 2 columns of X matrix
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Reshaping to mesh dimensions
    Z = Z.reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=color_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for i, class_label in enumerate(np.unique(y)):
        plt.scatter(X[y == class_label, 0], X[y == class_label, 1], alpha=0.8, c=colors[i], marker=markers[i],
                    label=class_label, edgecolors='black')
