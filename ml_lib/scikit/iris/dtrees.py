import matplotlib.pyplot as plt
import numpy as np
import os
from pydotplus import graph_from_dot_data
from sklearn import datasets, model_selection
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from ml_lib.utils.plots import plot_decision_regions


def gini(p: float) -> float:
    return 2 * p * (1 - p)


def entropy(p: float) -> float:
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


def classification_error(p: float) -> float:
    return 1 - np.max([p, 1 - p])


def export_dtree(tree_model, class_names: list[str], feature_names: list[str], out_file: str | None = None,
                 tree_name: str = 'tree.png') -> None:
    os.makedirs(os.path.dirname(tree_name), exist_ok=True)
    dot_data = export_graphviz(tree_model, filled=True, rounded=True, class_names=class_names,
                               feature_names=feature_names, out_file=out_file)
    graph = graph_from_dot_data(dot_data)
    graph.write_png(tree_name)


def test_impurity_measures() -> None:
    x = np.arange(0.0, 1.0, 0.01)
    entropy_values = [entropy(p) if p != 0.0 else None for p in x]
    scaled_entropy = [e * 0.5 if e else None for e in entropy_values]
    classification_errors = [classification_error(p) for p in x]
    gini_values = [gini(p) for p in x]
    plt.figure()
    ax = plt.subplot(111)
    for value, label \
            in zip((entropy_values, scaled_entropy, gini_values, classification_errors,),
                   ('Entropy', 'Scaled entropy', "Gini's measure", 'Classification error',)):
        ax.plot(x, value, label=label, lw=2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p')
    plt.ylabel('Impurity measure')
    plt.show(block=False)


def train_simple_tree(*, min_depth: int = 1, max_depth: int = 6) -> None:
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    block = False
    for depth in range(min_depth, max_depth):
        if depth >= max_depth - 1:
            block = True

        model = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=1)
        model.fit(X_train, y_train)

        export_dtree(model, ['Setosa', 'Versicolor', 'Virginica'], ['Petal length', 'Petal width'],
                     tree_name=f'trees/tree{depth:02}.png')

        plot_decision_regions(X_combined, y_combined, model, test_idx=range(105, 150))
        plt.title(f'D-tree depth: {depth}')
        plt.ylabel('Petal width [standardized]')
        plt.xlabel('Petal length [standardized]')
        plt.legend(loc='upper left')
        plt.show(block=block)


def main():
    # test_impurity_measures()
    # train_simple_tree(min_depth=3, max_depth=10)
    train_simple_tree(min_depth=3, max_depth=5)
