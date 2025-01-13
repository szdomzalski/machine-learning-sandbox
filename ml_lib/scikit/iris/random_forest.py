from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from ml_lib.scikit.iris import provide_data
from ml_lib.utils.plots import plot_decision_regions


def main():
    X_train, y_train, X_combined, y_combined = provide_data()

    # 25 trees trained by Gini's impurity measure, 2 cores training process
    model = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
    model.fit(X_train, y_train)

    plot_decision_regions(X_combined, y_combined, classifier=model, test_idx=range(105, 150))

    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
