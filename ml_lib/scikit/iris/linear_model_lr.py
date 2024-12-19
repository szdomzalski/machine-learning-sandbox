import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics, model_selection, preprocessing

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

    # C is inverse of lambda, lower C means stronger regularization
    model = linear_model.LogisticRegression(C=1000.0, random_state=1)
    model.fit(X_train_std, y_train)

    y_predict = model.predict(X_test_std)

    print(f'Number of falsely classified test samples: {(y_test != y_predict).sum()} / {len(y_test)}')
    print(f'Ratio of falsely classified test samples: {(y_test != y_predict).sum()/len(y_test):.2f}')
    print(f'Accuracy score: {metrics.accuracy_score(y_test, y_predict):.2f}')

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std, y_combined, model, test_idx=[*range(105, 150)])

    plt.ylabel('Petal width [standardized]')
    plt.xlabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show(block=False)

    probability_predicted = model.predict_proba(X_test_std[:3, :])
    print(f'Predicted probability for 1st 3 samples:\n{probability_predicted}')
    print(f'Summing probability check:\n{probability_predicted.sum(axis=1)}')
    print(f'Highest probability column for each sample:\n{probability_predicted.argmax(axis=1)}')
    print(f'Comparison with predict method:\n{model.predict(X_test_std[:3, :])}')

    weights, params = [], []
    for c in np.arange(-5, 5):
        model = linear_model.LogisticRegression(C=10.0 ** c, random_state=1)
        model.fit(X_train_std, y_train)
        weights.append(model.coef_[1])
        params.append(10.0 ** c)
    weights = np.array(weights)

    plt.figure()
    plt.plot(params, weights[:, 0], label='Petal length')
    plt.plot(params, weights[:, 1], label='Petal width')
    plt.ylabel('Weights coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()
