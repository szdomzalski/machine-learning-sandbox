import numpy as np
from sklearn import datasets, linear_model, metrics, model_selection, preprocessing

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

model = linear_model.Perceptron(max_iter=40, eta0=0.1, random_state=1)
model.fit(X_train_std, y_train)

y_predict = model.predict(X_test_std)

print(f'Number of falsely classified test samples: {(y_test != y_predict).sum()} / {len(y_test)}')
print(f'Ratio of falsely classified test samples: {(y_test != y_predict).sum()/len(y_test):.2f}')
print(f'Accuracy score: {metrics.accuracy_score(y_test, y_predict):.2f}')
