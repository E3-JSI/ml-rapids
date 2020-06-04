from ml_rapids import HoeffdingTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import numpy as np
import time


def test():
    methods = [
        {
            'name': 'Decision Tree (scikit-learn)',
            'ctor': DecisionTreeClassifier
        },
        {
            'name': 'Hoeffding Tree (ml-rapids)',
            'ctor': HoeffdingTree
        }
    ]

    # Prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(iris.DESCR)

    for method in methods:
        print('\n' + '=' * 53)
        print(method['name'])
        print('=' * 53 + '\n')

        kf = KFold(n_splits=10)

        t = time.time()
        y_pred = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Build model
            learner = method['ctor']()
            learner.fit(X_train, y_train)

            # Predict
            pred = learner.predict(X_test)
            y_pred += pred.tolist()

        print(f'Time: {time.time() - t}\n')

        # Report
        print(classification_report(y, y_pred, target_names=iris.target_names))


if __name__ == '__main__':
    test()
