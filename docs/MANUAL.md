# Users Manual

## HoeffdingTree
The method implements Very Fast Decision Tree (VFDT) aka Hoeffding Tree classification algorithm based on the following publication:
> Domingos, Pedro, and Geoff Hulten. "Mining high-speed data streams." In Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 71-80. 2000.

```python
class ml_rapids.HoeffdingTree( \
    max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200, split_confidence=0.0000001, \
    tie_threshold=0.05, binary_splits=False, stop_mem_management=False, remove_poor_atts=False, \
    leaf_learner='NB', nb_threshold=0, tree_property_index_list="", no_pre_prune=False)
```

| Parameter | Default Value | Description |
|:--------- |:------------- |:----------- |
| `max_byte_size` | `33554432` | |
| `memory_estimate_period` | `1000000` | |
| `grace_period` | `200` | |
| `split_confidence` | `0.0000001` | |
| `tie_threshold` | `0.05` | |
| `binary_splits` | `False` | |
| `stop_mem_management` | `False` | |
| `remove_poor_atts` | `False` | |
| `leaf_learner` | `'NB'` | |
| `nb_threshold` | `0` | |
| `tree_property_index_list` | `''` | |
| `no_pre_prune` | `False` | |

| Method                    | Description                 |
|:------------------------- |:--------------------------- |
| `fit(self, X, y)`         | Fits the model to the input data where `X` is a vector of feature vectors and `y` is a vector of targets. |
| `partial_fit(self, X, y)` | Not implemented. |
| `predict(self, X)`        | Predicts target values from a list of input feature vectors `X`. |
| `export_json(self)`       | Exports current model in JSON. |

## Example that tests all methods

```python
from streamdm import \
    HoeffdingTree, HoeffdingAdaptiveTree, NaiveBayes, \
    LogisticRegression, MajorityClass, Perceptron, Bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np
import time

methods = [
    {
        'name': 'Decision Tree (scikit-learn)',
        'ctor': DecisionTreeClassifier,
        'params': {}
    },
    {
        'name': 'Hoeffding Tree (streamDM)',
        'ctor': HoeffdingTree,
        'params': {
            'max_byte_size': 33554432,
            'memory_estimate_period': 1000000,
            'grace_period': 200,
            'split_confidence': 0.0000001,
            'tie_threshold': 0.05,
            'binary_splits': False,
            'stop_mem_management': False,
            'remove_poor_atts': False,
            'leaf_learner': 'NB',
            'nb_threshold': 0,
            'tree_property_index_list': "",
            'no_pre_prune': False
        }
    },
    {
        'name': 'Hoeffding Adaptive Tree (streamDM)',
        'ctor': HoeffdingAdaptiveTree,
        'params': {
            'max_byte_size': 33554432,
            'memory_estimate_period': 1000000,
            'grace_period': 200,
            'split_confidence': 0.0000001,
            'tie_threshold': 0.05,
            'binary_splits': False,
            'stop_mem_management': False,
            'remove_poor_atts': False,
            'leaf_learner': 'NB',
            'nb_threshold': 0,
            'tree_property_index_list': "",
            'no_pre_prune': False
        }
    },
    {
        'name': 'Bagging (streamDM)',
        'ctor': Bagging,
        'params': {
            'ensemble_size': 10,
            'learner': {
                'name': 'HoeffdingTree',
                'max_byte_size': 33554432,
                'memory_estimate_period': 1000000,
                'grace_period': 200,
                'split_confidence': 0.0000001,
                'tie_threshold': 0.05,
                'binary_splits': False,
                'stop_mem_management': False,
                'remove_poor_atts': False,
                'leaf_learner': 'NB',
                'nb_threshold': 0,
                'tree_property_index_list': "",
                'no_pre_prune': False
            }
        }
    },
    {
        'name': 'Naive Bayes (streamDM)',
        'ctor': NaiveBayes,
        'params': {}
    },
    {
        'name': 'Logistic Regression (streamDM)',
        'ctor': LogisticRegression,
        'params': {
            'learning_ratio': 0.01,
            'lambda': 0.0001
        }
    },
    {
        'name': 'Perceptron (streamDM)',
        'ctor': Perceptron,
        'params': {
            'learning_ratio': 1.0
        }
    },
    {
        'name': 'Majority Class (streamDM)',
        'ctor': MajorityClass,
        'params': {}
    }
]

# Prepare data
iris = load_iris()
X = iris.data
y = iris.target
print(iris.DESCR)

X_shuffled, y_shuffled = shuffle(X, y)

print('\n')
for i, x in enumerate(X_shuffled):
    print(x, y_shuffled[i])

for method in methods:
    print('\n' + '=' * 53)
    print(method['name'])
    print('=' * 53 + '\n')

    kf = KFold(n_splits=10, shuffle=False)

    t = time.time()
    y_pred = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]

        # Build model
        learner = method['ctor'](**method['params'])
        learner.fit(X_train, y_train)

        # Predict
        pred = learner.predict(X_test)
        y_pred += pred.tolist()

    print(f'Time: {time.time() - t}\n')

    # Report
     print(classification_report(y, y_pred, target_names=iris.target_names))

if __name__ == '__main__':
    test()
```
