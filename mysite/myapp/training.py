"""Trains and evaluates an iris classifier from a configuration.

Pure scikit-learn: this module imports no Django, so it can be used both by
the web app and by the models/train.py command line script. It is the single
definition of how this project trains and scores a model.

A configuration is a plain dict:

    {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs',
     'test_size': 0.2, 'random_state': 42,
     'features': ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width']}
"""
import warnings

import numpy as np
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Column order of iris.data. Everything keys off this order.
FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Every one of these supports the l2 penalty, so any choice stays valid
SOLVERS = ['lbfgs', 'liblinear', 'newton-cg', 'saga']

# The baseline the configuration page starts from and resets to.
#
# The seed is 7 rather than the conventional 42 on purpose. Iris is close to
# linearly separable on the petals, so with seed 42 this split scores 100%
# across nearly the whole range the page allows: every test_size from 0.05
# to 0.5, every solver, C from 0.1 to 1000. A user could move any control
# and watch nothing happen, with no room to improve on a perfect score.
#
# Seed 7 starts at 86.7% and responds in both directions: raising test_size
# reaches 94.7%, while a weak C, too few iterations or dropping the petals
# all visibly degrade it. Nothing about the model changed, only which
# flowers land in the held-out split.
DEFAULTS = {
    'C': 1.0,
    'max_iter': 1000,
    'solver': 'lbfgs',
    'test_size': 0.2,
    'random_state': 7,
    'features': list(FEATURE_KEYS),
}


def feature_indices(features):
    """Column positions for the named features, in dataset order."""
    return [FEATURE_KEYS.index(key) for key in FEATURE_KEYS if key in features]


def feature_stats():
    """Range of each measurement across the whole dataset.

    Independent of any configuration: the form asks for all four
    measurements regardless of which ones the model is currently using.
    """
    iris = load_iris()
    stats = []
    for column, key in enumerate(FEATURE_KEYS):
        values = iris.data[:, column]
        stats.append({
            'key': key,
            'name': str(iris.feature_names[column]),
            'min': round(float(values.min()), 1),
            'max': round(float(values.max()), 1),
            'mean': round(float(values.mean()), 2),
        })
    return stats


def train_and_evaluate(config):
    """Fit a classifier under `config` and score it on the held-out split.

    Returns (estimator, metrics). The estimator expects only the configured
    features, in dataset order, so callers must slice inputs the same way
    `feature_indices` does.
    """
    iris = load_iris()
    indices = feature_indices(config['features'])
    if not indices:
        raise ValueError('at least one feature is required to train')

    X = iris.data[:, indices]
    y = iris.target
    class_names = [str(name) for name in iris.target_names]
    class_indices = list(range(len(class_names)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=config['random_state'],
    )

    model = LogisticRegression(
        C=config['C'],
        max_iter=config['max_iter'],
        solver=config['solver'],
    )

    # A low max_iter makes the fit stop before it settles. That is a real
    # behaviour change the user asked to see, so capture it instead of
    # letting the warning disappear into the server log.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X_train, y_train)
    converged = not any(
        issubclass(entry.category, ConvergenceWarning) for entry in caught)

    y_pred = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=class_indices, zero_division=0)

    used = [FEATURE_KEYS[i] for i in indices]
    metrics = {
        'algorithm': (
            f"Logistic Regression (solver={config['solver']}, "
            f"C={config['C']}, max_iter={config['max_iter']})"
        ),
        'dataset': 'Fisher iris, 150 samples, 3 species',
        'classes': class_names,
        'features_used': used,
        'features': feature_stats(),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'converged': converged,
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        # Row = true species, column = predicted species
        'confusion_matrix': confusion_matrix(
            y_test, y_pred, labels=class_indices).tolist(),
        'per_class': [
            {
                'name': class_names[i],
                'precision': round(float(precision[i]), 3),
                'recall': round(float(recall[i]), 3),
                'f1': round(float(f1[i]), 3),
                'support': int(support[i]),
            }
            for i in class_indices
        ],
        'coefficients': [
            {
                'name': class_names[i],
                'weights': [round(float(w), 3) for w in model.coef_[i]],
            }
            for i in class_indices
        ],
    }
    return model, metrics


def predict_one(model, config, measurements):
    """Run one flower through `model`, using only the configured features."""
    indices = feature_indices(config['features'])
    vector = np.array(
        [measurements[FEATURE_KEYS[i]] for i in indices]).reshape(1, -1)

    class_index = int(model.predict(vector)[0])
    probabilities = [float(p) for p in model.predict_proba(vector)[0]]
    return class_index, probabilities
