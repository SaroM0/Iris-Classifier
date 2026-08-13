"""Runtime access to the classifier described by the active configuration.

Holds the fitted estimator in memory, keyed by the configuration's
fingerprint, so editing the configuration retrains exactly once and every
later request is served from the cache. Nothing here writes to disk.

This is the only module views and templates talk to. It renamed from
model.py when myapp gained a real models.py, so that "model" always means
the Django model and "classifier" always means the scikit-learn one.
"""
from . import training
from .models import ModelConfiguration

# Human-facing labels and guidance, keyed by feature
FEATURE_LABELS = {
    'sepal_length': ('Sepal length', 'The outer leaf-like petal that protects the bud.'),
    'sepal_width': ('Sepal width', 'Measured across the widest point of the sepal.'),
    'petal_length': ('Petal length', 'Usually the single strongest signal.'),
    'petal_width': ('Petal width', 'Almost as decisive as petal length.'),
}

# A known setosa sample, offered on the form as a worked example
EXAMPLE_INPUT = {
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2,
}

# Readable names for the settings, used when reporting what the user changed
_SETTING_LABELS = {
    'C': 'regularisation (C)',
    'max_iter': 'maximum iterations',
    'solver': 'solver',
    'test_size': 'held-out proportion',
    'random_state': 'split seed',
    'features': 'measurements used',
}

# Single-process cache. A race between two threads only costs a duplicate
# fit, so no lock is warranted for a dataset this size.
_cache: dict = {'fingerprint': None, 'estimator': None, 'metrics': None}


def _active():
    """The active configuration with its fitted estimator and metrics.

    The configuration is re-read and re-fingerprinted on every call, so the
    cache invalidates itself whenever the row changes — whether that came
    from the configuration page, the admin, or a shell. Nothing has to
    remember to clear it.
    """
    configuration = ModelConfiguration.load()
    fingerprint = configuration.fingerprint()

    if _cache['fingerprint'] != fingerprint:
        estimator, metrics = training.train_and_evaluate(
            configuration.as_training_config())
        _cache.update({
            'fingerprint': fingerprint,
            'estimator': estimator,
            'metrics': metrics,
        })

    return configuration, _cache['estimator'], _cache['metrics']


def predict(**measurements):
    """Classify one flower under the active configuration."""
    configuration, estimator, metrics = _active()
    class_index, probabilities = training.predict_one(
        estimator, configuration.as_training_config(), measurements)

    classes = metrics['classes']
    ranked = sorted(
        (
            {
                'name': classes[i],
                'probability': round(probabilities[i], 4),
                'percent': round(probabilities[i] * 100, 1),
                'is_predicted': i == class_index,
            }
            for i in range(len(classes))
        ),
        key=lambda row: row['probability'],
        reverse=True,
    )

    return {
        'species': classes[class_index],
        'confidence': round(probabilities[class_index] * 100, 1),
        'probabilities': ranked,
    }


def _confusion_rows(metrics):
    """Confusion matrix as rows of cells, each tagged for display.

    Row = true species, column = predicted species. `share` drives the
    heat tint in the template.
    """
    classes = metrics['classes']
    rows = []
    for true_index, counts in enumerate(metrics['confusion_matrix']):
        row_total = sum(counts)
        rows.append({
            'true_label': classes[true_index],
            'total': row_total,
            'cells': [
                {
                    'count': count,
                    'predicted_label': classes[predicted_index],
                    'is_correct': predicted_index == true_index,
                    'is_error': predicted_index != true_index and count > 0,
                    'share': round(count / row_total * 100) if row_total else 0,
                }
                for predicted_index, count in enumerate(counts)
            ],
        })
    return rows


def _coefficient_rows(metrics):
    """Per-species weights, scaled so the template can draw comparable bars.

    Only the features the model actually uses appear here.
    """
    names = [FEATURE_LABELS[key][0] for key in metrics['features_used']]
    largest = max(
        (abs(w) for entry in metrics['coefficients'] for w in entry['weights']),
        default=1.0,
    ) or 1.0

    return [
        {
            'name': entry['name'],
            'weights': [
                {
                    'feature': names[i],
                    'value': weight,
                    'magnitude': round(abs(weight) / largest * 100),
                    'is_positive': weight > 0,
                }
                for i, weight in enumerate(entry['weights'])
            ],
        }
        for entry in metrics['coefficients']
    ]


def _describe_setting(name, value):
    """Turn a stored setting into something readable on the page."""
    if name == 'features':
        return ', '.join(FEATURE_LABELS[key][0] for key in value)
    return value


def _differences(configuration):
    """What the user changed, in words rather than raw Python values."""
    return [
        {
            'name': _SETTING_LABELS[difference['name']],
            'current': _describe_setting(difference['name'], difference['current']),
            'default': _describe_setting(difference['name'], difference['default']),
        }
        for difference in configuration.differences_from_default()
    ]


def form_fields():
    """Guidance for each input, flagging the ones the model ignores.

    All four are always asked for: you measured the whole flower. Marking
    the unused ones keeps the same input comparable across configurations.
    """
    _, _, metrics = _active()
    used = metrics['features_used']

    fields = []
    for stat in metrics['features']:
        label, hint = FEATURE_LABELS[stat['key']]
        fields.append({
            'key': stat['key'],
            'label': label,
            'hint': hint,
            'min': stat['min'],
            'max': stat['max'],
            'mean': stat['mean'],
            'example': EXAMPLE_INPUT[stat['key']],
            'is_used': stat['key'] in used,
        })
    return fields


def model_info():
    """Everything the templates need to explain the current model."""
    configuration, _, metrics = _active()
    rows = _confusion_rows(metrics)
    errors = sum(
        cell['count'] for row in rows for cell in row['cells']
        if cell['is_error']
    )
    fields = form_fields()

    return {
        'algorithm': metrics['algorithm'],
        'dataset': metrics['dataset'],
        'classes': metrics['classes'],
        'fields': fields,
        'ignored_fields': [f for f in fields if not f['is_used']],
        'n_train': metrics['n_train'],
        'n_test': metrics['n_test'],
        'converged': metrics['converged'],
        'accuracy': metrics['accuracy'],
        'accuracy_percent': round(metrics['accuracy'] * 100, 1),
        'errors': errors,
        'confusion_rows': rows,
        'per_class': metrics['per_class'],
        'coefficients': _coefficient_rows(metrics),
        'configuration': configuration,
        'is_default': configuration.is_default(),
        'differences': _differences(configuration),
        # The story about petals dominating only holds while they are in use
        'uses_petals': any(
            key.startswith('petal') for key in metrics['features_used']),
    }


def accuracy_now():
    """Current accuracy only, for before/after comparisons."""
    _, _, metrics = _active()
    return metrics['accuracy']
