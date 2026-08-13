from django.core.exceptions import ValidationError
from django.core.validators import (
    MaxValueValidator,
    MinValueValidator,
)
from django.db import models

from . import training


class ModelConfiguration(models.Model):
    """The hyperparameters the classifier is currently trained with.

    Deliberately a single row: there is one active configuration for the
    whole app, and editing it retrains the model. `load()` is the only
    supported way to read it.
    """

    SOLVER_CHOICES = [(name, name) for name in training.SOLVERS]

    C = models.FloatField(
        'regularisation strength (C)',
        default=training.DEFAULTS['C'],
        validators=[MinValueValidator(0.0001), MaxValueValidator(1000)],
        help_text='Lower values force a simpler, more cautious model. '
                  'Very low values make it visibly worse.',
    )
    max_iter = models.PositiveIntegerField(
        'maximum iterations',
        default=training.DEFAULTS['max_iter'],
        validators=[MinValueValidator(1), MaxValueValidator(10000)],
        help_text='How long the fit is allowed to keep improving. '
                  'Too few and it stops before it has settled.',
    )
    solver = models.CharField(
        'solver',
        max_length=20,
        default=training.DEFAULTS['solver'],
        choices=SOLVER_CHOICES,
        help_text='The algorithm that does the fitting. On a dataset this '
                  'small they mostly agree.',
    )
    test_size = models.FloatField(
        'held-out proportion',
        default=training.DEFAULTS['test_size'],
        validators=[MinValueValidator(0.05), MaxValueValidator(0.5)],
        help_text='Share of the 150 flowers kept back for scoring. '
                  'Larger means a more trustworthy score but less training data.',
    )
    random_state = models.IntegerField(
        'split seed',
        default=training.DEFAULTS['random_state'],
        help_text='Changing this reshuffles which flowers are used for '
                  'training and which for scoring.',
    )

    use_sepal_length = models.BooleanField('use sepal length', default=True)
    use_sepal_width = models.BooleanField('use sepal width', default=True)
    use_petal_length = models.BooleanField('use petal length', default=True)
    use_petal_width = models.BooleanField('use petal width', default=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'model configuration'

    def __str__(self):
        return f'C={self.C} max_iter={self.max_iter} solver={self.solver}'

    @classmethod
    def load(cls):
        """The one and only configuration row, created on first use."""
        configuration, _ = cls.objects.get_or_create(pk=1)
        return configuration

    def save(self, *args, **kwargs):
        # Pin the primary key so this table can never hold a second row.
        # 'id' is excluded from validation because colliding with the
        # existing row is the intended outcome, not an error.
        self.pk = 1
        self.full_clean(exclude=['id'])
        return super().save(*args, **kwargs)

    def clean(self):
        if not self.selected_features():
            raise ValidationError(
                'The model needs at least one measurement to work with. '
                'Select at least one.'
            )

    def selected_features(self):
        """Configured feature keys, in dataset order."""
        return [
            key for key in training.FEATURE_KEYS
            if getattr(self, f'use_{key}')
        ]

    def as_training_config(self):
        """Plain dict for myapp.training, which knows nothing about Django."""
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'solver': self.solver,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'features': self.selected_features(),
        }

    def fingerprint(self):
        """Identity of the trained model this configuration produces.

        The classifier cache keys off this, so any edit that would change
        the fitted model invalidates it and anything else does not.
        """
        config = self.as_training_config()
        return (
            config['C'], config['max_iter'], config['solver'],
            config['test_size'], config['random_state'],
            tuple(config['features']),
        )

    def is_default(self):
        return self.as_training_config() == training.DEFAULTS

    def differences_from_default(self):
        """Which settings the user has moved away from the baseline."""
        config = self.as_training_config()
        return [
            {'name': key, 'current': config[key], 'default': default}
            for key, default in training.DEFAULTS.items()
            if config[key] != default
        ]
