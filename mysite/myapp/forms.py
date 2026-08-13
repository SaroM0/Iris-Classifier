from django import forms

from . import classifier, training
from .models import ModelConfiguration


class IrisPredictionForm(forms.Form):
    """The four measurements, in centimetres.

    All four are always requested even when the active configuration
    ignores some of them, so the same flower can be compared across
    configurations. Labels and observed ranges come from the model itself.
    """

    sepal_length = forms.FloatField(required=True, min_value=0, max_value=20)
    sepal_width = forms.FloatField(required=True, min_value=0, max_value=20)
    petal_length = forms.FloatField(required=True, min_value=0, max_value=20)
    petal_width = forms.FloatField(required=True, min_value=0, max_value=20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for spec in classifier.form_fields():
            field = self.fields[spec['key']]
            field.label = spec['label']
            field.help_text = spec['hint']
            # Read by the template to describe the training data
            field.range_text = (
                f"observed range {spec['min']}–{spec['max']} cm "
                f"· average {spec['mean']}"
            )
            # Read by the template to grey out measurements the model ignores
            field.is_used = spec['is_used']
            field.widget.attrs.update({
                'step': '0.1',
                'min': 0,
                'max': 20,
                'inputmode': 'decimal',
                'placeholder': f"e.g. {spec['example']}",
            })


class ModelConfigurationForm(forms.ModelForm):
    """Edits the single configuration row that drives training."""

    class Meta:
        model = ModelConfiguration
        fields = [
            'C', 'max_iter', 'solver', 'test_size', 'random_state',
            'use_sepal_length', 'use_sepal_width',
            'use_petal_length', 'use_petal_width',
        ]

    # Grouped for the template: these drive the fit itself
    ESTIMATOR_FIELDS = ['C', 'max_iter', 'solver']
    # ...and these decide what it is trained and scored on
    SPLIT_FIELDS = ['test_size', 'random_state']
    FEATURE_FIELDS = [f'use_{key}' for key in training.FEATURE_KEYS]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self.ESTIMATOR_FIELDS + self.SPLIT_FIELDS:
            self.fields[name].widget.attrs.update({'class': 'config-input'})
        self.fields['C'].widget.attrs.update({'step': '0.01'})
        self.fields['test_size'].widget.attrs.update({'step': '0.05'})

    def grouped(self, names):
        """Bound fields for one section of the page, in the given order."""
        return [self[name] for name in names]

    @property
    def estimator_fields(self):
        return self.grouped(self.ESTIMATOR_FIELDS)

    @property
    def split_fields(self):
        return self.grouped(self.SPLIT_FIELDS)

    @property
    def feature_fields(self):
        return self.grouped(self.FEATURE_FIELDS)
