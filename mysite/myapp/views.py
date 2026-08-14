import json

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from . import classifier, surface, training
from .forms import IrisPredictionForm, ModelConfigurationForm
from .models import ModelConfiguration


def _form_context(form):
    """Context for the measurement page, whether blank or showing errors."""
    return {
        'form': form,
        'model': classifier.model_info(),
        'example_input': json.dumps(classifier.EXAMPLE_INPUT),
    }


@login_required
def index(request):
    """The measurement form, alongside an explanation of the model."""
    return render(request, 'myapp/index.html', _form_context(IrisPredictionForm()))


@login_required
def predict_species(request):
    """Classify one flower and show the result against the model's record."""
    if request.method != 'POST':
        return redirect('index')

    form = IrisPredictionForm(request.POST)
    if not form.is_valid():
        return render(request, 'myapp/index.html', _form_context(form), status=400)

    measurements = form.cleaned_data
    info = classifier.model_info()
    context = {
        'result': classifier.predict(**measurements),
        'measurements': [
            dict(field, value=measurements[field['key']])
            for field in info['fields']
        ],
        'model': info,
    }
    return render(request, 'myapp/response.html', context)


@login_required
def configuration(request):
    """Edit the hyperparameters and retrain, reporting the effect."""
    instance = ModelConfiguration.load()

    if request.method == 'POST':
        accuracy_before = classifier.accuracy_now()

        if 'reset' in request.POST:
            for key, value in training.DEFAULTS.items():
                if key == 'features':
                    for feature in training.FEATURE_KEYS:
                        setattr(instance, f'use_{feature}', feature in value)
                else:
                    setattr(instance, key, value)
            instance.save()
            form = ModelConfigurationForm(instance=instance)
            saved = True
        else:
            form = ModelConfigurationForm(request.POST, instance=instance)
            saved = form.is_valid()
            if saved:
                form.save()

        if saved:
            # No cache clearing needed: the classifier re-fingerprints the
            # configuration on every call, so this already reflects the save.
            accuracy_after = classifier.accuracy_now()
            # Survives the redirect so refreshing the page cannot resubmit
            request.session['retrain'] = {
                'before': round(accuracy_before * 100, 1),
                'after': round(accuracy_after * 100, 1),
            }
            return redirect('configuration')
    else:
        form = ModelConfigurationForm(instance=instance)

    return render(request, 'myapp/configuration.html', {
        'form': form,
        'model': classifier.model_info(),
        'retrain': request.session.pop('retrain', None),
        # Read from the database rather than from `instance`, which a failed
        # validation can leave holding values that are not on any grid.
        'surface': surface.payload(ModelConfiguration.load()),
    })
