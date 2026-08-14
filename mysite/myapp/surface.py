"""Reads the precomputed accuracy surface, when one has been built.

Written by the `build_surface` management command. Absent on a fresh
checkout, so every entry point here returns None rather than raising: the
configuration page then renders exactly as it did before the charts
existed, which keeps the app working for anyone who has not spent the
minutes it takes to generate a build artifact.
"""
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent / 'surface.json'

# Read once per process. The file cannot change under a running server: it
# is baked into the image next to the code.
_cache = {'read': False, 'data': None}


def _load():
    if not _cache['read']:
        try:
            _cache['data'] = json.loads(_PATH.read_text())
        except (OSError, ValueError):
            _cache['data'] = None
        _cache['read'] = True
    return _cache['data']


def _nearest(values, target):
    """Index of the closest grid point to `target`."""
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def payload(configuration):
    """The surface, plus where `configuration` sits on it.

    None when there is no usable surface, which the template reads as
    'draw no charts'.
    """
    data = _load()
    if data is None:
        return None

    axes = data['axes']
    if configuration.solver not in axes['solver']:
        # A solver the surface predates. Better no charts than charts
        # quietly describing a different estimator.
        return None

    try:
        current = {
            'solver': axes['solver'].index(configuration.solver),
            'C': _nearest(axes['C'], configuration.C),
            'max_iter': _nearest(axes['max_iter'], configuration.max_iter),
            'test_size': _nearest(axes['test_size'], configuration.test_size),
        }
    except TypeError:
        # A half-filled configuration from a form that failed validation.
        return None

    return {
        **data,
        'current': current,
        # The grid holds one set of measurements and one seed. When the
        # stored configuration has moved away from them the curves still
        # describe the parameter space, but a different one, and the page
        # has to say so.
        'matches_context': (
            data['context']['features'] == configuration.selected_features()
            and data['context']['random_state'] == configuration.random_state
        ),
    }
