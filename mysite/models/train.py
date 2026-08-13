"""Regenerate the baseline iris model artifacts from the command line.

Run from anywhere:  python models/train.py

Writes iris_model.pkl and iris_metrics.json side by side in this directory,
using the default configuration in myapp/training.py.

The running web app does NOT read these files: it trains from whatever is
stored in the ModelConfiguration table and keeps the result in memory. These
artifacts exist as a reproducible record of the baseline, so you can always
see what the model looked like before anyone changed the settings.
"""
import json
import os
import sys

import joblib

# myapp.training is plain scikit-learn with no Django imports, so it can be
# used directly from a script. Both paths train the same way by construction.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myapp import training  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'iris_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'iris_metrics.json')

model, metrics = training.train_and_evaluate(training.DEFAULTS)

joblib.dump(model, MODEL_PATH)
with open(METRICS_PATH, 'w') as handle:
    json.dump(metrics, handle, indent=2)

print(f'Model written to {MODEL_PATH}')
print(f'Metrics written to {METRICS_PATH} (accuracy {metrics["accuracy"]})')
