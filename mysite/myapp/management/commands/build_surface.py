"""Precomputes the accuracy of every configuration on a grid.

Running these fits ahead of time is what lets the configuration page redraw
its charts while the user drags a slider: the page ships the whole surface
and the browser only reads from it. Computing a single curve on demand
costs about half a second, which is fine for one page load and useless for
dragging.

The output is a build artifact, not source. It is written next to the app
and excluded from version control; `deploy/Dockerfile` runs this command so
that every image carries a surface matching the code that built it.
"""
import json
import time
from pathlib import Path

from django.core.management.base import BaseCommand

from myapp import training
from myapp.models import ModelConfiguration

# Log-spaced across the range the model validators allow. On a linear axis
# every value below 100 would crowd into the leftmost tenth of the chart.
C_VALUES = [
    0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
    1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
]

# Dense where convergence actually happens. Past roughly fifty iterations
# the fit has settled, so further points would all be the same height.
MAX_ITER_VALUES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 100, 300, 1000]

# The full permitted range, at the step the form already uses.
TEST_SIZE_VALUES = [round(0.05 * step, 2) for step in range(1, 11)]

OUTPUT = Path(__file__).resolve().parents[2] / 'surface.json'


class Command(BaseCommand):
    help = 'Precompute the accuracy surface behind the configuration charts.'

    def handle(self, *args, **options):
        # Measurements and seed cannot be grid axes: the combinations of
        # measurements multiply out to far too many fits, and seeds are
        # unbounded. The surface is therefore built for whatever the stored
        # configuration uses, and records it so the page can say so.
        configuration = ModelConfiguration.load()
        features = configuration.selected_features()
        random_state = configuration.random_state

        total = (len(training.SOLVERS) * len(C_VALUES)
                 * len(MAX_ITER_VALUES) * len(TEST_SIZE_VALUES))
        self.stdout.write(
            f'Building {total} fits '
            f'(features={",".join(features)} seed={random_state})'
        )

        started = time.perf_counter()
        done = 0
        accuracy = []

        for solver in training.SOLVERS:
            by_c = []
            for c in C_VALUES:
                by_iter = []
                for max_iter in MAX_ITER_VALUES:
                    by_size = []
                    for test_size in TEST_SIZE_VALUES:
                        _, metrics = training.train_and_evaluate({
                            'C': c,
                            'max_iter': max_iter,
                            'solver': solver,
                            'test_size': test_size,
                            'random_state': random_state,
                            'features': features,
                        })
                        # Four decimals is finer than a chart can show and
                        # keeps the payload around fifty kilobytes.
                        by_size.append(round(metrics['accuracy'], 4))
                        done += 1
                    by_iter.append(by_size)
                by_c.append(by_iter)
            accuracy.append(by_c)
            elapsed = time.perf_counter() - started
            self.stdout.write(f'  {solver:<10} {done}/{total} fits  {elapsed:.0f}s')

        OUTPUT.write_text(json.dumps({
            'axes': {
                'solver': list(training.SOLVERS),
                'C': C_VALUES,
                'max_iter': MAX_ITER_VALUES,
                'test_size': TEST_SIZE_VALUES,
            },
            'context': {'features': features, 'random_state': random_state},
            'accuracy': accuracy,
        }, separators=(',', ':')))

        size = OUTPUT.stat().st_size / 1024
        self.stdout.write(self.style.SUCCESS(
            f'Wrote {OUTPUT.name} ({size:.0f} KB) in {time.perf_counter() - started:.0f}s'
        ))
