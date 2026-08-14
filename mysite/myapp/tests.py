from pathlib import Path
from unittest import mock

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from . import classifier, surface, training
from .models import ModelConfiguration

# A textbook setosa and a textbook virginica from the iris dataset
SETOSA = {'sepal_length': 5.1, 'sepal_width': 3.5,
          'petal_length': 1.4, 'petal_width': 0.2}
VIRGINICA = {'sepal_length': 6.9, 'sepal_width': 3.1,
             'petal_length': 5.4, 'petal_width': 2.1}


class TrainingTests(TestCase):
    """The scikit-learn layer, exercised without Django models."""

    def test_defaults_train_an_accurate_model(self):
        _, metrics = training.train_and_evaluate(training.DEFAULTS)

        # Comfortably above the 33% a coin-flip between three species would
        # reach. Not higher: the default seed is deliberately one that
        # leaves the model room to improve and to get worse.
        self.assertGreater(metrics['accuracy'], 0.8)
        self.assertTrue(metrics['converged'])
        self.assertEqual(metrics['n_train'] + metrics['n_test'], 150)

    def test_dropping_petals_makes_the_model_measurably_worse(self):
        _, baseline = training.train_and_evaluate(training.DEFAULTS)
        _, sepals_only = training.train_and_evaluate(
            dict(training.DEFAULTS,
                 features=['sepal_length', 'sepal_width']))

        self.assertLess(sepals_only['accuracy'], baseline['accuracy'])
        self.assertEqual(sepals_only['features_used'],
                         ['sepal_length', 'sepal_width'])

    def test_too_few_iterations_reports_no_convergence(self):
        _, metrics = training.train_and_evaluate(
            dict(training.DEFAULTS, max_iter=1, solver='saga'))

        self.assertFalse(metrics['converged'])

    def test_test_size_controls_the_split(self):
        _, metrics = training.train_and_evaluate(
            dict(training.DEFAULTS, test_size=0.4))

        self.assertEqual(metrics['n_test'], 60)

    def test_no_features_is_rejected(self):
        with self.assertRaises(ValueError):
            training.train_and_evaluate(dict(training.DEFAULTS, features=[]))


class ConfigurationModelTests(TestCase):
    def test_load_creates_a_single_default_row(self):
        first = ModelConfiguration.load()
        second = ModelConfiguration.load()

        self.assertEqual(first.pk, second.pk)
        self.assertEqual(ModelConfiguration.objects.count(), 1)
        self.assertTrue(first.is_default())

    def test_saving_a_second_row_overwrites_the_first(self):
        ModelConfiguration.load()
        ModelConfiguration(C=0.5).save()

        self.assertEqual(ModelConfiguration.objects.count(), 1)
        self.assertEqual(ModelConfiguration.load().C, 0.5)

    def test_deselecting_every_feature_is_rejected(self):
        configuration = ModelConfiguration.load()
        configuration.use_sepal_length = False
        configuration.use_sepal_width = False
        configuration.use_petal_length = False
        configuration.use_petal_width = False

        with self.assertRaises(ValidationError):
            configuration.save()

    def test_out_of_range_values_are_rejected(self):
        configuration = ModelConfiguration.load()
        configuration.C = -1

        with self.assertRaises(ValidationError):
            configuration.save()

    def test_fingerprint_tracks_what_changes_the_fitted_model(self):
        configuration = ModelConfiguration.load()
        before = configuration.fingerprint()

        configuration.C = 0.01
        self.assertNotEqual(configuration.fingerprint(), before)

    def test_differences_from_default_lists_only_changes(self):
        configuration = ModelConfiguration.load()
        configuration.random_state = 42
        configuration.save()

        differences = configuration.differences_from_default()
        self.assertEqual([d['name'] for d in differences], ['random_state'])
        self.assertEqual(differences[0]['current'], 42)


class ClassifierTests(TestCase):
    def test_predicts_species_by_name_not_index(self):
        self.assertEqual(classifier.predict(**SETOSA)['species'], 'setosa')
        self.assertEqual(classifier.predict(**VIRGINICA)['species'], 'virginica')

    def test_probabilities_are_ranked_and_sum_to_one(self):
        probabilities = classifier.predict(**SETOSA)['probabilities']

        self.assertEqual(probabilities[0]['name'], 'setosa')
        self.assertTrue(probabilities[0]['is_predicted'])
        self.assertAlmostEqual(
            sum(row['probability'] for row in probabilities), 1.0, places=3)

    def test_confusion_matrix_totals_the_test_set(self):
        info = classifier.model_info()
        rows = info['confusion_rows']

        self.assertEqual(len(rows), 3)
        total = sum(cell['count'] for row in rows for cell in row['cells'])
        self.assertEqual(total, info['n_test'])

    def test_cache_rebuilds_when_the_configuration_changes(self):
        baseline = classifier.model_info()['accuracy']

        configuration = ModelConfiguration.load()
        configuration.use_petal_length = False
        configuration.use_petal_width = False
        configuration.save()

        after = classifier.model_info()
        self.assertLess(after['accuracy'], baseline)
        self.assertEqual([f['key'] for f in after['ignored_fields']],
                         ['petal_length', 'petal_width'])

    def test_changed_settings_are_reported_in_words(self):
        configuration = ModelConfiguration.load()
        configuration.use_petal_length = False
        configuration.use_petal_width = False
        configuration.save()

        difference = classifier.model_info()['differences'][0]
        self.assertEqual(difference['name'], 'measurements used')
        self.assertEqual(difference['current'], 'Sepal length, Sepal width')
        # no raw Python list repr leaking onto the page
        self.assertNotIn('[', str(difference['current']))

    def test_petal_claim_is_dropped_when_petals_are_switched_off(self):
        self.assertTrue(classifier.model_info()['uses_petals'])

        configuration = ModelConfiguration.load()
        configuration.use_petal_length = False
        configuration.use_petal_width = False
        configuration.save()

        self.assertFalse(classifier.model_info()['uses_petals'])

    def test_ignored_measurements_do_not_change_the_prediction(self):
        configuration = ModelConfiguration.load()
        configuration.use_sepal_length = False
        configuration.use_sepal_width = False
        configuration.save()

        # Same petals, wildly different sepals: the answer must not move
        first = classifier.predict(sepal_length=4.3, sepal_width=2.0,
                                   petal_length=1.4, petal_width=0.2)
        second = classifier.predict(sepal_length=7.9, sepal_width=4.4,
                                    petal_length=1.4, petal_width=0.2)

        self.assertEqual(first['species'], second['species'])
        self.assertEqual(first['confidence'], second['confidence'])


class SurfaceTests(TestCase):
    """The precomputed grid behind the configuration charts.

    Every test skips when no surface has been built, because the file is a
    build artifact and a fresh checkout legitimately lacks it.
    """

    def setUp(self):
        self.data = surface._load()
        if self.data is None:
            self.skipTest('no surface built; run manage.py build_surface')

    def test_grid_has_the_shape_its_axes_declare(self):
        axes = self.data['axes']
        accuracy = self.data['accuracy']

        self.assertEqual(len(accuracy), len(axes['solver']))
        self.assertEqual(len(accuracy[0]), len(axes['C']))
        self.assertEqual(len(accuracy[0][0]), len(axes['max_iter']))
        self.assertEqual(len(accuracy[0][0][0]), len(axes['test_size']))

    def test_every_cell_is_a_plausible_accuracy(self):
        for by_c in self.data['accuracy']:
            for by_iter in by_c:
                for by_size in by_iter:
                    for value in by_size:
                        self.assertIsInstance(value, float)
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

    def test_numeric_axes_ascend(self):
        for name in ['C', 'max_iter', 'test_size']:
            values = self.data['axes'][name]
            self.assertEqual(values, sorted(values), f'{name} is out of order')

    def test_surface_agrees_with_a_real_fit(self):
        """The one that catches the grid drifting away from the model.

        A stale surface would keep drawing confident curves describing
        training code that no longer exists.
        """
        axes = self.data['axes']
        # An arbitrary interior point, away from the edges.
        picked = {'solver': 0, 'C': 8, 'max_iter': 9, 'test_size': 3}

        _, metrics = training.train_and_evaluate({
            'solver': axes['solver'][picked['solver']],
            'C': axes['C'][picked['C']],
            'max_iter': axes['max_iter'][picked['max_iter']],
            'test_size': axes['test_size'][picked['test_size']],
            'random_state': self.data['context']['random_state'],
            'features': self.data['context']['features'],
        })

        stored = self.data['accuracy'][picked['solver']][picked['C']] \
                                     [picked['max_iter']][picked['test_size']]
        self.assertAlmostEqual(metrics['accuracy'], stored, places=4)

    def test_payload_locates_the_current_configuration(self):
        configuration = ModelConfiguration.load()
        configuration.solver = 'lbfgs'
        configuration.C = self.data['axes']['C'][8]
        configuration.save()

        payload = surface.payload(configuration)
        self.assertEqual(payload['current']['solver'], 0)
        self.assertEqual(payload['current']['C'], 8)

    def test_off_grid_values_snap_to_the_nearest_point(self):
        values = self.data['axes']['C']
        configuration = ModelConfiguration.load()
        # Between two grid points, nearer the lower one.
        configuration.C = values[8] + (values[9] - values[8]) * 0.1
        configuration.save()

        self.assertEqual(surface.payload(configuration)['current']['C'], 8)

    def test_context_mismatch_is_reported(self):
        configuration = ModelConfiguration.load()
        configuration.random_state = self.data['context']['random_state'] + 1
        configuration.save()

        self.assertFalse(surface.payload(configuration)['matches_context'])

    def test_missing_surface_yields_no_charts(self):
        with mock.patch.object(surface, '_PATH', Path('/no/such/surface.json')), \
             mock.patch.dict(surface._cache, {'read': False, 'data': None}):
            self.assertIsNone(surface.payload(ModelConfiguration.load()))


class ViewTests(TestCase):
    def setUp(self):
        User.objects.create_user(username='botanist', password='herbarium-42')
        self.client.login(username='botanist', password='herbarium-42')

    def test_form_page_explains_the_model(self):
        response = self.client.get(reverse('index'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Logistic Regression')
        self.assertContains(response, 'Sepal length')

    def test_valid_measurements_render_named_species(self):
        response = self.client.post(reverse('predict_species'), SETOSA)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['result']['species'], 'setosa')
        # the old raw-dict rendering must not come back
        self.assertNotContains(response, 'predicted_species')

    def test_result_shows_the_matrix_highlighted_on_the_predicted_species(self):
        response = self.client.post(reverse('predict_species'), SETOSA)
        html = response.content.decode()

        # the matrix belongs to the result, not just to the notes below it
        self.assertContains(response, 'How often this answer is right')
        # the predicted column is marked up (the bare string also appears in
        # the stylesheet, so match the attribute rather than the class name)
        self.assertIn('class="is-highlighted"', html)
        # and it is not repeated further down the same page
        self.assertEqual(html.count('Model predicted &rarr;'), 1)

    def test_form_page_still_shows_the_plain_matrix(self):
        html = self.client.get(reverse('index')).content.decode()

        self.assertEqual(html.count('Model predicted &rarr;'), 1)
        self.assertNotIn('class="is-highlighted"', html)

    def test_invalid_measurements_return_to_form_with_error(self):
        response = self.client.post(
            reverse('predict_species'), dict(SETOSA, petal_width=99))

        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.context['form'].errors)

    def test_get_on_predict_redirects_to_the_form(self):
        self.assertRedirects(
            self.client.get(reverse('predict_species')), reverse('index'))

    def test_configuration_page_renders_current_settings(self):
        response = self.client.get(reverse('configuration'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['form'].instance.C,
                         training.DEFAULTS['C'])

    def test_saving_configuration_retrains_and_reports_the_effect(self):
        # Only the measurements change. Moving the seed at the same time
        # would reshuffle the split too, and a different split can raise
        # the score even as the model loses its strongest signal.
        response = self.client.post(reverse('configuration'), {
            'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs',
            'test_size': 0.2, 'random_state': training.DEFAULTS['random_state'],
            'use_sepal_length': 'on', 'use_sepal_width': 'on',
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context['retrain'])
        self.assertLess(response.context['retrain']['after'],
                        response.context['retrain']['before'])
        self.assertFalse(ModelConfiguration.load().use_petal_length)

    def test_saving_no_features_shows_an_error_and_keeps_the_old_model(self):
        response = self.client.post(reverse('configuration'), {
            'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs',
            'test_size': 0.2, 'random_state': 42,
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['form'].errors)
        self.assertTrue(ModelConfiguration.load().use_petal_length)

    def test_reset_restores_the_defaults(self):
        configuration = ModelConfiguration.load()
        configuration.C = 0.01
        configuration.use_petal_width = False
        configuration.save()

        self.client.post(reverse('configuration'), {'reset': '1'})

        self.assertTrue(ModelConfiguration.load().is_default())

    def test_configuration_page_embeds_the_surface_when_one_exists(self):
        if surface._load() is None:
            self.skipTest('no surface built; run manage.py build_surface')

        response = self.client.get(reverse('configuration'))
        html = response.content.decode()

        self.assertIsNotNone(response.context['surface'])
        self.assertIn('id="surface-data"', html)
        # the controls the script attaches sliders to
        self.assertIn('data-axis="C"', html)
        self.assertIn('data-axis="test_size"', html)

    def test_configuration_page_works_without_a_surface(self):
        with mock.patch.object(surface, '_PATH', Path('/no/such/surface.json')), \
             mock.patch.dict(surface._cache, {'read': False, 'data': None}):
            response = self.client.get(reverse('configuration'))

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context['surface'])
        self.assertNotContains(response, 'id="surface-data"')
        # the page itself still does its job
        self.assertContains(response, 'How it fits')

    def test_pages_require_login(self):
        self.client.logout()

        self.assertEqual(self.client.get(reverse('index')).status_code, 302)
        self.assertEqual(
            self.client.get(reverse('configuration')).status_code, 302)
