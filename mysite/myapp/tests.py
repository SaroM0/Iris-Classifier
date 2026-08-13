from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse

from . import classifier, training
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

        self.assertGreater(metrics['accuracy'], 0.9)
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
        configuration.random_state = 7
        configuration.save()

        differences = configuration.differences_from_default()
        self.assertEqual([d['name'] for d in differences], ['random_state'])
        self.assertEqual(differences[0]['current'], 7)


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
        response = self.client.post(reverse('configuration'), {
            'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs',
            'test_size': 0.2, 'random_state': 42,
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

    def test_pages_require_login(self):
        self.client.logout()

        self.assertEqual(self.client.get(reverse('index')).status_code, 302)
        self.assertEqual(
            self.client.get(reverse('configuration')).status_code, 302)
