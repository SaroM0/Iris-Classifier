from django.contrib import admin

from .models import ModelConfiguration


@admin.register(ModelConfiguration)
class ModelConfigurationAdmin(admin.ModelAdmin):
    """Read/write access to the single configuration row.

    The app's own page at /myapp/configuracion/ is the intended way to edit
    this, since it explains what each setting does and reports the effect on
    accuracy. Saving here retrains too, but silently.
    """

    list_display = ('__str__', 'test_size', 'random_state', 'updated_at')

    def has_add_permission(self, request):
        # load() owns creating the single row
        return not ModelConfiguration.objects.exists()

    def has_delete_permission(self, request, obj=None):
        # Deleting it would leave the app with nothing to train from
        return False
