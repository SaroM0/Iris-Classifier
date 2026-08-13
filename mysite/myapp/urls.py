from django.urls import path

from myapp import views

urlpatterns = [
    path('', views.index, name="index"),
    path('predict_species/', views.predict_species, name='predict_species'),
    path('configuracion/', views.configuration, name='configuration'),
]
