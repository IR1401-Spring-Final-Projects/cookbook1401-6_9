from django.urls import path

from . import views

urlpatterns = [
    path('<approach>/<text>', views.index, name='index'),
]