from django.urls import path

from . import views

urlpatterns = [
    path('<search>/<text>', views.index, name='index'),
]