"""AIRProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
import searchEng, FoodView

urlpatterns = [
    path('searchEng/', include('searchEng.urls')),
    path('', include('searchEng.urls')),
    path('foodview/', include('FoodView.urls')),
    path('admin/', admin.site.urls),
    path('show-results/', include('ShowResults.urls')),
    path('next_words/', include('NextWords.urls'))

]
