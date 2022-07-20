from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from django.http import HttpResponseRedirect
from django.shortcuts import render

def index(request):
    return render(request, "searchEng.html")