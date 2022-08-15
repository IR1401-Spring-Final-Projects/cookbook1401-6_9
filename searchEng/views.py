from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from django.http import HttpResponseRedirect
from django.shortcuts import render

def index(request):
    return render(request, "searchEng.html",{"text": "", "options": [""]*5, "search": "boolean"})

def favicon(request):
    print("OHHH")
    image_data = open("static/favicon.ico", "rb").read()
    return HttpResponse(image_data, content_type="image/x-icon")