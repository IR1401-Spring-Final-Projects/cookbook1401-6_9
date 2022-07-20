from django.shortcuts import render
from searches.preprocess import all_foods


def show_food(request, id):
    food = all_foods[id]
    return render(request, "foodView.html",
                  {"name": food['name'], "ingredients": food['ingredients'], 'Preparation': food['Preparation'],
                   "url": food["url"], "source": food["source"]})

# Create your views here.
