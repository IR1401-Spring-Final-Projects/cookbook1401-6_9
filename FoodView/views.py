import json

from django.shortcuts import render
from searches.preprocess import all_foods


def get_score(scores, ingredients):
    for ingredient in ingredients:
        good = False
        for text, score in scores:
            if text == ingredient:
                good = True
                yield round(score*(10**6), 1)
                break
        if not good:
            yield -1


def show_food(request, id):
    food = all_foods[id]
    with open("searches/data/link1.json", "r", encoding="utf-8") as read_file:
        scores = json.load(read_file)
    return render(request, "foodView.html",
                  {"name": food['name'],
                   "ingredients": zip(food['ingredients'], get_score(scores, food['ingredients'])),
                   'Preparation': food['Preparation'],
                   "url": food["url"], "source": food["source"]})

# Create your views here.
