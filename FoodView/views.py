import json

from django.shortcuts import render
from searches.preprocess import all_foods


def get_imp(text):
    for i in range(10):
        text = text.split(str(i))[0]
    for i in ['یک', 'دو', 'سه', 'چهار']+[":"]:
        text = text.split(i)[0]
    return text.split("به")[0].rstrip().lstrip()


def get_score(scores, ingredients):
    for ingredient in ingredients:
        ingredient2 = get_imp(ingredient)
        good = False
        for text, score in scores:
            if text == ingredient2:
                good = True
                yield round(score * (10 ** 4), 2)
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
