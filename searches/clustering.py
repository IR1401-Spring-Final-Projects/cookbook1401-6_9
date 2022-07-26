
from searches.preprocess import get_all_foods
import torch
import numpy as np
from sklearn.cluster import KMeans

vecs = None
KM = None
cats = None


def get_categories():
    return cats


def preprocess():
    global vecs, KM, cats
    vecs = torch.load("searches/data/embed.pt").numpy()
    all_foods = get_all_foods()
    KM = KMeans(n_clusters=50).fit(all_foods)
    cats = {}
    for i, j in zip(all_foods, KM.predict(vecs)):
        if j not in cats:
            cats[j] = [i['name']]
        else:
            cats[j].append(i['name'])
    for c in cats.values():
        print(', '.join(c), end='\n------------------------------\n')