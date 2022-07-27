import pickle
import random

from searches.preprocess import get_all_foods
import torch
import numpy as np
from sklearn.cluster import KMeans
from searches.transformer import search_transformer_text

vecs = None
KM = None
cats = None
groups = None


def get_cluster(text):
    _, top_food_id = search_transformer_text(text)[0]
    ans = [(0, top_food_id)]
    cat = cats[groups[top_food_id]]
    random_foods = random.choices(cat, k=min(100, len(cat)))
    for random_food in random_foods:
        ans.append((0, random_food))
    return ans


def preprocess():
    global vecs, KM, cats, groups
    vecs = torch.load("searches/data/embed.pt").numpy()
    all_foods = get_all_foods()
    # KM = KMeans(n_clusters=50).fit(vecs)
    # with open("searches/data/kmean.pkl", "wb") as f:
    #     pickle.dump(KM, f)
    with open("searches/data/kmean.pkl", "rb") as f:
        KM = pickle.load(f)
    cats = {}
    for id, (i, j) in enumerate(zip(all_foods, KM.predict(vecs))):
        if j not in cats:
            cats[j] = [id]
        else:
            cats[j].append(id)
    groups = [0] * len(all_foods)
    for c in cats.keys():
        for x in cats[c]:
            groups[x] = c
        #     print(all_foods[x]['name'], end=" ")
        # print('\n------------------------------\n')
