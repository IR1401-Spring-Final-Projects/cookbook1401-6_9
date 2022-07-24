import ast
import json

import tqdm
from hazm import *

from searches.preprocess import get_K, get_stopwords, get_all_foods

boolian_word_accur = None
normalizer = Normalizer()
stemmer = Stemmer()
stopwords = None
K = None
all_foods = None


def boolian_normalize(text):
    text_norm = [stemmer.stem(normalizer.normalize(y)) for y in text.split() if y not in stopwords]
    return text_norm


def boolian_to_text(food):
    ans = food['name'] + " "
    for i in food['tags']:
        ans += i + " "
    for i in food['ingredients']:
        ans += i + ' '
    ans += food['Preparation']
    return boolian_normalize(ans)


def preprocess():
    global boolian_word_accur, stopwords, all_foods, K
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()
    with open("searches/data/boolean.txt", "r", encoding="utf-8") as read_file:
        boolian_word_accur = ast.literal_eval(read_file.read())

    # for id, i in enumerate(tqdm.tqdm(all_foods)):
    #     doc = boolian_to_text(i)
    #     for w in doc:
    #         if w not in boolian_word_accur:
    #             boolian_word_accur[w] = set()
    #         boolian_word_accur[w].add(id)
    # with open("searches/data/boolean.txt", "w", encoding="utf-8") as write_file:
    #     write_file.write(str(boolian_word_accur))


def get_boolian_score(id, text):
    ans = 0
    for w in text:
        if w in boolian_word_accur and id in boolian_word_accur[w]:
            ans += 1
    return ans


def search_boolian_text(text):
    print(text)
    text = boolian_normalize(text)
    scores = [0] * len(all_foods)
    for id, food in enumerate(all_foods):
        scores[id] = (get_boolian_score(id, text), id)
    scores = list(reversed(sorted(scores)[-K:]))
    rank = []
    for score, id in scores:
        rank.append((score, id))
    return rank
