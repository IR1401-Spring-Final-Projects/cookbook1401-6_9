import tqdm
from hazm import *
import math

from searches.boolean import boolian_normalize
from searches.preprocess import get_K, get_stopwords, get_all_foods

tf_word_accur = {}
tf_word_count = {}
idf_word = {}
tfidf = {}

normalizer = Normalizer()
stemmer = Stemmer()
stopwords = None
K = None
all_foods = None


def preprocess():
    global tfidf, idf_word, tf_word_accur, tf_word_count, stopwords, all_foods, K
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()

    for id, i in enumerate(tqdm.tqdm(all_foods)):
        doc = td_to_text(i)
        tf_word_count[id] = len(doc)
        for w in doc:
            if w not in tf_word_accur:
                tf_word_accur[w] = {}
            if id not in tf_word_accur[w]:
                tf_word_accur[w][id] = 0
            tf_word_accur[w][id] += 1

    for w in tf_word_accur.keys():
        for id in tf_word_accur[w].keys():
            tf_word_accur[w][id] /= tf_word_count[id]
        idf_word[w] = 1 + math.log(len(all_foods) / len(tf_word_accur[w]))
        tfidf[w] = {}
        for id in tf_word_accur[w].keys():
            tfidf[w][id] = tf_word_accur[w][id] * idf_word[w]


def td_normalize(text):
    text_norm = [stemmer.stem(normalizer.normalize(y)) for y in text.split() if y not in stopwords]
    return text_norm


def td_to_text(food):
    ans = food['name'] + " "
    for i in food['tags']:
        ans += i + " "
    for i in food['ingredients']:
        ans += i + ' '
    ans += food['Preparation']
    return td_normalize(ans)


def get_td_score(id, text):
    ans = 0
    for w in text:
        if w in tfidf and id in tfidf[w]:
            ans += tfidf[w][id]
    return ans


def search_td_text(text):
    text = boolian_normalize(text)
    print(text)
    scores = [0] * len(all_foods)
    for id, food in enumerate(all_foods):
        scores[id] = (get_td_score(id, text), id)
    scores = list(reversed(sorted(scores)[-K:]))
    rank = []
    for score, id in scores:
        rank.append((score, id))
    return rank
