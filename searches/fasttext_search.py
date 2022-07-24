import tqdm
from hazm import *
import numpy as np
from numpy.linalg import norm
from searches.preprocess import get_K, get_stopwords, get_all_foods
import fasttext

normalizer = Normalizer()
stemmer = Stemmer()
stopwords = None
K = None
all_foods = None
all_food_vectors, model= None, None


def fasttext_normalize(text):
    text_norm = [stemmer.stem(normalizer.normalize(y)) for y in text.split() if y not in stopwords]
    return text_norm
    # return normalizer.normalize(text)


def fasttext_to_text(food):
    ans = ""
    ans += food['name']
    for i in food['tags']:
        ans += i + " "
    for i in food['ingredients']:
        ans += i + ' '
    ans += food['Preparation']
    ans = " ".join(ans.split())
    return fasttext_normalize(ans)


def get_fasttext_score(id, vec):
    if id not in all_food_vectors:
        return 0
    else:
        A = vec
        B = all_food_vectors[id]
        return np.dot(A, B) / (norm(A) * norm(B))


def search_fasttext_text(text):
    search_pooler_output = model.get_word_vector(text)
    scores = [0] * len(all_foods)
    for id, food in enumerate(all_foods):
        scores[id] = (get_fasttext_score(id, search_pooler_output), id)
    scores = list(reversed(sorted(scores)[-K:]))
    rank = []
    for score, id in scores:
        rank.append((score, id))
    return rank


def preprocess():
    global all_food_vectors, model , stopwords, all_foods, K
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()

    all_food_text = ''
    for index, food in enumerate(tqdm.tqdm(all_foods)):
        simple_food = fasttext_to_text(food)
        all_food_text += ' '.join(str(t) for t in simple_food)
        all_food_text += '\n\n'

    with open('data/fasttext_food_normalized.txt', 'w', encoding="utf-8") as f:
        f.write(all_food_text)

    dim = 150  # default: 100
    lr = 0.05  # default: 0.05


    train_model = False

    model = fasttext.train_unsupervised('fasttext_food_normalized.txt', model='cbow', dim=dim, lr=lr)
    model.save_model("fasttext_model_cbow.bin")

    model = fasttext.load_model("data/fasttext_model_cbow.bin")
    # model = fasttext.load_model("fasttext_model_cbow_-Preparation.bin")
    all_food_vectors = {}
    for index, food in enumerate(tqdm.tqdm(all_foods)):
        simple_food = fasttext_to_text(food)
        final_vec = np.array([0] * dim, 'float64')
        for text in simple_food:
            final_vec += model.get_word_vector(text)
        final_vec /= len(simple_food)
        all_food_vectors[index] = final_vec
