import tqdm
from hazm import *
import math

from searches import bert_bone
from searches.preprocess import get_K, get_stopwords, get_all_foods
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
import torch
from torch import nn
from tqdm import tqdm

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
vecs = None
tokenizer = None
model = None



def get_transformer_score(id, vec):
    if id not in vecs:
        return 0
    else:
        return cos(vec, vecs[id]).item()


def search_transformer_text(text):
    print("transformer: ", text)
    search_pooler_output = model(**(tokenizer(text, return_tensors='pt')))[1]
    print("compute model output")
    scores = [0] * len(all_foods)
    for id, food in enumerate(all_foods):
        scores[id] = (get_transformer_score(id, search_pooler_output), id)
    scores = list(reversed(sorted(scores)[-K:]))
    rank = []
    for score, id in scores:
        rank.append((score, id))
    return rank


def preprocess():
    global vecs, model, tokenizer, all_foods, K
    all_foods = get_all_foods()
    K = get_K()
    model = bert_bone.get_model()
    tokenizer = bert_bone.get_tokenizer()
    vecs = torch.load("searches/data/embed.pt")


# def transformer_normalize(text):
#     return text.split()
#
#
# def transformer_to_text(food):
#     t = transformer_normalize(food['name'])
#     yield 10, t
#
#     for i in food['tags']:
#         t = transformer_normalize(i)
#         yield 7, t
#     for i in food['ingredients']:
#         t = transformer_normalize(i)
#         yield 1, t
#
#     t = transformer_normalize(food['Preparation'])
#     yield 5, t
#     return
