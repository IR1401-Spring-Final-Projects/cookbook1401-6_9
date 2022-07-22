import tqdm
from hazm import *
import math
from searches.preprocess import get_K, get_stopwords, get_all_foods
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
import torch
from torch import nn
from tqdm import tqdm

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
device = None
vecs = None
tokenizer = None
model = None


def transformer_normalize(text):
    return text.split()


def transformer_to_text(food):
    t = transformer_normalize(food['name'])
    yield 10, t

    for i in food['tags']:
        t = transformer_normalize(i)
        yield 7, t
    for i in food['ingredients']:
        t = transformer_normalize(i)
        yield 1, t

    t = transformer_normalize(food['Preparation'])
    yield 5, t
    return


def get_transformer_score(id, vec):
    if id not in vecs:
        return 0
    else:
        return cos(vec, vecs[id]).item()


def search_transformer_text(text):
    search_pooler_output = model(**(tokenizer(text, return_tensors='pt')))[1]
    scores = [0] * len(all_foods)
    for id, food in enumerate(all_foods):
        scores[id] = (get_transformer_score(id, search_pooler_output), id)
    scores = list(reversed(sorted(scores)[-K:]))
    rank = []
    for score, id in scores:
        rank.append((score, id))
    return rank


def preprocess():
    global vecs, model, tokenizer, device, stopwords, all_foods, K
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()

    device = "cpu"
    # v3.0
    model_name = "HooshvareLab/bert-fa-base-uncased" # "HooshvareLab/bert-fa-zwnj-base"
    config = BertConfig.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    vecs = {}
    print(len(all_foods))
    with torch.no_grad():
        for id, food in enumerate(tqdm(all_foods)):
            l = 0
            cnt = 0
            sum = None
            for imp, text in transformer_to_text(food):
                while l < len(text):
                    r = min(len(text), l + 300)
                    # print(tokens_tensor_part.shape, segments_tensors_part.shape)
                    pooler_output = model(**(tokenizer(" ".join(text[l:r]), return_tensors='pt')))[1]
                    if sum is None:
                        sum = pooler_output * imp
                    else:
                        sum += pooler_output * imp
                    l = r
                    cnt += imp
            vecs[id] = sum / cnt
