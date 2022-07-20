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
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    return tokens_tensor, segments_tensors


def transformer_to_text(food):
    t, s = transformer_normalize(food['name'])
    yield 10, t, s

    for i in food['tags']:
        t, s = transformer_normalize(i)
        yield 7, t, s
    for i in food['ingredients']:
        t, s = transformer_normalize(i)
        yield 5, t, s

    t, s = transformer_normalize(food['Preparation'])
    yield 1, t, s
    return


def get_transformer_score(id, vec):
    if id not in vecs:
        return 0
    else:
        return cos(vec, vecs[id]).item()


def search_transformer_text(text):
    tokens_tensor, segments_tensors = transformer_normalize(text)
    search_pooler_output = model(tokens_tensor, segments_tensors)[1]
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
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
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
            for imp, tokens_tensor, segments_tensor in transformer_to_text(food):
                while l < tokens_tensor.shape[1]:
                    r = min(tokens_tensor.shape[1], l + 300)
                    tokens_tensor_part = tokens_tensor[:, l:r]
                    segments_tensor_part = segments_tensor[:, l:r]
                    # print(tokens_tensor_part.shape, segments_tensors_part.shape)
                    pooler_output = model(tokens_tensor_part, segments_tensor_part)[1]
                    if sum is None:
                        sum = pooler_output * imp
                    else:
                        sum += pooler_output * imp
                    l = r
                    cnt += imp
            vecs[id] = sum / cnt
