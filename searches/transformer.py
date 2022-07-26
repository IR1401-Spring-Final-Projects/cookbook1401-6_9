import tqdm
from searches import bert_bone
from searches.preprocess import get_K, get_all_foods
import torch
from torch import nn
from tqdm import tqdm

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
vecs = None
tokenizer = None
model = None
device = "cpu"


def transformer_normalize(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    return tokens_tensor, segments_tensors



def transformer_to_text_classification(food):
    t, s = transformer_normalize(food['name'])
    yield 1, t, s
    return


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
    return cos(vec, vecs[id]).item()


def search_transformer_text(text):
    print("transformer: ", text)
    tokens_tensor, segments_tensors = transformer_normalize(text)
    search_pooler_output = model(tokens_tensor, segments_tensors)[1]
    print("compute model output")
    fast_scores = cos(search_pooler_output, vecs)
    scores = [0] * len(all_foods)
    for id, food in enumerate(all_foods):
        scores[id] = (fast_scores[id], id)
    scores = list(reversed(sorted(scores)[-K:]))
    rank = []
    for score, id in scores:
        rank.append((score, id))
    return rank


def train_model():
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
                    pooler_output = model(tokens_tensor_part, segments_tensor_part)[1]
                    if sum is None:
                        sum = pooler_output * imp
                    else:
                        sum += pooler_output * imp
                    l = r
                    cnt += imp
            vecs[id] = sum / cnt
    word_vecs = torch.zeros((len(vecs), vecs[0].shape[1]))
    for i in range(len(vecs)):
        word_vecs[i] = vecs[i][0]
    torch.save(word_vecs, "searches/data/embed.pt")


def preprocess():
    global vecs, model, tokenizer, all_foods, K
    all_foods = get_all_foods()
    K = get_K()
    model = bert_bone.get_model()
    tokenizer = bert_bone.get_tokenizer()
    # train_model()
    vecs = torch.load("searches/data/embed.pt")
