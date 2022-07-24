import random

import tqdm
from hazm import *
import math

from searches import bert_bone
from searches.preprocess import get_K, get_all_foods
from transformers import BertConfig, BertTokenizer, BertModel
import torch
from torch import nn
from tqdm import tqdm

vecs = None
tokenizer = None
model = None


def classify_transformer_text(text):
    search_pooler_output = model(**(tokenizer(text, return_tensors='pt')))[1]


def evaluate(dev_set, classifier):
    classifier.eval()
    n = len(dev_set)
    feature_size = 768
    batch_x = torch.zeros(n, feature_size)
    batch_y = torch.zeros(n, dtype=torch.long)
    for i in range(100):
        r = dev_set[i]
        batch_x[i] = vecs[r]
        batch_y[i] = all_foods[r]["label"]
    output = classifier(batch_x)
    output = torch.argmax(output, 1)
    print("dev set accuracy:",(torch.sum(output == batch_y) / n).item())
    classifier.train()


def preprocess():
    global vecs, model, tokenizer, all_foods, K
    all_foods = get_all_foods()
    main_group = {}
    main_group_to_label = {'پیش غذا': 0,
                           'کیک، شیرینی و نوشیدنی': 1,
                           'غذای اصلی': 2,
                           'سلامت غذایی': 3,
                           'ساندویچ، پیتزا و ماکارونی،پاستا': 4
                           }
    vecs = torch.load("searches/data/embed.pt")
    with_label = []
    for i, food in enumerate(all_foods):
        if "main_group" in food and food["main_group"] != 'مقالات ':
            if food["main_group"] not in main_group:
                all_foods[i]["label"] = main_group_to_label[food["main_group"]]
                with_label.append(i)
    n = len(with_label)
    random.shuffle(with_label)
    training_set = with_label[:int(0.9 * n)]
    dev_set = with_label[int(0.9 * n):int(0.95 * n)]
    test_set = with_label[int(0.95 * n):]

    feature_size = 768
    classifier = torch.nn.Sequential(nn.Linear(feature_size, feature_size), nn.ReLU(), nn.Linear(feature_size, 5))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), 0.001)
    steps = 1000000
    mean = 0
    count = 0
    for step in tqdm(range(steps)):
        if step % 1000 == 3:
            evaluate(dev_set, classifier)
            print("training loss:",mean/count)
            mean=0
            count=0

        batch_x = torch.zeros(100, feature_size)
        batch_y = torch.zeros(100, dtype=torch.long)
        for i in range(100):
            r = random.randint(0, len(training_set) - 1)
            r = training_set[r]
            batch_x[i] = vecs[r]
            batch_y[i] = all_foods[r]["label"]
        output = classifier(batch_x)
        # print(output.shape, batch_y.shape, torch.max(batch_y), torch.min(batch_y))
        loss = loss_func(output, batch_y)
        loss.backward()
        optimizer.step()
        classifier.zero_grad()
        mean += loss.item()
        count += 1
        # print(loss.item())

    # model = bert_bone.get_model()
    # tokenizer = bert_bone.get_tokenizer()
