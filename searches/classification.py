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

from searches.transformer import transformer_normalize

trained_classifier = None
vecs = None
model = None
main_group_to_label = {'پیش غذا': 0,
                       'کیک، شیرینی و نوشیدنی': 1,
                       'غذای اصلی': 2,
                       'سلامت غذایی': 3,
                       'ساندویچ، پیتزا و ماکارونی،پاستا': 4
                       }


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn1 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024)
        )

        self.conv_bn2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )
        self.linear = nn.Linear(512, 5)

    def forward(self, x):
        out = self.conv_bn1(x)
        out = self.conv_bn2(out)
        out = self.linear(out)
        return out


def classify_transformer_text(text):
    tokens_tensor, segments_tensors = transformer_normalize(text)
    vec = model(tokens_tensor, segments_tensors)[1]
    feature_size = 768
    batch_x = torch.zeros(2, feature_size)
    batch_x[0] = vec
    batch_x[1] = vec
    output = trained_classifier(batch_x)
    output = torch.argmax(output, 1)[0].item()
    for label in main_group_to_label.keys():
        if main_group_to_label[label] == output:
            return label
    print(output, " not in set!")
    return "WTF?"


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
    print("dev set accuracy:", (torch.sum(output == batch_y) / 100).item())
    classifier.train()


def train_model(training_set, dev_set):
    feature_size = 768
    classifier = NN()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
    steps = 1000
    train_correct = 0
    train_total = 0
    train_loss = 0
    classifier.train()
    for step in tqdm(range(steps)):
        if step % 100 == 3:
            print(f'train loss: {train_loss}, train accuracy: {train_correct / train_total}')
            evaluate(dev_set, classifier)
            train_correct, train_loss, train_total = 0, 0, 0

        batch_x = torch.zeros(100, feature_size)
        batch_y = torch.zeros(100, dtype=torch.long)
        for i in range(100):
            r = random.randint(0, len(training_set) - 1)
            r = training_set[r]
            batch_x[i] = vecs[r]
            batch_y[i] = all_foods[r]["label"]
        optimizer.zero_grad()
        output = classifier(batch_x)
        loss = loss_func(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += batch_y.size(0)
        train_correct += predicted.eq(batch_y).sum().item()
    torch.save(classifier.state_dict(), "searches/data/classifier.pt")


def preprocess():
    global vecs, model, all_foods, K, trained_classifier
    all_foods = get_all_foods()
    model = bert_bone.get_model()
    vecs = torch.load("searches/data/embedclas.pt")
    with_label = []
    for i, food in enumerate(all_foods):
        if "main_group" in food and food["main_group"] != 'مقالات ':
            food["label"] = main_group_to_label[food["main_group"]]
            with_label.append(i)
    n = len(with_label)
    random.shuffle(with_label)
    training_set = with_label[:int(0.9 * n)]
    dev_set = with_label[int(0.9 * n):]
    # train_model(training_set, dev_set)
    trained_classifier = NN()
    trained_classifier.load_state_dict(torch.load("searches/data/classifier.pt"))
    evaluate(dev_set, trained_classifier)
