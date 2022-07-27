from __future__ import unicode_literals
import gdown as gdown
import random
import re
import tqdm
import json
from w3lib.html import remove_tags
import json
import textwrap
from hazm import *
import requests
import codecs

all_foods = None
normalizer = None
stemmer = None
stopwords = None

K = 10


def get_stopwords():
    return stopwords


def get_all_foods():
    return all_foods


def get_K():
    return K


def good_name(name):
    bad_subs = ['سرطان', 'نگه داری', 'نگه‌داری', 'نگهداری', 'پشت پرده', 'ورزش', 'ناباروری', 'تغذیه', 'تومان', 'تومن']
    for bad_sub in bad_subs:
        if bad_sub in name:
            return False
    return True


def download_foods():
    gdown.download("https://drive.google.com/file/d/1_IoZr2WYXBfFWeMM1o-IWLfZWudpOb1h/view?usp=sharing", fuzzy=True)
    gdown.download("https://drive.google.com/file/d/1nhKKg1L7at13yepgs-mcCjt_NE-CXYx2/view?usp=sharing", fuzzy=True)


def normalize_foods():
    f = open("searches/data/WIKIpuredata.json", encoding="utf-8")
    all_wikis = [x for x in json.load(f)]
    f.close()
    for i in tqdm.tqdm(range(len(all_wikis)), leave=False):
        all_wikis[i]['Preparation'] = re.sub(
            "مواد لازم:|طرز تهیه:|طرزتهیه|روش تهیه:|دستور پخت:", "",
            remove_tags(all_wikis[i]['Preparation'])
        )
        all_wikis[i]['ingredients'] = [re.sub("=", "", remove_tags(y)) for y in all_wikis[i]['ingredients']]

    f = open("searches/data/MamyFoodpuredata.json", encoding="utf-8")
    all_mamy = [x for x in json.load(f)]
    f.close()
    all_mamy = [x for x in all_mamy if x['main_group'] not in ['مقالات', 'آموزش آشپزی تصویری', 'تزئینات غذا']]
    all_mamy = [x for x in all_mamy if good_name(x['name'])]
    for i in range(len(all_mamy)):
        all_mamy[i]['tags'] = [all_mamy[i]['main_group'], all_mamy[i]['sub_group']]
        all_mamy[i]['source'] = "mamyfood"
    all_mamy = [x for x in all_mamy if len(x['Preparation'].rstrip().lstrip()) > 5]
    for i in range(len(all_wikis)):
        all_wikis[i]['source'] = "WikiPedia"
    all_wikis = [x for x in all_wikis if len(x['Preparation'].rstrip().lstrip()) > 5]
    global all_foods
    all_foods = all_wikis + all_mamy
    print(len(all_foods))
    with open("searches/data/all_foods.json", "w", encoding="utf-8") as file:
        json.dump(all_foods, file)
    return


def preprocess_all_searches():
    global normalizer, stemmer, stopwords
    normalizer = Normalizer()
    stemmer = Stemmer()
    r = requests.get(
        "https://raw.githubusercontent.com/sobhe/hazm/master/hazm/data/stopwords.dat",
        allow_redirects=True
    )
    open('searches/data/stopwords.txt', 'wb').write(r.content)
    stopwords = [normalizer.normalize(x.strip()) for x in
                 codecs.open('searches/data/stopwords.txt', 'r', 'utf-8').readlines()]
    stopwords += ["(", ")", "،", ".", ":", "؛"]
    print(normalizer)
