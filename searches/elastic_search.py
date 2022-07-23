import json

from hazm import *
from searches.preprocess import get_K, get_stopwords, get_all_foods
from elasticsearch import Elasticsearch

es = None
normalizer = Normalizer()
stemmer = Stemmer()
stopwords = None
K = None
all_foods = None

host = 'https://127.0.0.1:9200'


def search_text(text):
    pass


def preprocess():
    global stopwords, all_foods, K, es
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()
    es = Elasticsearch(
        host,
    )

    wiki_data = open('WIKIpuredata.json', 'r')
    mamy_data = open('MamyFoodpuredata.json', 'r')

    # todo: do hazm pre-process on data?
    all_wiki = [x for x in json.load(wiki_data)]
    all_mamy = [x for x in json.load(mamy_data)]

    for index, x in enumerate(all_wiki):
        es.index(index="test-wiki-index", id=index + 1, document=x)
    for index, x in enumerate(all_mamy):
        es.index(index="test-mamy-index", id=index + 1, document=x)

    # todo: for search implement: es.search(index="test-{wiki | mamy}-index", query={"match_all": {}})
