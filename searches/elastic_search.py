import json

from hazm import *
from tqdm import tqdm

from searches.preprocess import get_K, get_stopwords, get_all_foods
from elasticsearch import Elasticsearch

es = None
normalizer = Normalizer()
stemmer = Stemmer()
stopwords = None
K = None
all_foods = get_all_foods()

host = 'https://127.0.0.1:9200'
ELASTIC_PASSWORD = ''

es = Elasticsearch(
    host,
    ca_certs="http_ca.crt",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)


def search_text(text):
    res = es.search(index='test-index', query={
        "function_score": {
            "query": {
                "multi_match": {
                    "fields": ["tags", "ingredients", 'name'],
                    "query": text,
                    "fuzziness": "AUTO",
                },
            },
            "boost": "5",
            "functions": [
                {
                    "filter": {'match': {'name': text}},
                    "weight": 42
                },
                {
                    "filter": {
                        "multi_match": {
                            "fields": ["tags", "ingredients"],
                            "query": text,
                        }
                    },
                    "weight": 40
                },
            ],
            "max_boost": 42,
            "score_mode": "max",
            "boost_mode": "multiply",
            "min_score": 30
        },
    })
    all_hits = res.get("hits").get('hits')
    res = []
    for hit in all_hits:
        res.append((hit.get("_score"), int(hit.get("_id"))))

    return res


def preprocess():
    global stopwords, all_foods, K, es
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()

    all_foods = get_all_foods()

    for index, x in tqdm(enumerate(all_foods)):
        es.index(index="test-index", id=str(index), document=x)
