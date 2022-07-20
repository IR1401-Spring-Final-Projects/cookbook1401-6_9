from hazm import *
from searches.preprocess import get_K, get_stopwords, get_all_foods
from elasticsearch import Elasticsearch

es = None
normalizer = Normalizer()
stemmer = Stemmer()
stopwords = None
K = None
all_foods = None


def search_text(text):
    pass


def preprocess():
    global stopwords, all_foods, K, es
    all_foods = get_all_foods()
    stopwords = get_stopwords()
    K = get_K()
    es = Elasticsearch()
