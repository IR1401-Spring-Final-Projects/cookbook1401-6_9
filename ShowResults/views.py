from django.shortcuts import render
from searches import tfidf, boolean, transformer, fasttext_search, elastic_search, classification, clustering
from searches.preprocess import all_foods
from searches.spell_correction import correct_spelling


def index(request, approach, text):
    text = ' '.join(correct_spelling(word) for word in text.split())
    if approach == "boolean":
        rank = boolean.search_boolian_text(text)
    elif approach == "tf-idf":
        rank = tfidf.search_td_text(text)
    elif approach == "transformer":
        rank = transformer.search_transformer_text(text)
    elif approach == "fast-text":
        rank = fasttext_search.search_fasttext_text(text)
    elif approach == "elastic-search":
        rank = elastic_search.search_text(text)
    elif approach == "clustering":
        rank = clustering.get_cluster(text)
    elif approach == "classification":
        rank = classification.classify_transformer_text(text)
    else:
        raise ValueError

    if approach != "classification":
        names = [all_foods[id]['name'] for score, id in rank]
        ids = [id for score, id in rank]
    else:
        names = [rank]
        ids = [0]

    return render(request, "show_result.html", {"data": zip(names, ids), "search": text, "approach": approach})
