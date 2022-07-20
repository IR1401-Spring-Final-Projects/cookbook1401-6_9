from searches import fasttext_search, preprocess, boolean, tfidf, transformer

preprocess.download_foods()

preprocess.normalize_foods()
preprocess.preprocess_all_searches()
boolean.preprocess()
tfidf.preprocess()
# transformer.preprocess()
fasttext_search.preprocess()
# elastic_search.preprocess()
