from searches import fasttext_search, preprocess, boolean, tfidf, transformer, next_words_prediction, elastic_search, \
    bert_bone

# todo: write commands for preprocesses
# preprocess.download_foods()
print("normalize foods start")
preprocess.normalize_foods()
print("normalize foods end/ preprocess all searches start")
preprocess.preprocess_all_searches()
print("preprocess all searches end/ load bert start")
bert_bone.load_pars_bert()
print("load bert finished/ boolean preprocess start")
boolean.preprocess()
print("boolean preprocess end/ tfidf preprocess start")
tfidf.preprocess()
print("tfidf preprocess end/ transformer preprocess start")
transformer.preprocess()
print("transformer preprocess end/ fasttext preprocess start")
fasttext_search.preprocess()
print("fasttext preprocess end/ elastic search preprocess start")
# elastic_search.preprocess()
print("elastic search preprocess end/next words prediction preprocess start")
next_words_prediction.preprocess()
print("next word prediction end")