from searches import fasttext_search, preprocess, boolean, tfidf, transformer, next_words_prediction, elastic_search, \
    bert_bone

# todo: write commands for preprocesses
# preprocess.download_foods()

preprocess.normalize_foods()
preprocess.preprocess_all_searches()
bert_bone.load_pars_bert()
boolean.preprocess()
# tfidf.preprocess()
# transformer.preprocess()
# fasttext_search.preprocess()
elastic_search.preprocess()
next_words_prediction.preprocess()
