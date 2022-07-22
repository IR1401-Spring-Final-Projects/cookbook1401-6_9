from transformers import pipeline

from searches.preprocess import get_stopwords

unmasker = None
stopwords = None

def preprocess():
    global unmasker, stopwords
    unmasker = pipeline('fill-mask', model="HooshvareLab/bert-fa-base-uncased")
    stopwords = get_stopwords()

def complete_with_many_mask(last, text, mask_num):
    sentence_end = " را میتوان خورد."
    while mask_num > 0:
        if mask_num > 1:
            list = unmasker(text + " [MASK] " * mask_num + sentence_end)[0]
        else:
            list = unmasker(text + " [MASK] " * mask_num + sentence_end)
        for option in list:
            if option['token_str'] not in text.split() and option['token_str'] not in stopwords and option['token_str'] not in [".",":","!","?", "هم", "و"]:
                if last is None or not last.startswith(text + " " + option['token_str']):
                    text += " " + option['token_str']
                    break
        mask_num -= 1
    return text


def next_words_options(text):
    ans = []
    ans.append(complete_with_many_mask(None, text, 1))
    ans.append(complete_with_many_mask(ans[len(ans) - 1], text, 1))
    ans.append(complete_with_many_mask(None, text, 2))
    ans.append(complete_with_many_mask(ans[len(ans) - 1], text, 2))
    ans.append(complete_with_many_mask(None, text, 3))
    ans.append(complete_with_many_mask(ans[len(ans) - 1], text, 3))
    return ans
