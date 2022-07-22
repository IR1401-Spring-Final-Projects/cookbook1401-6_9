from transformers import pipeline

unmasker = None


def preprocess():
    global unmasker
    unmasker = pipeline('fill-mask', model="HooshvareLab/bert-fa-base-uncased")


def complete_with_many_mask(text, mask_num):
    while mask_num > 0:
        if mask_num > 1:
            next_word = unmasker(text + " [MASK] " * mask_num)[0][0]['token_str']
        else:
            next_word = unmasker(text + " [MASK] " * mask_num)[0]['token_str']
        text += " " + next_word
        mask_num -= 1
    return text


def next_words_options(text):
    ans = []
    for i in range(1, 5):
        ans.append(complete_with_many_mask(text, i))
    return ans
