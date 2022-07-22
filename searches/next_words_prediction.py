from transformers import pipeline

unmasker = None


def preprocess():
    global unmasker
    unmasker = pipeline('fill-mask', model="HooshvareLab/bert-fa-base-uncased")


def complete_with_many_mask(text, mask_num):
    sentence_end=" را میتوان خورد."
    while mask_num > 0:
        if mask_num > 1:
            list = unmasker(text + " [MASK] " * mask_num + sentence_end)[0]
        else:
            list = unmasker(text + " [MASK] " * mask_num + sentence_end)
        for option in list:
            if option['token_str'] not in text.split():
                text += " " + option['token_str']
                break
        mask_num -= 1
    return text


def next_words_options(text):
    ans = []
    for i in range(1, 5):
        ans.append(complete_with_many_mask(text, i))
    return ans
