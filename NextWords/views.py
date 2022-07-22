from django.shortcuts import render
from searches import next_words_prediction


def index(request, search, text):
    options = next_words_prediction.next_words_options(text)
    print(options)
    return render(request, "inner_word_prediction.html", {"text": text, "options": options, "search": search})
