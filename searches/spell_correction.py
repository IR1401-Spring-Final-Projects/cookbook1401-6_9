import math
from collections import Counter
from typing import List

from tqdm import tqdm

from searches.preprocess import get_all_foods


class LanguageModel:
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_
        self.total_num_tokens = 0
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()

        for food in tqdm(get_all_foods()):
            for line in food['Preparation'].split('\n'):
                last_word = None
                for word in line.strip().split():
                    self.total_num_tokens += 1
                    self.unigram_counts[word] += 1
                    if last_word is not None:
                        self.bigram_counts[(last_word, word)] += 1
                    last_word = word

    def get_count(self, unigram):
        return self.unigram_counts[unigram] + 1

    def get_unigram_p(self, unigram):
        return self.get_count(unigram) / (self.total_num_tokens + len(self.unigram_counts))

    def get_unigram_logp(self, unigram):
        return math.log(self.get_unigram_p(unigram))

    def get_bigram_logp(self, w_1, w_2):
        bigram_p = self.bigram_counts[(w_1, w_2)] / self.get_count(w_1)
        return math.log(
            self.lambda_ * self.get_unigram_p(w_2) +
            (1 - self.lambda_) * bigram_p
        )

    def get_query_logp(self, query):
        words = query.split()
        if not words:
            return 0
        s = self.get_unigram_logp(words[0])
        for word, last_word in zip(words[1:], words):
            s += self.get_bigram_logp(last_word, word)

        return s


class CandidateGenerator:
    alphabet = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیيئ'

    def __init__(self, lm, epm):
        self.lm = lm
        self.epm = epm

    def get_num_oov(self, query):
        return sum(
            1 for w in query.strip().split()
            if w not in self.lm.unigram_counts
        )

    def filter_and_yield(self, query):
        query = query.strip()
        if query and self.get_num_oov(query) == 0:
            yield query

    def do_filter_and_yield(self, edited, query):
        yield from map(
            lambda candidate: (candidate, self.epm.get_edit_logp(edited, query)),
            self.filter_and_yield(edited),
        )

    def _get_all_distance_ones_of_term(self, term):
        n = len(term)
        #  replace
        for i in range(n):
            for c in self.alphabet:
                if c != term[i]:
                    yield term[:i] + c + term[i + 1:]

        #  insert
        for i in range(0, n + 1):
            for c in self.alphabet:
                yield term[:i] + c + term[i:]

        #  transposition
        for i in range(1, n):
            yield term[:i - 1] + term[i] + term[i - 1] + term[i + 1:]

    def get_all_distance_ones_of_term(self, term):
        return set(self._get_all_distance_ones_of_term(term))

    def get_valid_distance_ones_from_term(self, term):
        for candidate in self.get_all_distance_ones_of_term(term):
            yield from self.do_filter_and_yield(candidate, term)

    def _get_candidates(self, query):
        yield from self.do_filter_and_yield(query, query)

        for candidate, p in self.get_valid_distance_ones_from_term(query):
            yield from map(
                lambda candidate: (candidate, p),
                self.filter_and_yield(candidate),
            )

        terms = query.split()
        distance_ones = [
            list(self.get_valid_distance_ones_from_term(term))
            for term in terms
        ]
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                from itertools import product
                for (first_distance_one, p1), (second_distance_one, p2) in product(distance_ones[i], distance_ones[j]):
                    edited = ' '.join(
                        [
                            *terms[:i],
                            first_distance_one,
                            *terms[i + 1:j],
                            second_distance_one,
                            *terms[j + 1:]
                        ]
                    )
                    for candidate in self.filter_and_yield(edited):
                        yield candidate, p1 + p2

    def get_candidates(self, query):
        return set(self._get_candidates(query))


class CandidateScorer:
    def __init__(self, lm, cg, mu=1.):
        self.lm = lm
        self.cg = cg
        self.mu = mu

    def get_score(self, query, log_edit_prob):
        return log_edit_prob + self.mu * self.lm.get_query_logp(query)

    def correct_spelling(self, r) -> str:
        return max(
            self.cg.get_candidates(r),
            key=lambda x: self.get_score(x[0], x[1]),
            default=(r, 0),
        )[0]


class Edit:
    INSERTION = 1
    DELETION = 2
    TRANSPOSITION = 3
    SUBSTITUTION = 4

    def __init__(self, edit_type, c1=None, c2=None):
        self.edit_type = edit_type
        self.c1 = c1
        self.c2 = c2


class UniformEditProbabilityModel:
    def __init__(self, edit_prob=0.05):
        """
        Args:
            edit_prob (float): Probability of a single edit occurring, where
                an edit is an insertion, deletion, substitution, or transposition,
                as defined by the Damerau-Levenshtein distance.
        """
        self.edit_prob = edit_prob

    def get_edit_logp(self, edited, original):
        if edited != original:
            return math.log(self.edit_prob)
        else:
            return math.log(1 - self.edit_prob)

csp: CandidateScorer = None

def preprocess():
    global csp
    lm = LanguageModel()
    epm = UniformEditProbabilityModel()
    cg = CandidateGenerator(lm, epm)
    csp = CandidateScorer(lm, cg, mu=1.0)


def correct_spelling(query) -> str:
    return csp.correct_spelling(query)


def correct_spelling_candidates(query):
    correct = correct_spelling(query)
    if correct != query:
        return correct
    return None