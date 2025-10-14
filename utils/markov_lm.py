import nltk
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from collections import defaultdict
from typing import List


class MarkovLM:
    """Implements a Markov LM
    """

    def __init__(self, k: int = 2, tokenizer_model: str = "dbmdz/bert-base-italian-uncased"):
        self.k = k
        self.unigram = defaultdict(lambda: 1)
        self.k_index = defaultdict(lambda: defaultdict(lambda: 1))
        self.U = float('inf')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.start_symbol = "[#S]"
        self.end_symbol = "[#E]"

    def train(self, corpus: List[str]):
        """fill if the indexes

        Args:
            corpus (List[str]): List of textual documents
        """
        for document in corpus:
            try:
                tokens = self.tokenizer.tokenize(document)  # Tokenize each document
                ngrams = nltk.ngrams(
                    tokens,
                    n=self.k,
                    pad_left=True,
                    pad_right=True,
                    left_pad_symbol=self.start_symbol,
                    right_pad_symbol=self.end_symbol
                )
                for keys in ngrams:
                    # [A, B, C, D] means A, B, C => D (increase by one the count of D's)
                    self.k_index[keys[:-1]][keys[-1]] += 1
                    for k in keys:
                        self.unigram[k] += 1  # Each token also counts as unigram
            except TypeError:
                pass

    def _pickup(self, prefix: tuple = None):
        """Picks a random word sampled based on the preceding words (prefix)"""
        if prefix is None:  # Pick a unigram
            s = pd.Series(self.unigram) / sum(self.unigram.values())  # Distribution of unigram frequencies
            return np.random.choice(s.index.values, p=s.values)
        else:
            assert len(prefix) == self.k - 1
            data = self.k_index[prefix]
            s = pd.Series(data)  # Distribution of frequencies of all the possible pickup choices
            if s.empty:
                token = self._pickup()  # Fallback to unigram if there's no possible pickup choice
            else:
                s = s / s.sum()
                token = np.random.choice(s.index.values, p=s.values)
            return token

    def generate(self, prefix: tuple = None, unigrams: bool = False, max_len: int = 2000):
        """Generates text as the continuation of the prefix"""
        text = []
        if prefix is None:  # Prefix is either a sentence or the starting prefix (start symbol k-1 times)
            prefix = tuple([self.start_symbol] * (self.k - 1))
        text.extend(prefix)
        for i in range(max_len):
            if unigrams:
                token = self._pickup()
            else:
                token = self._pickup(prefix=prefix)
            text.append(token)
            if token == self.end_symbol:
                break
            else:
                prefix = tuple(text[-(self.k - 1):])
        return text

    def log_prob(self, text: str):
        """Determines the probability associated with text (useful for classification)"""
        tokens = self.tokenizer.tokenize(text)
        log_probs = []
        ngrams = nltk.ngrams(
            tokens,
            n=self.k,
            pad_left=True,
            pad_right=True,
            left_pad_symbol=self.start_symbol,
            right_pad_symbol=self.end_symbol
        )
        for keys in ngrams:
            prefix, next_word = keys[:-1], keys[-1]
            try:
                total = sum(self.k_index[prefix].values())
                count = self.k_index[prefix][next_word]
                log_p = np.log(count / total)  # Probability of next_word coming after prefix
                log_probs.append(log_p)
            except KeyError:
                log_probs.append(0)
            except ZeroDivisionError:
                log_probs.append(0)
        return sum(log_probs)

    @staticmethod
    def read_txt(file_path: str):
        with open(file_path, 'r') as infile:
            text = infile.read()
            return text
