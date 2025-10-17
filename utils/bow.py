"""Utilities to manage wordbags
"""
from collections import defaultdict
from typing import List
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch


class Bow:
    def __init__(
            self,
            corpus: List[list],
            min_occurrences: int = 0,  # Min number of occurrences to be in the BoW
            max_occurrences: int = 500  # Max number of occurrences to be in the BoW
    ):
        self.corpus = corpus

        # Build the BoW, counting word frequencies
        self.vocabulary = defaultdict(lambda: 0)
        for words in self.corpus:
            for word in words:
                self.vocabulary[word] += 1
        self.vocabulary = [
            x
            for x, y in self.vocabulary.items()
            if min_occurrences < y < max_occurrences
        ]

        # Words are indexed (for efficiency)
        self.word2idx = dict([(w, i) for i, w in enumerate(self.vocabulary)])
        self.idx2word = dict([(i, w) for i, w in enumerate(self.vocabulary)])

    @property
    def size(self):
        return len(self.vocabulary)

    def __getitem__(self, word):
        """Returns the index of the given word"""
        return self.word2idx[word]

    def one_hot_skip_gram_dataloader(
            self,
            window: int,  # How many words before and after the target are in its context
            batch: int = 4,
            shuffle: bool = False
    ) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
        """
        Create a dataloader for one hot encoded words
        using skip-gram

        Args:
            window (int): skip-gram window
            batch (int): batch size
            shuffle (bool): if shuffle
        """
        skipgram_inputs = []
        skipgram_outputs = []
        for words in self.corpus:
            for i, key in enumerate(words):

                # Skip central words which are not in the BoW
                if key not in self.word2idx:
                    continue

                # Build the input vector of the word (only one "1")
                key_vec = np.zeros(len(self.vocabulary))
                key_vec[self[key]] = 1

                # Build the target vector of the word ("1" for each context word)
                target_vec = np.zeros(self.size)
                context = words[max(0, i - window):i + window + 1]
                for context_word in context:

                    if key == context_word: continue
                    if context_word not in self.word2idx: continue

                    target_vec[self[context_word]] = 1

                skipgram_inputs.append(key_vec)
                skipgram_outputs.append(target_vec)

        inputs = torch.Tensor(np.array(skipgram_inputs))
        outputs = torch.Tensor(np.array(skipgram_outputs))
        dataset = TensorDataset(inputs, outputs)
        data_loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
        return data_loader, inputs, outputs
