import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding
from .utils import START_TOKEN, END_TOKEN, PADDING_TOKEN, device


class SentenceEmbedding(nn.Module):
    """
    SentenceEmbedding creates the Embeddings for the Sentences.

    It combines:
    * Traditional Token Embedding (using nn.Embedding)
    * Positional Encoding
    """

    def __init__(
            self,
            max_sequence_length: int,
            d_model: int,
            language_to_index: dict[str, int],
            tokenizer
    ):
        """
        Initializes the SentenceEmbedding class

        :param max_sequence_length: maximum number of tokens in a sentence
        :param d_model: dimension of the embeddings
        :param language_to_index: a map from each token to its index
        :param tokenizer: a tokenizer object to split sentences into tokens
        """
        super().__init__()
        self.vocabulary_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.language_to_index = language_to_index
        self.language_tokenizer = tokenizer

        self.embedding = nn.Embedding(self.vocabulary_size, d_model)
        self.positional_encoder = PositionalEncoding(max_sequence_length, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def batch_tokenize(
            self,
            batch: torch.Tensor,
            start_token: bool = True,
            end_token: bool = True
    ) -> torch.Tensor:
        """
        Tokenizes a batch of sentences

        :param batch: the list of sentences (as a tensor)
        :param start_token: if true, START_TOKEN is appended at the start
        :param end_token: if true, END_TOKEN is appended at the end
        :return: the list of tokenized sentences
        """

        def tokenize(sentence):
            # Tokenize and map to indexes
            sentence_word_ids = [self.language_to_index[token.text] for token in self.language_tokenizer(sentence)]

            # Eventually append start, padding and end tokens
            if start_token:
                sentence_word_ids.insert(0, self.language_to_index[START_TOKEN])
            if end_token:
                sentence_word_ids.append(self.language_to_index[END_TOKEN])
            for _ in range(len(sentence_word_ids), self.max_sequence_length):
                sentence_word_ids.append(self.language_to_index[PADDING_TOKEN])

            return torch.tensor(sentence_word_ids)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num]))
        tokenized = torch.stack(tokenized)
        return tokenized.to(device)

    def forward(
            self,
            x: torch.Tensor,
            start_token: bool = True,
            end_token: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :param start_token: if true, START_TOKEN is appended at the start
        :param end_token: if true, END_TOKEN is appended at the end
        :return: the new tensor x
        """
        # Token Embedding
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)

        # Positional Encoding
        pos = self.positional_encoder().to(device)

        return self.dropout(x + pos) # Concatenation of the two
