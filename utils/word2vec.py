from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from .bow import Bow
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


class Word2VecDataset(Dataset):
    """
    Dataset class to iterate over a corpus of data which has been previously
    processed using SkipWords (so the pairs <central_word, surrounding_context>
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        # Input layer of the Word2Vec network
        self.in_embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        # Output layer of the Word2Vec network
        self.out_embedding = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, center_word_idx):
        # Forward = run the input through the network and return output
        center_embedding = self.in_embedding[center_word_idx]  # Get the embedding
        scores = torch.matmul(center_embedding, self.out_embedding.T)  # Turn into scores
        return scores

    def train(
            self,
            dataloader,
            n_epochs: int,
            step: float = 0.1
    ) -> list[float]:
        """Performs the training, returns the loss every (n_epochs * step) iterations"""
        loss_history = []
        loss_checkpoint = int(n_epochs * step)
        for epoch in range(n_epochs):
            total_loss = 0
            for center, context in dataloader:
                # Gradient reset
                self.optimizer.zero_grad()
                # Forward pass
                output = self(center)  # Calls self.forward(center)
                # Loss
                loss = self.criterion(output, context)
                # Backpropagation
                loss.backward()
                # Optimization
                self.optimizer.step()
                total_loss += loss.item()
            if epoch % loss_checkpoint == 0:
                loss_history.append(total_loss)
        return loss_history

    @property
    def embeddings(self):
        """The embeddings learnt by the network"""
        return self.in_embedding.data

    def predict_context_words(
            self,
            word,
            vocabulary,
            top_k=10
    ):
        # Find word index
        word_idx = vocabulary.word2idx([word])[0]
        word_tensor = torch.tensor([word_idx], dtype=torch.long)

        # Compute output of the network (apply softmax)
        with torch.no_grad():
            output_scores = self(word_tensor)  # (1, vocab_size)
        probs = F.softmax(output_scores, dim=1)  # Turns scores into a probability distribution

        # Select top-k probabilities
        top_probs, top_indices = torch.topk(probs, top_k)
        top_indices = top_indices[0].tolist()

        # Turn indices into words lists
        predicted_words = vocabulary.idx2tokens(top_indices)
        predicted_probs = top_probs[0].tolist()
        return predicted_words, predicted_probs


