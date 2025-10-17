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


class Word2WordPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Word2WordPrediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.vectors = np.zeros((input_dim, hidden_dim))

    def forward(self, x):
        hidden = self.fc1(x)
        output = self.fc2(hidden)
        output = F.softmax(output, dim=-1)
        return output

    def train(self, data_loader, epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        history = []

        for epoch in tqdm(range(epochs), total=epochs):
            running_loss = 0.0
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            history.append(running_loss / len(data_loader))
        self._embeddings()
        return history

    def get_vector(self, word_idx: int):
        return self.vectors[word_idx]

    def _embeddings(self):
        weights_fc1 = self.fc1.weight.data.detach().numpy()
        self.vectors = weights_fc1.T


class WordEmbeddings:
    def __init__(self, words: Bow, model: Word2WordPrediction):
        self.bow = words
        self.w2w = model
        self.sigma = cosine_similarity(self.w2w.vectors, self.w2w.vectors)
        self.sim = pd.DataFrame(self.sigma,
                                index=self.bow.vocabulary,
                                columns=self.bow.vocabulary)

    def __getitem__(self, word: str):
        return self.w2w.get_vector(self.bow[word])

    def most_similar(self, word: str, topk: int = 10):
        return self.sim.loc[word].sort_values(ascending=False).head(topk)

    def predict(self, word: str, topk: int = 10):
        vector = np.zeros(self.bow.size)
        vector[self.bow.word2idx[word]] = 1
        y_pred = pd.Series(self.w2w(torch.Tensor(vector)).detach().numpy(),
                           index=self.bow.vocabulary
                           ).sort_values(ascending=False).head(topk)
        return y_pred

    def vectors(self, words: List[str]):
        return self.w2w.vectors[[self.bow[w] for w in words]]

    def analogy(self, a: str, b: str, c: str):
        positive = self.vectors([a, c]).sum(axis=0)
        negative = self.vectors([b]).sum(axis=0)
        answer = positive - negative
        sigma = cosine_similarity(np.array([answer]), self.w2w.vectors)
        i = np.argmax(sigma[0])
        return self.bow.idx2word[i], answer

    def vector_similarity(self, query: np.ndarray, topk: int = 10):
        sigma = cosine_similarity(np.array([query]), self.w2w.vectors)
        output = pd.Series(sigma[0], index=self.bow.vocabulary)
        return output.sort_values(ascending=False).head(topk)

    def search(self, positive: List[str], negative: List[str] = None,
               topk: int = 10):
        positive_v = self.vectors(positive).sum(axis=0)
        if negative is not None:
            negative_v = self.vectors(negative).sum(axis=0)
            answer_v = positive_v - negative_v
        else:
            answer_v = positive_v
        sigma = cosine_similarity(np.array([answer_v]), self.w2w.vectors)
        output = pd.Series(sigma[0], index=self.bow.vocabulary)
        return output.sort_values(ascending=False).head(topk)

    def spot_odd_one(self, words: List[str]):
        word_v = self.vectors(words=words)
        center_v = word_v.mean(axis=0)
        sigma = cosine_similarity(np.array([center_v]), word_v)
        return pd.Series(sigma[0], index=words).sort_values(ascending=True)

    def common_meanings(self, words: List[str], topk: int = 10):
        word_v = self.vectors(words=words)
        center_v = word_v.mean(axis=0)
        return self.vector_similarity(center_v, topk=topk)
