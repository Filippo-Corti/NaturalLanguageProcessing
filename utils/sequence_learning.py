import torch.nn as nn
import torch.optim as optim


class SequenceClassifierRNN(nn.Module):
    """Sequence Classifier built using a generic RNN """
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=None):
        super(SequenceClassifierRNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = 2 * output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, sequence_length)
        rnn_out, _ = self.rnn(embedded)  # (batch_size, sequence_length, hidden_dim)
        final_out = rnn_out[:, -1, :]  # Extract the last hidden state
        out = self.fc(final_out)
        return out


def training(model_class, dataset, loader, embedding_dim: int = 8, learning_rate: float = 0.001, epochs: int = 20):
    vocab_size = len(dataset.char_to_idx)
    output_dim = len(dataset.target_to_idx)
    model = model_class(vocab_size=vocab_size, embedding_dim=embedding_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        loss = None
        for batch_sequences, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        if loss:
            history.append(loss.item())
    return model, history
