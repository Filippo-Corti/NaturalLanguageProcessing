import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate: float = 0.005):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def predict(self, line_tensor):
        with torch.no_grad():
            hidden = self.init_hidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = self(line_tensor[i], hidden)
        return output

    def train(self, category_tensor, line_tensor):
        hidden = self.init_hidden()
        self.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def run_training(self, data_generator: callable, embedding_function: callable,
                     n_iterations: int = 10000, plot_every: int = 100):
        history = []
        current_loss = 0
        run = list(range(1, n_iterations + 1))
        for i in tqdm(run):
            category, line, category_tensor, line_tensor = data_generator(embedding_function)
            output, loss = self.train(category_tensor, line_tensor)
            current_loss += loss
            if i % plot_every == 0:
                history.append(current_loss / plot_every)
                current_loss = 0
        return history


class GenerateRomanRNN(RNN):
    def __init__(self, input_size, hidden_size, output_size, learning_rate: float = 0.005):
        super(GenerateRomanRNN, self).__init__(
            input_size=input_size, hidden_size=hidden_size,
            output_size=output_size, learning_rate=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, category_tensor, line_tensor):
        hidden = self.init_hidden()
        self.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
        loss = self.criterion(output, category_tensor[i])
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def run_training(self, data_generator: callable, argument: list,
                     n_iterations: int = 10000, plot_every: int = 100):
        history = []
        current_loss = 0
        run = list(range(1, n_iterations + 1))
        for i in tqdm(run):
            category, line, category_tensor, line_tensor = data_generator(argument)
            output, loss = self.train(category_tensor, line_tensor)
            current_loss += loss
            if i % plot_every == 0:
                history.append(current_loss / plot_every)
                current_loss = 0
        return history


class FakeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate: float = 0.005):
        super(FakeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def predict(self, line_tensor):
        with torch.no_grad():
            hidden = self.init_hidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = self(line_tensor[i], hidden)
        return output

    def train(self, category_tensor, line_tensor):
        hidden = self.init_hidden()
        self.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)
        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def run_training(self, data_generator: callable, embedding_function: callable,
                     n_iterations: int = 10000, plot_every: int = 100):
        history = []
        current_loss = 0
        run = list(range(1, n_iterations + 1))
        for i in tqdm(run):
            category, line, category_tensor, line_tensor = data_generator(embedding_function)
            output, loss = self.train(category_tensor, line_tensor)
            current_loss += loss
            if i % plot_every == 0:
                history.append(current_loss / plot_every)
                current_loss = 0
        return history


class FrequencyPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate: float = 0.005):
        super(FrequencyPredictor, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input di dimensione (batch_size, input_dim)
        """
        # Passaggio attraverso il layer nascosto con attivazione ReLU
        hidden_output = self.activation(self.hidden_layer(x))
        # Passaggio attraverso il layer di output con attivazione Sigmoid
        output = self.softmax(self.output_layer(hidden_output))
        return output

    def train(self, category_tensor, line_tensor):
        self.optimizer.zero_grad()
        output = self(line_tensor)
        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def run_training(self, data_generator: callable, embedding_function: callable,
                     n_iterations: int = 10000, plot_every: int = 100):
        history = []
        current_loss = 0
        run = list(range(1, n_iterations + 1))
        for i in tqdm(run):
            category, line, category_tensor, line_tensor = data_generator(embedding_function)
            output, loss = self.train(category_tensor, line_tensor)
            current_loss += loss
            if i % plot_every == 0:
                history.append(current_loss / plot_every)
                current_loss = 0
        return history

    def predict(self, line_tensor):
        with torch.no_grad():
            output = self(line_tensor)
        return output


class GeneratorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.005):
        super(GeneratorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def train(self, input_tensor, target_tensor):
        target = target_tensor.unsqueeze_(-1)
        hidden = self.init_hidden()
        self.zero_grad()
        loss = 0
        for i in range(input_tensor.size(0)):
            output, hidden = self(input_tensor[i], hidden)
            l = self.criterion(output, target[i])
            loss += l
        loss.backward()
        self.optimizer.step()
        return output, loss.item() / input_tensor.size(0)

    def run_training(self, data_generator: callable, data: list,
                     n_iterations: int = 10000, plot_every: int = 100):
        history = []
        current_loss = 0
        run = list(range(1, n_iterations + 1))
        for i in tqdm(run):
            input_tensor, target_tensor = data_generator(data)
            output, loss = self.train(input_tensor, target_tensor)
            current_loss += loss
            if i % plot_every == 0:
                history.append(current_loss / plot_every)
                current_loss = 0
        return history


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights