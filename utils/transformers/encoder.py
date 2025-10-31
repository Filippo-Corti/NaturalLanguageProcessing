import torch
from torch import nn

from .sentence_embedding import SentenceEmbedding
from .feed_forward import FeedForward
from .normalization_layer import NormalizationLayer
from .multi_head_attention import MultiHeadAttention


class Encoder(nn.Module):
    """
    The Encoder of a Transformer.

    It incorporates all Encoder layers into a single nn.Module (+ the embedding).
    """

    def __init__(
            self,
            d_model: int,
            ffn_hidden: int,
            heads_count: int,
            num_layers: int,
            max_sequence_length: int,
            language_to_index: dict[str, int],
            tokenizer
    ):
        """
        Initializes the Encoder class

        :param d_model: dimension of the embeddings
        :param ffn_hidden: dimension of the hidden layer in the FFN block
        :param heads_count: number of heads in the multi-head attention
        :param num_layers: number of encoder layers (N)
        :param max_sequence_length: maximum number of tokens in a sentence
        :param language_to_index: a map from each token to its index
        :param tokenizer: a tokenizer object to split sentences into tokens
        """
        super().__init__()
        self.embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, tokenizer)
        self.encoder_layers = SequentialEncoder(
            *[EncoderLayer(d_model, ffn_hidden, heads_count) for _ in range(num_layers)]
        )

    def forward(
            self,
            x: torch.Tensor,
            self_attention_mask: torch.Tensor | None = None,
            start_token: bool = True,
            end_token: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :param self_attention_mask: optional self attention mask
        :param start_token: if true, START_TOKEN is appended at the start
        :param end_token: if true, END_TOKEN is appended at the end
        :return: the new tensor x
        """
        x = self.embedding(x, start_token, end_token)
        x = self.encoder_layers(x, self_attention_mask)
        return x


class SequentialEncoder(nn.Sequential):
    """
    Custom implementation of a nn.Sequential module, which lets self attention flow across encoder layers.
    """

    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class EncoderLayer(nn.Module):
    """
    EncoderLayer is one of the blocks that form the Encoder in a Transformer.

    It is responsible for producing the encodings of the input sequence.
    """

    def __init__(
            self,
            d_model: int,
            ffn_hidden: int,
            heads_count: int
    ):
        """
        Initializes the EncoderLayer class

        :param d_model: dimension of the embeddings
        :param ffn_hidden: dimension of the hidden layer in the FFN block
        :param heads_count: number of heads in the multi-head attention
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, heads_count)
        self.norm1 = NormalizationLayer([d_model])
        self.dropout1 = nn.Dropout(0.1)

        self.ffn = FeedForward(d_model, ffn_hidden)
        self.norm2 = NormalizationLayer([d_model])
        self.dropout2 = nn.Dropout(0.1)

    def forward(
            self,
            x: torch.Tensor,
            self_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :param self_attention_mask: optional self attention mask
        :return: the new tensor x
        """
        # 1st sub-layer
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)  # Residual Connection!

        # 2nd sub-layer
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)  # Residual Connection!

        return x
