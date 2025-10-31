import torch
from torch import nn

from .sentence_embedding import SentenceEmbedding
from .feed_forward import FeedForward
from .normalization_layer import NormalizationLayer
from .multi_head_attention import MultiHeadAttention, MultiHeadCrossAttention


class Decoder(nn.Module):
    """
    The Decoder of a Transformer.

    It incorporates all Decoder layers into a single nn.Module (+ the embedding).
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
        Initializes the Decoder class

        :param d_model: dimension of the embeddings
        :param ffn_hidden: dimension of the hidden layer in the FFN block
        :param heads_count: number of heads in the multi-head attention
        :param num_layers: number of decoder layers (N)
        :param max_sequence_length: maximum number of tokens in a sentence
        :param language_to_index: a map from each token to its index
        :param tokenizer: a tokenizer object to split sentences into tokens
        """
        super().__init__()
        self.embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, tokenizer)
        self.encoder_layers = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, heads_count) for _ in range(num_layers)]
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


class SequentialDecoder(nn.Sequential):
    """
    Custom implementation of a nn.Sequential module, which lets:
    * self attention
    * cross attention
    * the encoder states
    flow across decoder layers.
    """

    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, y, cross_attention_mask, self_attention_mask)
        return x


class DecoderLayer(nn.Module):
    """
    DecoderLayer is one of the blocks that form the Decoder in a Transformer.

    It is responsible for generating the next token of the sequence.
    In order to do so, it utilizes all encoding states produced by the encoder.
    """

    def __init__(
            self,
            d_model: int,
            ffn_hidden: int,
            heads_count: int
    ):
        """
        Initializes the DecoderLayer class

        :param d_model: dimension of the embeddings
        :param ffn_hidden: dimension of the hidden layer in the FFN block
        :param heads_count: number of heads in the multi-head attention
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, heads_count)
        self.norm1 = NormalizationLayer([d_model])
        self.dropout1 = nn.Dropout(0.1)

        self.cross_attention = MultiHeadCrossAttention(d_model, heads_count)
        self.norm2 = NormalizationLayer([d_model])
        self.dropout2 = nn.Dropout(0.1)

        self.ffn = FeedForward(d_model, ffn_hidden)
        self.norm3 = NormalizationLayer([d_model])
        self.dropout3 = nn.Dropout(0.1)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            self_attention_mask: torch.Tensor | None = None,
            cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the decoder tensor
        :param y: the encoder tensor
        :param self_attention_mask: optional self attention mask
        :param cross_attention_mask: optional cross attention mask
        :return: the new decoder tensor x
        """
        # 1st sub-layer
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)  # Residual Connection!

        # 2nd sub-layer
        residual_x = x.clone()
        x = self.cross_attention(y, x, mask=cross_attention_mask)  # Cross Attention (Encoder x Decoder)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)  # Residual Connection!

        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + residual_x)  # Residual Connection!

        return x
