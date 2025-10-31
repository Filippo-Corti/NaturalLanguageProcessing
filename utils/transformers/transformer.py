import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer is the implementation of a Transformer as for the paper "Attention Is All You Need".

    It is composed of N Encoder Layers and N Decoder Layers.
    """

    def __init__(
            self,
            d_model: int,
            ffn_hidden: int,
            heads_count: int,
            num_layers: int,
            max_sequence_length: int,
            encoder_language_to_index: dict[str, int],
            encoder_tokenizer,
            decoder_language_to_index: dict[str, int],
            decoder_tokenizer,
            final_layer_dim: int
    ):
        """
        Initializes the DecoderLayer class

        :param d_model: dimension of the embeddings
        :param ffn_hidden: dimension of the hidden layer in the FFN block
        :param heads_count: number of heads in the multi-head attention
        :param num_layers: number of encoder and decoder layers
        :param max_sequence_length: maximum number of tokens in a sentence
        :param encoder_language_to_index: a map from each token to its index, for the encoder vocabulary
        :param encoder_tokenizer: a tokenizer object to split sentences into tokens, for the encoder language
        :param decoder_language_to_index: a map from each token to its index, for the decoder vocabulary
        :param decoder_tokenizer: a tokenizer object to split sentences into tokens, for the decoder language
        :param final_layer_dim: dimension of the output linear layer, which should coincide with the size of the vocabulary of the decoder language
        """
        super().__init__()
        self.encoder = Encoder(
            d_model,
            ffn_hidden,
            heads_count,
            num_layers,
            max_sequence_length,
            encoder_language_to_index,
            encoder_tokenizer,
        )
        self.decoder = Decoder(
            d_model,
            ffn_hidden,
            heads_count,
            num_layers,
            max_sequence_length,
            decoder_language_to_index,
            decoder_tokenizer,
        )
        self.linear = nn.Linear(d_model, final_layer_dim)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            encoder_self_attention_mask: torch.Tensor | None = None,
            encoder_start_token: bool = True,
            encoder_end_token: bool = True,
            decoder_self_attention_mask: torch.Tensor | None = None,
            decoder_cross_attention_mask: torch.Tensor | None = None,
            decoder_start_token: bool = True,
            decoder_end_token: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the decoder tensor
        :param y: the encoder tensor
        :param encoder_self_attention_mask: optional self attention mask, for the encoder
        :param encoder_start_token: if true, START_TOKEN is appended at the start for the encoder
        :param encoder_end_token: if true, END_TOKEN is appended at the end for the encoder
        :param decoder_self_attention_mask: optional self attention mask, for the decoder
        :param decoder_cross_attention_mask: optional cross attention mask, for the decoder
        :param decoder_start_token: if true, START_TOKEN is appended at the start for the decoder
        :param decoder_end_token: if true, END_TOKEN is appended at the end for the decoder
        :return: the output of the decoder
        """
        x = self.encoder(
            x,
            encoder_self_attention_mask,
            encoder_start_token,
            encoder_end_token,
        )
        out = self.decoder(
            y,
            x,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
            decoder_start_token,
            decoder_end_token,
        )
        return self.linear(out)
