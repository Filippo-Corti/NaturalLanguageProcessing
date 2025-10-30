import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding models the inputs by adding positional encodings using sine and cosine function frequencies.
    """

    def __init__(
            self,
            max_sequence_length: int,
            d_model: int,
    ):
        """
        Initializes the PositionalEncoding class

        :param max_sequence_length: maximum number of tokens in a sentence
        :param d_model: dimension of the embeddings
        """
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :return: the new tensor x
        """
        i = torch.arange(0, self.d_model, 2, dtype=torch.float).repeat_interleave(2)[:self.d_model]
        denominator = torch.pow(10000, 2 * i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        sin_cos_argument = position / denominator
        PE = torch.zeros(size=sin_cos_argument.shape)
        PE[:, 0::2] = torch.sin(sin_cos_argument[:, 0::2])
        PE[:, 1::2] = torch.cos(sin_cos_argument[:, 1::2])
        return PE
