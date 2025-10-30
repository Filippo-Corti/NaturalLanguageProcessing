import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is the module that computes the attention.

    Each Head is responsible for computing attention using different criteria.
    """

    def __init__(
            self,
            d_model: int,
            heads_count: int,
    ):
        """
        Initializes the MultiHeadAttention class

        :param d_model: dimension of the embeddings
        :param heads_count: the number of heads
        """
        super().__init__()
        self.d_model = d_model
        self.heads_count = heads_count

        self.head_dim = d_model // heads_count
        self.qkv_layer = nn.Linear(d_model, d_model * 3)  # Q, K and V are represented as a single layer
        self.linear = nn.Linear(d_model, d_model)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :return: the new tensor x
        """
        pass
