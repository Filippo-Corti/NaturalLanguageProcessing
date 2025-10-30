import torch
import math
from torch import nn
import torch.nn.functional as F


def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the scaled dot product attention, according to the original paper.

    :param query: the matrix Q
    :param key: the matrix K
    :param value: the matrix V
    :param mask: optional mask for hiding future values
    :return: the attention values
    """
    numerator = query @ key.transpose(-1, -2)
    denominator = math.sqrt(key.size(-1))
    scaled_attention = numerator / denominator
    if mask is not None:
        scaled_attention = scaled_attention.permute(1, 0, 2, 3) + mask
        scaled_attention = scaled_attention.permute(1, 0, 2, 3)
    softmax_attention = F.softmax(scaled_attention, dim=-1)
    return softmax_attention @ value


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is the module that computes the self attention.

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
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :param mask: optional mask for hiding future values
        :return: the new tensor x
        """
        # Separate the 3 matrices Q, K, V
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Scaled Dot-Product Attention
        values = scaled_dot_product_attention(q, k, v, mask)
        # Concat
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        # Linear
        values = self.linear(values)
        return values


class MultiHeadCrossAttention(nn.Module):
    """
    MultiHeadCrossAttention is the module that computes the cross attention.

    Each Head is responsible for computing attention using different criteria.
    """

    def __init__(
            self,
            d_model: int,
            heads_count: int,
    ):
        """
        Initializes the MultiHeadCrossAttention class

        :param d_model: dimension of the embeddings
        :param heads_count: the number of heads
        """
        super().__init__()
        self.d_model = d_model
        self.heads_count = heads_count

        self.head_dim = d_model // heads_count
        self.kv_layer = nn.Linear(d_model, d_model * 2) # Q and K come from the Encoder
        self.q_layer = nn.Linear(d_model, d_model) # V comes from the Decoder
        self.linear = nn.Linear(d_model, d_model)

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :param mask: optional mask for hiding future values
        :return: the new tensor x
        """
        batch_size, sequence_length, d_model = x.size()

        # Separate K and V
        kv = self.qkv_layer(x)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        q = self.q_layer(x)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        values = scaled_dot_product_attention(q, k, v, mask)
        # Concat
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        # Linear
        values = self.linear(values)
        return values