import torch
from torch import nn


class FeedForward(nn.Module):
    """
    FeedForward is a 2-layer MLP.

    It adds non-linearity to the model, thanks to its activation function (ReLu).
    """

    def __init__(
            self,
            d_model: int,
            hidden_dim: int,
    ):
        """
        Initializes the FeedForward class

        :param d_model: dimension of the embeddings
        :param hidden_dim: the dimension of the vectors in the hidden layer
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :return: the new tensor x
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
