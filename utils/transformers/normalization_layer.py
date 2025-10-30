import torch
from torch import nn


class NormalizationLayer(nn.Module):
    """
    NormalizationLayer applies normalization to the tensor flowing in the module. That is, it rescales the
    tensor in order for mean to be 0 and variance to be 1.

    Normalization keeps the training stable, avoiding gradient vanishing or explosion.
    """

    def __init__(
            self,
            shape: int,
            eps: float = 1e-5,
    ):
        """
        Initializes the NormalizationLayer class

        :param shape: the shape of the tensor
        :param eps: small constant to avoid division by zero
        """
        super().__init__()
        self.shape = shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Module

        :param x: the current tensor
        :return: the new tensor x
        """
        dims = [-(i + 1) for i in range(len(self.shape))]

        # Standard Scaling computation
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std

        out = self.gamma * y + self.beta
        return out
