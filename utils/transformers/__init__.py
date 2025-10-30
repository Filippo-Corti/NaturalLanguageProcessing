"""
Implementation of an Encoder-Decoder Transformer, following the paper "Attention Is All You Need".

The Transformer is the combination of:
* Encoder-Decoder Architecture
* Positional Encoding
* Attention (as the only thing we need to keep a memory)
"""

from .sentence_embedding import SentenceEmbedding
from .positional_encoding import PositionalEncoding
from .normalization_layer import NormalizationLayer
from .feed_forward import FeedForward

__all__ = [
    "SentenceEmbedding",
    "PositionalEncoding",
    "NormalizationLayer",
    "FeedForward",
]
