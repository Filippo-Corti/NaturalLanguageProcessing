"""
Implementation of an Encoder-Decoder Transformer, following the paper "Attention Is All You Need".

The Transformer is the combination of:
* Encoder-Decoder Architecture
* Positional Encoding
* Attention (as the only thing we need to keep a memory)

Original paper: https://arxiv.org/abs/1706.03762
"""

from .sentence_embedding import SentenceEmbedding
from .positional_encoding import PositionalEncoding
from .normalization_layer import NormalizationLayer
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention, MultiHeadCrossAttention
from .encoder import Encoder
from .decoder import Decoder
from .transformer import Transformer

__all__ = [
    "SentenceEmbedding",
    "PositionalEncoding",
    "NormalizationLayer",
    "FeedForward",
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "Encoder",
    "Decoder",
    "Transformer",
]
