"""
Engine C: ETM (Embedded Topic Model) for Social Issue Structure Modeling

This module implements a modified ETM that:
- Takes Qwen document embeddings as input (instead of BOW)
- Uses Qwen word embeddings for the decoder
- Outputs interpretable topic-word distributions
"""

from .etm import ETM
from .encoder import ETMEncoder
from .decoder import ETMDecoder

__all__ = ['ETM', 'ETMEncoder', 'ETMDecoder']
