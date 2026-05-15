"""Embedding providers package.

Modules
-------
base              — :class:`EmbeddingProvider` abstract base class.
qgenie_provider   — QGenie-backed embeddings (fail-fast, no local fallback).

Typical usage::

    from src.embeddings.qgenie_provider import FallbackEmbeddings

    embedder = FallbackEmbeddings()
    vectors = await embedder.aembed(texts)

Layering
--------
``embeddings`` imports from ``src.core``, ``src.utils``, ``src.constants``,
and ``src.logger``.  It must **not** import from ``src.clustering``,
``src.pipeline``, or any higher-level package.
"""

from src.embeddings.base import EmbeddingProvider
from src.embeddings.qgenie_provider import FallbackEmbeddings, QGenieBGEM3Embedding

__all__ = [
    "EmbeddingProvider",
    "QGenieBGEM3Embedding",
    "FallbackEmbeddings",
]
