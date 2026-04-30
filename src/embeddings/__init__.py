"""Embedding providers package.

Modules
-------
base              — :class:`EmbeddingProvider` abstract base class.
qgenie_provider   — QGenie-backed embeddings with fallback to local BGE-M3.
bge_provider      — Standalone local BGE-M3 embedding model.

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
from src.embeddings.bge_provider import BGEM3Embeddings
from src.embeddings.qgenie_provider import FallbackEmbeddings, QGenieBGEM3Embedding

__all__ = [
    "EmbeddingProvider",
    "BGEM3Embeddings",
    "QGenieBGEM3Embedding",
    "FallbackEmbeddings",
]
