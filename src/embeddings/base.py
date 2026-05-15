"""Abstract base class for embedding providers.

Defines the :class:`EmbeddingProvider` interface that all concrete embedding
backends must implement.  Higher-level code (pipeline, clustering) depends on
this ABC rather than any specific provider.

Layering
--------
This module has **no imports from any other ``src.*`` sub-package**.
"""

from __future__ import annotations

import abc
from typing import Any

__all__ = ["EmbeddingProvider"]


class EmbeddingProvider(abc.ABC):
    """Abstract interface for generating dense vector embeddings.

    All embedding backends (e.g. QGenie API) must implement
    this interface.  Consumers depend on :class:`EmbeddingProvider` rather
    than any specific implementation, enabling easy swapping of backends.

    Batch sizes, retry logic, and fallback handling are left entirely to
    concrete implementations.
    """

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts synchronously.

        Args:
            texts: List of raw or pre-processed text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats), in the same
            order as ``texts``.

        Raises:
            src.core.exceptions.EmbeddingError: When embedding fails after
                all retries or the fallback backend also fails.
        """

    @abc.abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts asynchronously.

        Args:
            texts: List of raw or pre-processed text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats), in the same
            order as ``texts``.

        Raises:
            src.core.exceptions.EmbeddingError: When embedding fails after
                all retries or the fallback backend also fails.
        """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Langchain-compatible alias for :meth:`embed`.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector as a list of floats.
        """
        return self.embed([text])[0]
