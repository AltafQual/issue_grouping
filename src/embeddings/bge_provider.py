"""Local BGE-M3 embedding provider (offline fallback).

Provides :class:`BGEM3Embeddings`, a singleton wrapper around the
``BAAI/bge-m3`` SentenceTransformer model stored under ``models/``.

This provider is used as the **fallback** when QGenie API calls time out or
fail, ensuring the clustering pipeline can always generate embeddings even
without network access.

Layering
--------
This module imports from ``src.embeddings.base``, ``src.utils.timer``,
``src.logger``, and third-party packages.
"""

from __future__ import annotations

import asyncio
import os

from sentence_transformers import SentenceTransformer

from src.embeddings.base import EmbeddingProvider
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["BGEM3Embeddings", "load_cached_model"]


def load_cached_model(model_name: str = "BAAI/bge-m3", models_dir: str = "models") -> SentenceTransformer | None:
    """Load a locally cached SentenceTransformer model.

    Looks for the model under ``{models_dir}/models--{model_name}/snapshots/``
    (the Hugging Face cache directory format).

    Args:
        model_name: Hugging Face model identifier (e.g. ``"BAAI/bge-m3"``).
        models_dir: Path to the local models directory relative to CWD.

    Returns:
        Loaded :class:`SentenceTransformer` instance, or ``None`` if the model
        is not found locally.
    """
    try:
        model_folder_name = f"models--{model_name.replace('/', '--')}"
        model_base_path = os.path.join(os.getcwd(), models_dir, model_folder_name, "snapshots")

        if not os.path.exists(model_base_path):
            raise FileNotFoundError(f"No cached model found at {model_base_path}")

        snapshots = os.listdir(model_base_path)
        if not snapshots:
            raise FileNotFoundError(f"No snapshot folders found in {model_base_path}")

        model_path = os.path.join(model_base_path, snapshots[0])
        logger.info(f"Loading model from: {model_path}")
        return SentenceTransformer(model_path)
    except Exception as e:
        logger.error(f"Exception while loading cached model: {e}")
        return None


class BGEM3Embeddings(EmbeddingProvider):
    """Singleton local BGE-M3 embedding provider.

    Loads the ``BAAI/bge-m3`` model once from the local ``models/`` directory.
    Falls back to downloading the model if no local cache is found.

    This is a **singleton** — multiple instantiations return the same
    underlying model instance.

    Example::

        embedder = BGEM3Embeddings()
        vectors = embedder.embed(["error log text", "another error"])
    """

    _instance: BGEM3Embeddings | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> BGEM3Embeddings:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = load_cached_model()
            if cls._model is None:
                logger.info("Unable to find model locally — downloading BAAI/bge-m3")
                cls._model = SentenceTransformer("BAAI/bge-m3", cache_folder="./models")
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """The underlying SentenceTransformer model."""
        return self._model  # type: ignore[return-value]

    @execution_timer
    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string.

        Args:
            text: Query text.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.model.encode_query(text)

    @execution_timer
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of document strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self.model.encode_document(texts).tolist()

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of document strings (async interface).

        This implementation is synchronous under the hood — it calls
        :meth:`embed` directly.  The async signature is provided so that
        :class:`BGEM3Embeddings` satisfies the :class:`EmbeddingProvider`
        interface for callers that use ``await``.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, texts)
