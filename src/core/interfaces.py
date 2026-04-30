"""Abstract base classes (interfaces) for all major components.

These interfaces enforce the layered architecture: every concrete
implementation in the higher-level packages (``embeddings``, ``clustering``,
``llm``, ``pipeline``, ``reports``, ``monitoring``) must satisfy the contract
defined here.

Design principles
-----------------
* **Dependency Inversion** — high-level orchestrators depend on these ABCs,
  not on concrete implementations.
* **Interface Segregation** — each ABC is small and focused; no class is forced
  to implement methods it does not need.
* **Open/Closed** — new implementations (e.g. a different vector store) can be
  plugged in without modifying the orchestration code.

Dependency rule
---------------
This module has **no imports from any other ``src.*`` sub-package**.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "IDataLoader",
    "INormalizer",
    "IVectorStore",
    "IMetadataStore",
    "IClusterSearcher",
    "ILLMClient",
    "INotifier",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class IDataLoader(abc.ABC):
    """Contract for loading raw test-result data into a DataFrame.

    Implementations include Excel file loaders and MySQL query loaders.
    The caller receives a ``pd.DataFrame`` with at least a ``result`` column
    and does not need to know the source format.
    """

    @abc.abstractmethod
    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load and return raw test data.

        Args:
            **kwargs: Source-specific parameters (e.g. ``path`` for file
                loaders, ``run_id`` for database loaders).

        Returns:
            DataFrame containing raw test records.  Must include a ``result``
            column.

        Raises:
            ValueError: If required parameters are missing.
            src.core.exceptions.DatabaseError: If a database loader fails.
        """


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


class INormalizer(abc.ABC):
    """Contract for error-log text normalisation.

    Implementations strip noise (paths, PIDs, version strings, timestamps)
    to produce a canonical form suitable for SPLADE encoding and embedding.
    All normalisation logic for error logs must live in a class that implements
    this interface — never inline in orchestration code.
    """

    @abc.abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize a single error log string.

        Args:
            text: Raw error log text.

        Returns:
            Normalised string with noise removed.

        Raises:
            src.core.exceptions.NormalizationError: On unexpected failures.
        """

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalise a list of error log strings.

        The default implementation calls :meth:`normalize` in a loop.
        Implementations may override this for vectorised performance.

        Args:
            texts: List of raw error log strings.

        Returns:
            List of normalised strings, same order and length as ``texts``.
        """
        return [self.normalize(t) for t in texts]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class IVectorStore(abc.ABC):
    """Contract for storing and loading normalised embedding centroids.

    Each *type* (e.g. ``"quantizer"``, ``"verifier"``) has its own set of
    centroids stored as a 2-D ``float32`` NumPy array (shape
    ``[N_clusters, embedding_dim]``).  Row ``i`` corresponds to cluster
    ``i`` in the associated :class:`IMetadataStore`.

    **Invariant**: the number of rows in ``centroids.npy`` must always equal
    the number of keys in ``metadata.json`` for the same type directory.
    Implementations must assert this on :meth:`save`.
    """

    @abc.abstractmethod
    def exists(self, cluster_type: str) -> bool:
        """Return True if a centroid index exists for *cluster_type*.

        Args:
            cluster_type: Test-type identifier (e.g. ``"quantizer"``).

        Returns:
            ``True`` if at least the ``centroids.npy`` file is present and
            non-empty.
        """

    @abc.abstractmethod
    def load(self, cluster_type: str) -> np.ndarray | None:
        """Load centroid matrix for *cluster_type*.

        Args:
            cluster_type: Test-type identifier.

        Returns:
            Float32 array of shape ``[N_clusters, embedding_dim]``, or
            ``None`` if no index exists for this type.

        Raises:
            src.core.exceptions.VectorStoreError: On I/O failure.
        """

    @abc.abstractmethod
    def save(self, cluster_type: str, centroids: np.ndarray) -> None:
        """Persist centroid matrix for *cluster_type*.

        Args:
            cluster_type: Test-type identifier.
            centroids: Float32 array of shape ``[N_clusters, embedding_dim]``.
                Rows must be in the same order as the associated metadata keys.

        Raises:
            src.core.exceptions.VectorStoreError: On I/O failure.
        """


# ---------------------------------------------------------------------------
# Metadata store
# ---------------------------------------------------------------------------


class IMetadataStore(abc.ABC):
    """Contract for storing cluster metadata (names, classes, run IDs, tc UUIDs).

    The metadata for each *type* is a dictionary with cluster names as keys::

        {
            "cluster_name": {
                "class": "...",
                "run_ids": {
                    "run_id": {"tc_uuid": {...}}
                }
            }
        }

    Row order in the associated :class:`IVectorStore` must match the key
    order of the dictionary returned by :meth:`load`.
    """

    @abc.abstractmethod
    def load(self, cluster_type: str) -> dict:
        """Load metadata for *cluster_type*.

        Args:
            cluster_type: Test-type identifier.

        Returns:
            Metadata dictionary (empty dict if none exists yet).

        Raises:
            src.core.exceptions.VectorStoreError: On I/O failure.
        """

    @abc.abstractmethod
    def save(self, cluster_type: str, metadata: dict) -> None:
        """Persist metadata for *cluster_type*.

        Args:
            cluster_type: Test-type identifier.
            metadata: Full metadata dictionary to write.

        Raises:
            src.core.exceptions.VectorStoreError: On I/O failure.
        """

    @abc.abstractmethod
    def get_cluster_names(self, cluster_type: str) -> list[str]:
        """Return ordered list of cluster names for *cluster_type*.

        The order must match the centroid row order in the associated
        :class:`IVectorStore`.

        Args:
            cluster_type: Test-type identifier.

        Returns:
            List of cluster name strings.
        """


# ---------------------------------------------------------------------------
# Cluster searcher
# ---------------------------------------------------------------------------


class IClusterSearcher(abc.ABC):
    """Contract for searching existing clusters for a best match.

    Implementations combine dense cosine similarity against stored centroids
    with optional sparse SPLADE scoring to find the cluster whose centroid
    is closest to a query embedding.

    Note: Implementations must be **stateless with respect to disk I/O** —
    they receive loaded centroids/metadata via the constructor or search
    call, and never read from disk directly.
    """

    @abc.abstractmethod
    def search(
        self,
        embedding: np.ndarray,
        cluster_type: str,
    ) -> dict | None:
        """Search for the best matching cluster for *embedding*.

        Args:
            embedding: Normalised query embedding vector, shape ``[dim]``.
            cluster_type: Test-type identifier used to select the index.

        Returns:
            A dict with at least ``{"cluster_name": str, "score": float}``,
            or ``None`` if no cluster exceeds the similarity threshold.
        """

    @abc.abstractmethod
    def batch_search(
        self,
        embeddings: np.ndarray,
        cluster_type: str,
    ) -> list[dict | None]:
        """Search for the best matching cluster for each row in *embeddings*.

        Args:
            embeddings: Float32 array of shape ``[N, dim]`` — one row per query.
            cluster_type: Test-type identifier.

        Returns:
            List of length ``N``.  Each element is either a result dict
            (see :meth:`search`) or ``None`` when no match was found.
        """


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class ILLMClient(abc.ABC):
    """Contract for interacting with a large language model.

    Abstracts away the underlying API (QGenie → Vertex AI Gemini) so that
    higher-level code (cluster naming, classification, deduplication) does
    not depend on QGenie SDK internals.
    """

    @abc.abstractmethod
    def generate(self, prompt: str, system_message: str = "") -> str:
        """Call the LLM synchronously and return the generated text.

        Args:
            prompt: User-turn message.
            system_message: Optional system-level instruction.

        Returns:
            Generated text response.

        Raises:
            src.core.exceptions.LLMError: When the call fails after all retries.
        """

    @abc.abstractmethod
    async def agenerate(self, prompt: str, system_message: str = "") -> str:
        """Call the LLM asynchronously and return the generated text.

        Args:
            prompt: User-turn message.
            system_message: Optional system-level instruction.

        Returns:
            Generated text response.

        Raises:
            src.core.exceptions.LLMError: When the call fails after all retries.
        """


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class INotifier(abc.ABC):
    """Contract for sending operational notifications (Teams, email, etc.).

    Implementations are injected into the monitoring layer and must not be
    created inside business-logic functions.
    """

    @abc.abstractmethod
    def send(self, subject: str, body: str, **kwargs: Any) -> None:
        """Send a notification.

        Args:
            subject: Short title or subject line.
            body: Full message body (plain text or HTML depending on
                the implementation).
            **kwargs: Implementation-specific extras (e.g. ``recipient``,
                ``card_data`` for Teams Adaptive Cards).

        Raises:
            Exception: Implementations should catch transient errors internally
                and only re-raise on permanent failures.
        """
