"""On-disk centroid (``centroids.npy``) management.

:class:`VectorStore` is responsible for exactly one thing: reading and writing
the ``float32`` centroid matrix for a given cluster type.  It has no knowledge
of metadata, search logic, or embeddings — those concerns live in
:mod:`~src.clustering.metadata_store` and :mod:`~src.clustering.searcher`.

Invariant
---------
The row count of the centroid matrix *must* equal the key count of the
corresponding ``metadata.json``.  :meth:`VectorStore.save` does **not** enforce
this — enforcement is the caller's responsibility (see
:class:`~src.clustering.searcher.ClusterSearcher`).

Layering
--------
Imports only from ``src.core``, ``src.constants``, ``src.logger``,
and the standard library / NumPy.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.constants import FaissConfigurations
from src.core.exceptions import VectorStoreError
from src.core.interfaces import IVectorStore
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["VectorStore"]


class VectorStore(IVectorStore):
    """Manages ``centroids.npy`` files for each cluster type.

    One instance is typically shared for the whole process.  Each cluster type
    has its own sub-directory under *base_path*::

        {base_path}/{cluster_type}_custom/centroids.npy

    Args:
        base_path: Root directory for the vector index.  Defaults to
            :attr:`~src.constants.FaissConfigurations.base_path`.

    Example:
        >>> vs = VectorStore()
        >>> centroids = vs.load("quantizer")   # shape: (N, dim) or None
        >>> vs.save("quantizer", new_centroids)
    """

    def __init__(self, base_path: str = FaissConfigurations.base_path) -> None:
        self.base_path = base_path

    # ------------------------------------------------------------------
    # IVectorStore contract
    # ------------------------------------------------------------------

    def exists(self, cluster_type: str) -> bool:
        """Return ``True`` if a non-empty centroids.npy exists for *cluster_type*.

        Args:
            cluster_type: Test-type identifier (e.g. ``"quantizer"``).

        Returns:
            ``True`` when the file exists and is non-empty.
        """
        path = self._centroid_path(cluster_type)
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def load(self, cluster_type: str) -> np.ndarray | None:
        """Load the centroid matrix for *cluster_type* from disk.

        Args:
            cluster_type: Test-type identifier.

        Returns:
            Float32 array of shape ``[N_clusters, embedding_dim]``, or
            ``None`` if no index exists yet for this type.

        Raises:
            VectorStoreError: On I/O failure.
        """
        path = self._centroid_path(cluster_type)
        if not os.path.isfile(path):
            logger.info(f"[VectorStore] No centroids found for type={cluster_type}")
            return None
        try:
            with open(path, "rb") as fh:
                centroids = np.load(fh)
            logger.debug(f"[VectorStore] Loaded centroids for type={cluster_type}: shape={centroids.shape}")
            return centroids
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to load centroids for type={cluster_type}", index_path=path, cause=exc
            ) from exc

    def save(self, cluster_type: str, centroids: np.ndarray) -> None:
        """Persist the centroid matrix for *cluster_type* to disk.

        Creates the type directory if it does not exist.

        Args:
            cluster_type: Test-type identifier.
            centroids: Float32 array of shape ``[N_clusters, embedding_dim]``.

        Raises:
            VectorStoreError: On I/O failure.
        """
        type_dir = self._type_dir(cluster_type)
        os.makedirs(type_dir, exist_ok=True)
        path = os.path.join(type_dir, "centroids.npy")
        try:
            with open(path, "wb") as fh:
                np.save(fh, centroids)
            logger.info(f"[VectorStore] Saved {len(centroids)} centroids for type={cluster_type}")
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to save centroids for type={cluster_type}", index_path=path, cause=exc
            ) from exc

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize *vectors* row-wise (in place on a copy).

        Zero-length vectors are left as-is to avoid division-by-zero.

        Args:
            vectors: Array of shape ``[N, dim]``.

        Returns:
            Normalised copy of *vectors*.
        """
        vectors = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def compute_centroids(self, df, cluster_name_col: str, embeddings_col: str, class_col: str) -> dict:
        """Compute per-cluster centroids from a clustered DataFrame.

        Args:
            df: DataFrame with cluster assignments and embeddings.
            cluster_name_col: Name of the column holding cluster labels.
            embeddings_col: Name of the column holding embedding vectors.
            class_col: Name of the column holding the cluster class label.

        Returns:
            Dict mapping cluster name →
            ``{"centroid": np.ndarray, "class": str, "tc_uuids": list}``.
        """
        centroids: dict = {}
        for cluster_name, group in df.groupby(cluster_name_col):
            embeddings = np.vstack(group[embeddings_col])
            centroid = embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[cluster_name] = {
                "centroid": centroid,
                "class": group[class_col].iloc[0],
                "tc_uuids": group["tc_uuid"].tolist() if "tc_uuid" in group.columns else [],
            }
        return centroids

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _type_dir(self, cluster_type: str) -> str:
        return os.path.join(self.base_path, f"{cluster_type}_custom")

    def _centroid_path(self, cluster_type: str) -> str:
        return os.path.join(self._type_dir(cluster_type), "centroids.npy")
