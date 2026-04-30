"""Cluster similarity search — pure logic, no disk I/O.

:class:`ClusterSearcher` performs cosine-similarity (+ optional hybrid SPLADE)
search over loaded centroids.  It has **no dependency on disk** — it receives
a :class:`~src.clustering.vector_store.VectorStore` and
:class:`~src.clustering.metadata_store.MetadataStore` via constructor
injection and loads index data on each call.

This eliminates the inline ``_get_hybrid_matcher()`` import workaround that
previously existed inside :class:`~src.custom_clustering.CustomEmbeddingCluster`.

Layering
--------
Imports from ``src.clustering.{vector_store,metadata_store,hybrid_matcher}``,
``src.core``, ``src.constants``, ``src.logger``, and standard library /
NumPy / sklearn.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.clustering.hybrid_matcher import HybridSPLADEMatcher
from src.clustering.metadata_store import MetadataStore
from src.clustering.vector_store import VectorStore
from src.constants import ClusterSpecificKeys
from src.core.exceptions import VectorStoreError
from src.core.interfaces import IClusterSearcher
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["ClusterSearcher"]

_DEFAULT_THRESHOLD = 0.88
_LOAD_RETRIES = 3
_RETRY_DELAY = 5  # seconds


class ClusterSearcher(IClusterSearcher):
    """Cosine + hybrid SPLADE search over cluster centroids.

    Decoupled from disk I/O — :class:`VectorStore` and :class:`MetadataStore`
    handle file operations; this class focuses only on the similarity
    computation and result extraction.

    Args:
        vector_store: Loaded :class:`VectorStore` instance.
        metadata_store: Loaded :class:`MetadataStore` instance.
        hybrid_matcher: Optional :class:`HybridSPLADEMatcher`.  When
            ``None``, a default instance is created (which falls back to pure
            cosine if the SPLADE encoder is unavailable).
        default_threshold: Default similarity threshold.  Calls can override
            this per-search via the ``similarity_threshold`` argument.

    Example:
        >>> vs = VectorStore()
        >>> ms = MetadataStore()
        >>> searcher = ClusterSearcher(vs, ms)
        >>> name, cls = searcher.search_single("quantizer", query="error text", embedding=emb)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        hybrid_matcher: Optional[HybridSPLADEMatcher] = None,
        default_threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._vs = vector_store
        self._ms = metadata_store
        self._matcher = hybrid_matcher or HybridSPLADEMatcher()
        self.default_threshold = default_threshold

    # ------------------------------------------------------------------
    # IClusterSearcher contract
    # ------------------------------------------------------------------

    def search(self, embedding: np.ndarray, cluster_type: str) -> dict | None:
        """Search for the best matching cluster for a single *embedding*.

        Args:
            embedding: Normalised query embedding, shape ``[dim]``.
            cluster_type: Test-type identifier.

        Returns:
            Dict ``{"cluster_name": str, "class": str, "score": float}``,
            or ``None`` if no match exceeds the threshold.
        """
        name, cls, score = self.search_single(
            cluster_type, query="", embedding=embedding, similarity_threshold=self.default_threshold
        )
        if name == ClusterSpecificKeys.non_grouped_key:
            return None
        return {"cluster_name": name, "class": cls, "score": score}

    def batch_search(self, embeddings: np.ndarray, cluster_type: str) -> list[dict | None]:
        """Search for the best matching cluster for each row in *embeddings*.

        Args:
            embeddings: Float32 array of shape ``[N, dim]``.
            cluster_type: Test-type identifier.

        Returns:
            List of length ``N``.  Each element is either a result dict
            (``{"cluster_name", "class", "score"}``) or ``None``.
        """
        names, classes, scores, _ = self.batch_search_full(
            cluster_type,
            queries=[""] * len(embeddings),
            embeddings=embeddings,
            similarity_threshold=self.default_threshold,
        )
        results = []
        for name, cls, score in zip(names, classes, scores):
            if name == ClusterSpecificKeys.non_grouped_key:
                results.append(None)
            else:
                results.append({"cluster_name": name, "class": cls, "score": score})
        return results

    # ------------------------------------------------------------------
    # Richer search methods used by the pipeline
    # ------------------------------------------------------------------

    def search_single(
        self,
        cluster_type: str,
        query: str,
        embedding: np.ndarray,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
    ) -> Tuple[str | int, str | float, float]:
        """Search a single query embedding against *cluster_type*.

        Args:
            cluster_type: Test-type identifier.
            query: Original (un-normalised) query text — used for SPLADE
                scoring.  Pass ``""`` to use pure cosine only.
            embedding: Pre-normalised query embedding, shape ``[dim]``.
            similarity_threshold: Minimum score to count as a match.

        Returns:
            3-tuple ``(cluster_name, class_name, score)``.
            *cluster_name* is :attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`
            and *class_name* is ``np.nan`` when no match is found.
        """
        metadata, centroids = self._load_with_retry(cluster_type)
        if metadata is None:
            return ClusterSpecificKeys.non_grouped_key, float("nan"), 0.0

        cluster_names = list(metadata.keys())
        best_idx, best_score = self._score(
            cluster_type, query, embedding, centroids, cluster_names, similarity_threshold
        )
        if best_idx < 0:
            logger.debug(f"[Searcher] No match for type={cluster_type} (best={best_score:.3f})")
            return ClusterSpecificKeys.non_grouped_key, float("nan"), best_score

        matched_name = cluster_names[best_idx]
        matched_class = metadata[matched_name].get("class", float("nan"))
        logger.info(f"[Searcher] Match: type={cluster_type} → '{matched_name}' (score={best_score:.3f})")
        return matched_name, matched_class, best_score

    def batch_search_full(
        self,
        cluster_type: str,
        queries: List[str],
        embeddings: np.ndarray,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
    ) -> Tuple[List[str | int], List[str | float], List[float], np.ndarray]:
        """Batch search for multiple queries against *cluster_type*.

        Args:
            cluster_type: Test-type identifier.
            queries: List of original query texts (for SPLADE scoring).
            embeddings: Pre-normalised embeddings, shape ``[N, dim]``.
            similarity_threshold: Minimum score to count as a match.

        Returns:
            4-tuple:
            - ``cluster_names`` — list of matched cluster names (or non_grouped_key)
            - ``class_names``   — list of matched class names (or ``nan``)
            - ``scores``        — list of best similarity scores
            - ``embeddings``    — the input embeddings array (pass-through for convenience)
        """
        n = len(queries)
        metadata, centroids = self._load_with_retry(cluster_type)
        if metadata is None:
            return (
                [ClusterSpecificKeys.non_grouped_key] * n,
                [float("nan")] * n,
                [0.0] * n,
                embeddings,
            )

        cluster_names = list(metadata.keys())

        # Hybrid batch scoring
        try:
            best_indices, best_scores = self._matcher.batch_search(
                type_=cluster_type,
                queries=queries,
                query_embeddings=embeddings,
                centroids=centroids,
                cluster_names=cluster_names,
                threshold=similarity_threshold,
            )
        except Exception as exc:
            logger.warning(f"[Searcher] Hybrid batch search failed ({exc}); falling back to cosine")
            best_indices, best_scores = self._cosine_batch(embeddings, centroids, similarity_threshold)

        result_names, result_classes = [], []
        for q, idx, score in zip(queries, best_indices, best_scores):
            if idx >= 0:
                name = cluster_names[idx]
                result_names.append(name)
                result_classes.append(metadata[name].get("class", float("nan")))
                logger.info(f"[Searcher] Batch match: '{q[:40]}' → '{name}' (score={score:.3f})")
            else:
                result_names.append(ClusterSpecificKeys.non_grouped_key)
                result_classes.append(float("nan"))
                logger.debug(f"[Searcher] Batch no match: '{q[:40]}' (best={score:.3f})")

        return result_names, result_classes, best_scores, embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_with_retry(self, cluster_type: str) -> Tuple[dict | None, np.ndarray | None]:
        """Load metadata and centroids with up to *_LOAD_RETRIES* attempts.

        Args:
            cluster_type: Test-type identifier.

        Returns:
            ``(metadata, centroids)`` or ``(None, None)`` on permanent failure.
        """
        if not self._vs.exists(cluster_type):
            logger.info(f"[Searcher] No index for type={cluster_type}")
            return None, None

        for attempt in range(_LOAD_RETRIES):
            try:
                metadata = self._ms.load(cluster_type)
                centroids = self._vs.load(cluster_type)
                if isinstance(metadata, dict) and centroids is not None:
                    return metadata, centroids
            except (VectorStoreError, Exception) as exc:
                logger.error(f"[Searcher] Load attempt {attempt + 1} failed for type={cluster_type}: {exc}")
            if attempt < _LOAD_RETRIES - 1:
                time.sleep(_RETRY_DELAY)

        logger.error(f"[Searcher] Permanently failed to load index for type={cluster_type}")
        return None, None

    def _score(
        self,
        cluster_type: str,
        query: str,
        embedding: np.ndarray,
        centroids: np.ndarray,
        cluster_names: List[str],
        threshold: float,
    ) -> Tuple[int, float]:
        """Compute the best matching cluster index and score for a single query.

        Falls back to pure cosine if the hybrid matcher raises.

        Returns:
            ``(best_idx, best_score)`` where *best_idx* is ``-1`` if no match
            exceeds *threshold*.
        """
        try:
            return self._matcher.search(
                type_=cluster_type,
                query=query,
                query_embedding=embedding,
                centroids=centroids,
                cluster_names=cluster_names,
                threshold=threshold,
            )
        except Exception as exc:
            logger.warning(f"[Searcher] Hybrid search failed ({exc}); using pure cosine")
            sims = cosine_similarity([embedding], centroids)[0]
            bi = int(np.argmax(sims)) if len(sims) > 0 else -1
            bs = float(sims[bi]) if bi >= 0 else 0.0
            return (bi if bs >= threshold else -1), bs

    @staticmethod
    def _cosine_batch(
        embeddings: np.ndarray,
        centroids: np.ndarray,
        threshold: float,
    ) -> Tuple[List[int], List[float]]:
        """Pure cosine batch fallback.

        Args:
            embeddings: Query embedding matrix ``[N, dim]``.
            centroids: Centroid matrix ``[C, dim]``.
            threshold: Match threshold.

        Returns:
            ``(best_indices, best_scores)`` lists of length ``N``.
        """
        sim_matrix = cosine_similarity(embeddings, centroids)
        indices, scores = [], []
        for row in sim_matrix:
            bi = int(np.argmax(row)) if len(row) > 0 else -1
            bs = float(row[bi]) if bi >= 0 else 0.0
            indices.append(bi if bs >= threshold else -1)
            scores.append(bs)
        return indices, scores
