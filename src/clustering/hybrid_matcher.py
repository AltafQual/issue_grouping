"""Cosine similarity scorer for cluster matching.

:class:`HybridSPLADEMatcher` performs cosine-similarity search over
pre-computed centroid embeddings.  SPLADE was previously used as a
second channel (``α·cosine + β·SPLADE``) but the per-type
``splade_centroids.npz`` files have been removed from the persistence
layer — cluster *matching* (DB lookup) now uses pure cosine exclusively.

SPLADE is still used **within** a single run by the ``splade_pregroup``
pipeline (pairwise sparse-vector similarity on the current batch), but
that path uses :class:`~src.clustering.splade_encoder.SPLADEEncoder`
directly and never touches centroid files.

Layering
--------
Imports from ``src.constants``, ``src.logger``, and standard library /
NumPy / sklearn.  No imports from ``src.helpers`` or higher-level packages.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["HybridSPLADEMatcher"]


class HybridSPLADEMatcher:
    """Pure-cosine cluster similarity scorer.

    Computes cosine similarity between query embeddings and centroid
    embeddings.  The class retains the ``HybridSPLADEMatcher`` name for
    backward compatibility with existing call sites.

    Example:
        >>> matcher = HybridSPLADEMatcher()
        >>> best_idx, score = matcher.search(
        ...     type_="quantizer",
        ...     query="error text",
        ...     query_embedding=emb,
        ...     centroids=centroids_array,
        ...     cluster_names=names,
        ...     threshold=0.82,
        ... )
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search(
        self,
        type_: str,
        query: str,
        query_embedding: np.ndarray,
        centroids: np.ndarray,
        cluster_names: List[str],
        threshold: float,
    ) -> Tuple[int, float]:
        """Find the best matching cluster for a single query via cosine similarity.

        Args:
            type_: Cluster type key (used for logging).
            query: Original query text (unused; kept for API compatibility).
            query_embedding: Pre-normalised embedding vector, shape ``[dim]``.
            centroids: Centroid matrix, shape ``[C, dim]``.
            cluster_names: Ordered list of cluster name strings.
            threshold: Minimum score to count as a match.

        Returns:
            ``(best_idx, best_score)`` where *best_idx* is ``-1`` when no
            cluster exceeds *threshold*.
        """
        scores = cosine_similarity([query_embedding], centroids)[0]

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            logger.info(
                f"[Matcher] type={type_} | cosine | " f"MATCH → '{cluster_names[best_idx]}' (score={best_score:.3f})"
            )
            return best_idx, best_score

        logger.debug(f"[Matcher] type={type_} | cosine | NO MATCH (best={best_score:.3f} < threshold={threshold})")
        return -1, best_score

    def batch_search(
        self,
        type_: str,
        queries: List[str],
        query_embeddings: np.ndarray,
        centroids: np.ndarray,
        cluster_names: List[str],
        threshold: float,
        precomputed_query_splade: Optional[object] = None,
    ) -> Tuple[List[int], List[float]]:
        """Find the best matching cluster for multiple queries via cosine similarity.

        Args:
            type_: Cluster type key.
            queries: List of original query texts (unused; kept for API compatibility).
            query_embeddings: Pre-normalised embeddings, shape ``[N, dim]``.
            centroids: Centroid matrix, shape ``[C, dim]``.
            cluster_names: Ordered list of cluster name strings.
            threshold: Minimum score to count as a match.
            precomputed_query_splade: Ignored; kept for backward-compatible call sites.

        Returns:
            ``(best_indices, best_scores)`` lists of length ``N``.
            *best_indices* elements are ``-1`` for no-match queries.
        """
        score_matrix = cosine_similarity(query_embeddings, centroids)  # (N, C)

        logger.info(
            f"[Matcher] Batch: type={type_} | cosine | " f"{len(queries)} queries × {len(cluster_names)} clusters"
        )

        best_indices, best_scores = [], []
        for row in score_matrix:
            bi = int(np.argmax(row))
            bs = float(row[bi])
            best_indices.append(bi if bs >= threshold else -1)
            best_scores.append(bs)

        matched = sum(1 for i in best_indices if i >= 0)
        logger.info(f"[Matcher] Batch results: {matched}/{len(queries)} matched (threshold={threshold})")
        return best_indices, best_scores
