"""Hybrid cosine + SPLADE similarity scorer.

:class:`HybridSPLADEMatcher` combines:
- **Dense cosine similarity** (weight α) over pre-computed centroid embeddings
- **SPLADE sparse dot-product** (weight β) over cluster-name SPLADE vectors

Both weights are configurable via :class:`~src.constants.SPLADEConfigurations`.
When the SPLADE encoder is unavailable the class transparently falls back to
pure cosine similarity.

Cluster SPLADE vectors are computed from cluster names on first use and cached
in a class-level dict (``_cluster_vec_cache``) that persists for the process
lifetime — no disk I/O at search time.

Layering
--------
Imports from ``src.clustering.splade_encoder``, ``src.constants``,
``src.logger``, and standard library / NumPy / scipy / sklearn.
No imports from ``src.helpers`` or higher-level packages.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.clustering.splade_encoder import SPLADEEncoder
from src.constants import FaissConfigurations, SPLADEConfigurations
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["HybridSPLADEMatcher"]


class HybridSPLADEMatcher:
    """Hybrid cosine + SPLADE similarity scorer.

    Args:
        alpha: Weight for the dense cosine component (default: 0.55).
        beta: Weight for the SPLADE sparse component (default: 0.45).

    Example:
        >>> matcher = HybridSPLADEMatcher()
        >>> best_idx, score = matcher.search(
        ...     type_="quantizer",
        ...     query="error text",
        ...     query_embedding=emb,
        ...     centroids=centroids_array,
        ...     cluster_names=names,
        ...     threshold=0.88,
        ... )
    """

    # Class-level cache: {type_: csr_matrix}
    # Populated lazily from splade_centroids.npz; entries persist for process lifetime.
    _cluster_vec_cache: Dict[str, scipy.sparse.csr_matrix] = {}

    def __init__(
        self,
        alpha: float = SPLADEConfigurations.hybrid_alpha,
        beta: float = SPLADEConfigurations.hybrid_beta,
    ) -> None:
        self.alpha = alpha
        self.beta = beta

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
        """Find the best matching cluster for a single query.

        Args:
            type_: Cluster type key — used to namespace the vector cache.
            query: Original (un-normalised) query text for SPLADE scoring.
            query_embedding: Pre-normalised embedding vector, shape ``[dim]``.
            centroids: Centroid matrix, shape ``[C, dim]``.
            cluster_names: Ordered list of cluster name strings.
            threshold: Minimum score to count as a match.

        Returns:
            ``(best_idx, best_score)`` where *best_idx* is ``-1`` when no
            cluster exceeds *threshold*.
        """
        cosine_scores = cosine_similarity([query_embedding], centroids)[0]
        scores = cosine_scores
        mode = "pure cosine"

        enc = self._encoder()
        if enc is not None:
            query_vec = enc.encode_single(query)
            cluster_mat = self._cluster_matrix(type_, cluster_names, enc)
            if query_vec is not None and cluster_mat is not None:
                raw_splade = (cluster_mat @ query_vec.T).toarray().flatten()
                splade_scores = self._minmax(raw_splade)
                scores = self.alpha * cosine_scores + self.beta * splade_scores
                mode = f"hybrid(α={self.alpha},β={self.beta})"

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            logger.info(
                f"[HybridMatcher] type={type_} | {mode} | "
                f"MATCH → '{cluster_names[best_idx]}' (score={best_score:.3f})"
            )
            return best_idx, best_score

        logger.debug(
            f"[HybridMatcher] type={type_} | {mode} | " f"NO MATCH (best={best_score:.3f} < threshold={threshold})"
        )
        return -1, best_score

    def batch_search(
        self,
        type_: str,
        queries: List[str],
        query_embeddings: np.ndarray,
        centroids: np.ndarray,
        cluster_names: List[str],
        threshold: float,
        precomputed_query_splade: Optional[scipy.sparse.csr_matrix] = None,
    ) -> Tuple[List[int], List[float]]:
        """Find the best matching cluster for multiple queries at once.

        Args:
            type_: Cluster type key.
            queries: List of original query texts for SPLADE scoring.
            query_embeddings: Pre-normalised embeddings, shape ``[N, dim]``.
            centroids: Centroid matrix, shape ``[C, dim]``.
            cluster_names: Ordered list of cluster name strings.
            threshold: Minimum score to count as a match.
            precomputed_query_splade: Optional pre-computed SPLADE vectors for
                queries, shape ``(N, vocab_size)``.  Skips re-encoding if provided.

        Returns:
            ``(best_indices, best_scores)`` lists of length ``N``.
            *best_indices* elements are ``-1`` for no-match queries.
        """
        cosine_matrix = cosine_similarity(query_embeddings, centroids)  # (N, C)
        score_matrix = cosine_matrix
        mode = "pure cosine (SPLADE unavailable)"

        enc = self._encoder()
        if enc is not None:
            query_vecs = precomputed_query_splade if precomputed_query_splade is not None else enc.encode(queries)
            cluster_mat = self._cluster_matrix(type_, cluster_names, enc)
            if query_vecs is not None and cluster_mat is not None:
                raw_splade = (query_vecs @ cluster_mat.T).toarray()  # (N, C)
                splade_matrix = np.vstack([self._minmax(row) for row in raw_splade])
                score_matrix = self.alpha * cosine_matrix + self.beta * splade_matrix
                mode = f"hybrid(α={self.alpha},β={self.beta})"

        logger.info(
            f"[HybridMatcher] Batch: type={type_} | {mode} | " f"{len(queries)} queries × {len(cluster_names)} clusters"
        )

        best_indices, best_scores = [], []
        for row in score_matrix:
            bi = int(np.argmax(row))
            bs = float(row[bi])
            best_indices.append(bi if bs >= threshold else -1)
            best_scores.append(bs)

        matched = sum(1 for i in best_indices if i >= 0)
        logger.info(f"[HybridMatcher] Batch results: {matched}/{len(queries)} matched " f"(threshold={threshold})")
        return best_indices, best_scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encoder() -> Optional[SPLADEEncoder]:
        """Return the :class:`SPLADEEncoder` singleton if it is available."""
        enc = SPLADEEncoder()
        return enc if enc.is_available else None

    def _cluster_matrix(
        self,
        type_: str,
        cluster_names: List[str],
        enc: SPLADEEncoder,
    ) -> Optional[scipy.sparse.csr_matrix]:
        """Return a ``(C, vocab_size)`` CSR matrix of per-cluster SPLADE vectors.

        Loads pre-computed SPLADE centroids from ``splade_centroids.npz`` on
        disk (computed from actual error texts during ``save_threaded``).
        Falls back to ``None`` if no stored vectors exist.

        Args:
            type_: Cluster type key (namespaces the cache).
            cluster_names: Names to look up (used for count validation).
            enc: Active :class:`SPLADEEncoder` instance (unused, kept for signature compat).

        Returns:
            CSR matrix, or ``None`` if stored vectors are unavailable.
        """
        cached = HybridSPLADEMatcher._cluster_vec_cache.get(type_)
        if cached is not None:
            if cached.shape[0] == len(cluster_names):
                return cached
            # Stale cache — cluster count changed, reload
            HybridSPLADEMatcher._cluster_vec_cache.pop(type_, None)

        splade_path = os.path.join(FaissConfigurations.base_path, f"{type_}_custom", "splade_centroids.npz")
        if not os.path.exists(splade_path):
            return None

        try:
            mat = scipy.sparse.load_npz(splade_path)
            if mat.shape[0] == len(cluster_names):
                HybridSPLADEMatcher._cluster_vec_cache[type_] = mat
                return mat
            logger.warning(
                f"[HybridMatcher] splade_centroids.npz row count ({mat.shape[0]}) "
                f"!= cluster count ({len(cluster_names)}) for type={type_}, skipping SPLADE"
            )
            return None
        except Exception as exc:
            logger.warning(f"[HybridMatcher] Failed to load splade_centroids.npz for type={type_}: {exc}")
            return None

    @staticmethod
    def _minmax(raw: np.ndarray) -> np.ndarray:
        """Min-max normalise *raw* to ``[0, 1]``.

        Returns a zero array when the range is negligible.
        """
        lo, hi = raw.min(), raw.max()
        if hi - lo < 1e-9:
            return np.zeros_like(raw)
        return (raw - lo) / (hi - lo)
