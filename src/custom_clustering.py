"""Backward-compatibility shim — use ``src.clustering.*`` instead.

The original monolithic ``CustomEmbeddingCluster`` has been split into:

- :class:`~src.clustering.vector_store.VectorStore` — centroids.npy management
- :class:`~src.clustering.metadata_store.MetadataStore` — metadata.json management
- :class:`~src.clustering.searcher.ClusterSearcher` — similarity search

This shim re-exports a thin ``CustomEmbeddingCluster`` wrapper that composes
these three classes to maintain backward compatibility for call sites that
have not yet been updated.

.. deprecated::
    Use the individual classes from ``src.clustering`` directly.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse

from src.clustering.hybrid_matcher import HybridSPLADEMatcher
from src.clustering.metadata_store import MetadataStore
from src.clustering.searcher import ClusterSearcher
from src.clustering.splade_encoder import SPLADEEncoder
from src.clustering.vector_store import VectorStore
from src.constants import ClusterSpecificKeys, DataFrameKeys, FaissConfigurations
from src.embeddings import FallbackEmbeddings
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["CustomEmbeddingCluster"]


class CustomEmbeddingCluster:
    """Thin backward-compatibility wrapper around the split clustering classes.

    New code should use :class:`~src.clustering.vector_store.VectorStore`,
    :class:`~src.clustering.metadata_store.MetadataStore`, and
    :class:`~src.clustering.searcher.ClusterSearcher` directly.

    Args:
        base_path: Root directory for the vector index.
    """

    def __init__(self, base_path: str = FaissConfigurations.base_path) -> None:
        self.base_path = base_path
        self._vs = VectorStore(base_path)
        self._ms = MetadataStore(base_path)
        self._searcher = ClusterSearcher(self._vs, self._ms, HybridSPLADEMatcher())

    # ------------------------------------------------------------------
    # Delegating wrappers — preserve old call signatures
    # ------------------------------------------------------------------

    def save_threaded(self, dataframe: pd.DataFrame, type_: Optional[str] = None, run_id: Optional[str] = None) -> None:
        """Persist cluster centroids and metadata to the on-disk vector index.

        Groups the DataFrame by the ``type`` column (or uses *type_* when given)
        and for each type:

        1. Computes per-cluster centroids from ``DataFrameKeys.embeddings_key``.
        2. Merges new clusters into the existing ``centroids.npy`` /
           ``metadata.json`` — new clusters are appended; existing clusters get
           an EMA centroid update (decay=0.9).
        3. Saves ``centroids.npy`` and ``metadata.json`` atomically.
        4. Marks *run_id* as processed in ``processed_runids.json``.

        Args:
            dataframe: Clustered DataFrame containing ``type``,
                ``DataFrameKeys.cluster_name``, ``DataFrameKeys.cluster_class``,
                and ``DataFrameKeys.embeddings_key`` columns.
            type_: Override type name.  When ``None``, the ``type`` column is
                used to discover types.
            run_id: Run identifier recorded in ``processed_runids.json``.
        """
        _EMA_DECAY = 0.9  # exponentially moving average
        exclude_labels = {
            str(ClusterSpecificKeys.non_grouped_key),
            ClusterSpecificKeys.non_grouped_key,
            "EmptyErrorLog",
            "NoErrorLog",
        }

        if type_ is not None:
            groups = [(type_, dataframe)]
        elif "type" in dataframe.columns:
            groups = list(dataframe.groupby("type"))
        elif "cluster_type" in dataframe.columns:
            groups = list(dataframe.groupby("cluster_type"))
        else:
            logger.warning("[CustomEmbeddingCluster] No 'type' column found — skipping save_threaded")
            return

        # Pre-pass: validity filter + concurrent SPLADE encoding across all types.
        # When SPLADE_API_URL is set, asyncio.gather fans out N httpx.AsyncClient
        # POSTs in parallel against the SPLADE service; in local mode each aencode
        # delegates to loop.run_in_executor(None, encode, texts), so torch inference
        # stays serialized on the single model instance — behaviourally unchanged.
        encoder = SPLADEEncoder()
        prepared: List[Tuple[str, pd.DataFrame, bool]] = []
        texts_by_type: Dict[str, List[str]] = {}
        for current_type, type_df in groups:
            valid_mask = ~type_df[DataFrameKeys.cluster_name].isin(exclude_labels)
            valid_df = type_df[valid_mask].copy()
            no_valid_rows = valid_df.empty or DataFrameKeys.embeddings_key not in valid_df.columns
            if not no_valid_rows:
                valid_df = valid_df[valid_df[DataFrameKeys.embeddings_key].notna()]
                if not valid_df.empty and DataFrameKeys.preprocessed_text_key in valid_df.columns:
                    texts_by_type[current_type] = valid_df[DataFrameKeys.preprocessed_text_key].astype(str).tolist()
            prepared.append((current_type, valid_df, no_valid_rows))

        splade_by_type: Dict[str, Optional[scipy.sparse.csr_matrix]] = {}
        if encoder.is_available and texts_by_type:

            async def _encode_all() -> Dict[str, Optional[scipy.sparse.csr_matrix]]:
                keys = list(texts_by_type.keys())
                results = await asyncio.gather(
                    *[encoder.aencode(texts_by_type[k]) for k in keys],
                    return_exceptions=True,
                )
                out: Dict[str, Optional[scipy.sparse.csr_matrix]] = {}
                for k, r in zip(keys, results):
                    if isinstance(r, Exception):
                        logger.warning(f"[save_threaded] aencode failed for type={k}: {r}")
                        out[k] = None
                    else:
                        out[k] = r
                return out

            try:
                splade_by_type = asyncio.run(_encode_all())
            except Exception as exc:
                logger.warning(f"[save_threaded] concurrent SPLADE fan-out failed: {exc}")
                splade_by_type = {}

        for current_type, valid_df, no_valid_rows in prepared:
            try:
                if no_valid_rows:
                    logger.info(f"[save_threaded] type={current_type}: no valid rows to save")
                    continue
                if valid_df.empty:
                    continue

                # Load existing state
                existing_centroids = self._vs.load(current_type)  # ndarray or None
                existing_metadata: dict = self._ms.load(current_type)
                existing_names: List[str] = list(existing_metadata.keys())

                new_centroids_list: List[np.ndarray] = (
                    list(existing_centroids) if existing_centroids is not None else []
                )
                new_metadata: dict = dict(existing_metadata)

                # Load existing SPLADE centroids (if any)
                type_dir = os.path.join(self.base_path, f"{current_type}_custom")
                splade_path = os.path.join(type_dir, "splade_centroids.npz")
                existing_splade_list: List[Optional[scipy.sparse.csr_matrix]] = []
                if os.path.exists(splade_path):
                    try:
                        existing_splade_mat = scipy.sparse.load_npz(splade_path)
                        existing_splade_list = [existing_splade_mat[i] for i in range(existing_splade_mat.shape[0])]
                    except Exception:
                        existing_splade_list = [None] * len(existing_names)
                else:
                    existing_splade_list = [None] * len(existing_names)

                # Pad to match existing centroids count
                while len(existing_splade_list) < len(new_centroids_list):
                    existing_splade_list.append(None)

                # SPLADE vectors for this type (pre-computed concurrently above).
                all_splade_vecs = splade_by_type.get(current_type)

                # Build a mapping from valid_df row index to splade vec index
                splade_idx_map = {idx: i for i, idx in enumerate(valid_df.index)}

                for cluster_name, cluster_df in valid_df.groupby(DataFrameKeys.cluster_name):
                    cluster_name_str = str(cluster_name)
                    embeddings = np.vstack(cluster_df[DataFrameKeys.embeddings_key].tolist())
                    centroid = embeddings.mean(axis=0).astype(np.float32)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm

                    # Compute SPLADE centroid for this cluster
                    splade_centroid = None
                    if all_splade_vecs is not None:
                        member_indices = [splade_idx_map[idx] for idx in cluster_df.index if idx in splade_idx_map]
                        if member_indices:
                            member_vecs = all_splade_vecs[member_indices]
                            splade_centroid = scipy.sparse.csr_matrix(member_vecs.mean(axis=0))

                    class_val = (
                        cluster_df[DataFrameKeys.cluster_class].iloc[0]
                        if DataFrameKeys.cluster_class in cluster_df.columns
                        else "sdk_issue"
                    )

                    # Build run_ids dict for this cluster in this run
                    run_entry: dict = {}
                    if run_id and "tc_uuid" in cluster_df.columns:
                        for _, row in cluster_df.iterrows():
                            tc_key = str(row.get("tc_uuid", row.name))
                            run_entry[tc_key] = {
                                k: row[k]
                                for k in ("runtime", "soc_name")
                                if k in cluster_df.columns and not pd.isna(row.get(k))
                            }

                    if cluster_name_str in existing_names:
                        # EMA update existing centroid
                        idx = existing_names.index(cluster_name_str)
                        old_c = np.array(new_centroids_list[idx], dtype=np.float32)
                        updated = _EMA_DECAY * old_c + (1.0 - _EMA_DECAY) * centroid
                        norm2 = np.linalg.norm(updated)
                        if norm2 > 0:
                            updated = updated / norm2
                        new_centroids_list[idx] = updated
                        # EMA update SPLADE centroid
                        if splade_centroid is not None:
                            old_splade = existing_splade_list[idx]
                            if old_splade is not None and old_splade.shape[1] == splade_centroid.shape[1]:
                                updated_splade = _EMA_DECAY * old_splade + (1.0 - _EMA_DECAY) * splade_centroid
                                existing_splade_list[idx] = scipy.sparse.csr_matrix(updated_splade)
                            else:
                                existing_splade_list[idx] = splade_centroid
                        # Merge run_ids into existing metadata
                        if run_id and run_entry:
                            new_metadata[cluster_name_str].setdefault("run_ids", {})[run_id] = run_entry
                    else:
                        # New cluster — append
                        new_centroids_list.append(centroid)
                        existing_splade_list.append(splade_centroid)
                        new_metadata[cluster_name_str] = {
                            "class": str(class_val) if not pd.isna(class_val) else "sdk_issue",
                            "run_ids": {run_id: run_entry} if run_id else {},
                        }

                if not new_centroids_list:
                    continue

                updated_centroids = np.vstack(new_centroids_list).astype(np.float32)
                assert len(updated_centroids) == len(new_metadata), (
                    f"[save_threaded] Invariant violated: centroids={len(updated_centroids)} "
                    f"!= metadata={len(new_metadata)} for type={current_type}"
                )

                self._vs.save(current_type, updated_centroids)
                self._ms.save(current_type, new_metadata)

                # Save SPLADE centroids — fill None entries with zero vectors
                # so the file always gets created once we have at least one SPLADE vector
                non_none_vecs = [v for v in existing_splade_list if v is not None]
                if non_none_vecs:
                    vocab_size = non_none_vecs[0].shape[1]
                    filled_splade = [
                        v if v is not None else scipy.sparse.csr_matrix((1, vocab_size)) for v in existing_splade_list
                    ]
                    if len(filled_splade) == len(new_metadata):
                        splade_matrix = scipy.sparse.vstack(filled_splade, format="csr")
                        os.makedirs(type_dir, exist_ok=True)
                        scipy.sparse.save_npz(splade_path, splade_matrix)
                        # Invalidate in-memory cache so searcher picks up new data
                        HybridSPLADEMatcher._cluster_vec_cache.pop(current_type, None)
                        logger.info(
                            f"[save_threaded] type={current_type}: saved SPLADE centroids "
                            f"({len(non_none_vecs)}/{len(filled_splade)} non-zero)"
                        )

                if run_id:
                    self._ms.mark_run_processed(current_type, run_id)

                logger.info(
                    f"[save_threaded] type={current_type}: saved {len(updated_centroids)} centroids "
                    f"({len(new_centroids_list) - len(existing_names)} new), run_id={run_id}"
                )
            except Exception as exc:
                logger.error(f"[save_threaded] type={current_type}: failed — {exc}")

    async def batch_search(
        self,
        type_: str,
        queries: Union[str, List[str]],
        similarity_threshold: float = 0.82,
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_splade_vecs=None,
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """Async batch search — delegates to :class:`ClusterSearcher`.

        Args:
            type_: Test-type identifier.
            queries: Query text(s).
            similarity_threshold: Minimum cosine score to count as a match.
            precomputed_embeddings: Optional pre-computed embedding array.
                Skips QGenie embedding call if provided.
            precomputed_splade_vecs: Optional pre-computed SPLADE sparse matrix.
                Skips SPLADE encoding if provided.
        """
        if isinstance(queries, str):
            queries = [queries]
        if precomputed_embeddings is not None:
            embeddings = self._vs.normalize(precomputed_embeddings)
        else:
            embeddings = await FallbackEmbeddings().aembed(queries)
            embeddings = self._vs.normalize(np.array(embeddings))
        names, classes, scores, embs = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._searcher.batch_search_full(
                type_,
                queries,
                embeddings,
                similarity_threshold,
                precomputed_query_splade=precomputed_splade_vecs,
            ),
        )
        return names, classes, embs

    def get_all_clusters(self, type_: str) -> Dict:
        """Return all cluster metadata for *type_*."""
        return self._ms.load(type_)

    def _search_with_embedding(
        self,
        type_: str,
        embedding: np.ndarray,
        similarity_threshold: float = 0.82,
    ) -> tuple:
        """Search for a cluster matching *embedding* for *type_*.

        Thin wrapper around :meth:`~src.clustering.searcher.ClusterSearcher.search_single`
        to satisfy the legacy call site in ``helpers.py``.

        Args:
            type_: Test-type identifier.
            embedding: Normalised query embedding, shape ``[dim]``.
            similarity_threshold: Minimum score to return a match.

        Returns:
            2-tuple ``(cluster_name, class_name)``.  Returns
            ``(ClusterSpecificKeys.non_grouped_key, nan)`` when no match.
        """
        name, cls, _ = self._searcher.search_single(
            type_,
            query="",
            embedding=embedding,
            similarity_threshold=similarity_threshold,
        )
        return name, cls

    def search(self, type_: str, text: str, similarity_threshold: float = 0.82) -> tuple:
        """Synchronous text search — embeds *text* and delegates to ClusterSearcher.

        Args:
            type_: Test-type identifier.
            text: Pre-processed error text to search for.
            similarity_threshold: Minimum cosine score to count as a match.

        Returns:
            2-tuple ``(cluster_name, class_name)``.  Returns
            ``(ClusterSpecificKeys.non_grouped_key, nan)`` when no match.
        """
        embs = FallbackEmbeddings().embed([text])
        emb = self._vs.normalize(np.array(embs))[0]
        name, cls, _ = self._searcher.search_single(
            type_, query=text, embedding=emb, similarity_threshold=similarity_threshold
        )
        return name, cls

    def _check_existing_faiss_for_type(self, type_: str) -> bool:
        return self._vs.exists(type_)
