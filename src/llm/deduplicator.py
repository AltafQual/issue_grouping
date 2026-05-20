"""Near-duplicate cluster detection and merging.

:class:`Deduplicator` detects near-duplicate clusters using centroid cosine
similarity and confirms candidates with the LLM before merging.  It replaces
``detect_and_merge_near_duplicate_clusters()`` and the associated merge helper
functions from ``qgenie_llm_calls.py``.

Layering
--------
Imports from ``src.llm.client``, ``src.constants``, ``src.logger``,
``src.prompts``, and standard library / pandas / numpy.
"""

from __future__ import annotations

import asyncio
import re
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.clustering.ranker import detect_cluster_outlier, reassign_unclustered_logs
from src.constants import ClusterSpecificKeys, DataFrameKeys
from src.llm import prompts
from src.llm.client import (
    ClusteringResult,
    MergeResult,
    NearDuplicateResultList,
    QgenieModels,
    ReclusterResult,
    SubClusterVerifierFailed
)
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "Deduplicator",
    "detect_and_merge_near_duplicate_clusters",
    "qgenie_post_processing",
    "subcluster_verifier_failed",
]

_near_dup_parser = JsonOutputParser(pydantic_object=NearDuplicateResultList)
_merge_parser = JsonOutputParser(pydantic_object=MergeResult)
_cluster_parser = JsonOutputParser(pydantic_object=ClusteringResult)
_recluster_parser = JsonOutputParser(pydantic_object=ReclusterResult)
_subcluster_verifier_parser = JsonOutputParser(pydantic_object=SubClusterVerifierFailed)

_COSINE_THRESHOLD = 0.85
_PAIRS_PER_BATCH = 5
_MAX_LOGS_PER_CLUSTER = 8
_GENERIC_VERIFIER_PATTERN = re.compile(
    r"^(verifierfailedimageslist|manyverifierfailedimageslist|verifierfailedimages)$", re.IGNORECASE
)


class Deduplicator:
    """Detects and merges near-duplicate clusters using cosine similarity + LLM.

    The detection pipeline:

    1. Compute per-cluster centroid embeddings.
    2. Compute pairwise cosine similarity matrix.
    3. Collect upper-triangle pairs above *cosine_threshold*.
    4. Send batches of candidate pairs to the LLM for final confirmation.
    5. Apply confirmed merges by renaming cluster-B rows to cluster-A's name.

    Args:
        cosine_threshold: Minimum cosine similarity for a pair to be sent to
            the LLM.  Default: 0.85.
        pairs_per_batch: Number of candidate pairs per LLM call.  Default: 5.
        max_logs_per_cluster: Maximum representative logs per cluster shown to
            the LLM.  Default: 8.

    Example:
        >>> deduplicator = Deduplicator()
        >>> df = await deduplicator.detect_and_merge(df)
    """

    def __init__(
        self,
        cosine_threshold: float = _COSINE_THRESHOLD,
        pairs_per_batch: int = _PAIRS_PER_BATCH,
        max_logs_per_cluster: int = _MAX_LOGS_PER_CLUSTER,
    ) -> None:
        self.cosine_threshold = cosine_threshold
        self.pairs_per_batch = pairs_per_batch
        self.max_logs_per_cluster = max_logs_per_cluster

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect_and_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and merge near-duplicate clusters in *df*.

        Args:
            df: DataFrame with cluster assignments, embeddings, and
                preprocessed logs.

        Returns:
            Updated DataFrame with near-duplicate clusters merged.
        """
        excluded = {ClusterSpecificKeys.non_grouped_key, str(ClusterSpecificKeys.non_grouped_key)}
        valid_df = df[~df[DataFrameKeys.cluster_name].isin(excluded)].copy()
        unique_clusters = [c for c in valid_df[DataFrameKeys.cluster_name].unique() if c and str(c).strip()]

        if len(unique_clusters) <= 2:
            logger.info("[Deduplicator] Fewer than 2 clusters — skipping near-duplicate check.")
            return df

        cluster_centroids, representative_logs = self._build_cluster_data(valid_df, unique_clusters)
        clusters_with_centroids = list(cluster_centroids.keys())

        if len(clusters_with_centroids) <= 2:
            logger.info("[Deduplicator] Not enough clusters with embeddings — skipping.")
            return df

        candidate_pairs = self._find_candidate_pairs(clusters_with_centroids, cluster_centroids)
        if not candidate_pairs:
            logger.info(f"[Deduplicator] No candidate pairs above threshold {self.cosine_threshold}.")
            return df

        logger.info(f"[Deduplicator] {len(candidate_pairs)} candidate pairs to review.")
        rename_map = await self._llm_confirm_merges(candidate_pairs, representative_logs)

        if not rename_map:
            return df

        df = df.copy()
        for old_name, new_name in rename_map.items():
            df.loc[df[DataFrameKeys.cluster_name] == old_name, DataFrameKeys.cluster_name] = new_name
            logger.info(f"[Deduplicator] Merged '{old_name}' → '{new_name}'")

        return df

    async def merge_two_clusters(
        self,
        df: pd.DataFrame,
        cluster_name_a: str,
        cluster_name_b: str,
    ) -> dict:
        """Ask the LLM to merge two clusters by name and return the result.

        Args:
            df: DataFrame containing both clusters.
            cluster_name_a: Name of the first cluster.
            cluster_name_b: Name of the second cluster.

        Returns:
            Dict with ``"merged_name"`` (str) and ``"outlier_indices"`` (list[int]).
        """
        df_a = df[df[DataFrameKeys.cluster_name] == cluster_name_a]
        df_b = df[df[DataFrameKeys.cluster_name] == cluster_name_b]

        logs_a = [
            {"index": int(i), "error log": row[DataFrameKeys.preprocessed_text_key]} for i, row in df_a.iterrows()
        ]
        logs_b = [
            {"index": int(i), "error log": row[DataFrameKeys.preprocessed_text_key]} for i, row in df_b.iterrows()
        ]

        chain = ChatPromptTemplate.from_template(prompts.MERGE_PROMPT_TEMPLATE) | QgenieModels.azure_o3 | _merge_parser
        chunks_a = list(self._chunked(logs_a, 50, 3))
        chunks_b = list(self._chunked(logs_b, 50, 3))

        tasks = [
            chain.ainvoke({"id_a": cluster_name_a, "logs_a": ca, "id_b": cluster_name_b, "logs_b": cb})
            for ca in chunks_a
            for cb in chunks_b
        ]
        responses = await asyncio.gather(*tasks)

        name = responses[0].get("merged_name", "") if responses else ""
        indices: set[int] = set()
        for r in responses:
            indices.update(int(i) for i in r.get("outlier_indices", []))

        return {"merged_name": name, "outlier_indices": sorted(indices)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_cluster_data(
        self, df: pd.DataFrame, cluster_names: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        """Compute centroids and gather representative logs per cluster.

        Args:
            df: DataFrame filtered to valid cluster rows.
            cluster_names: Ordered list of cluster names.

        Returns:
            ``(cluster_centroids, representative_logs)`` dicts.
        """
        centroids: dict[str, np.ndarray] = {}
        rep_logs: dict[str, list[str]] = {}

        for name in cluster_names:
            rows = df[df[DataFrameKeys.cluster_name] == name]
            rep_logs[name] = rows[DataFrameKeys.preprocessed_text_key].dropna().tolist()[: self.max_logs_per_cluster]

            valid_embs = rows[DataFrameKeys.embeddings_key].dropna()
            if valid_embs.empty:
                continue
            emb_arr = np.array([np.array(e) for e in valid_embs if e is not None and not isinstance(e, float)])
            if emb_arr.ndim == 2 and emb_arr.shape[0] > 0:
                centroids[name] = emb_arr.mean(axis=0)

        return centroids, rep_logs

    def _find_candidate_pairs(
        self,
        names: list[str],
        centroids: dict[str, np.ndarray],
    ) -> list[tuple[str, str, float]]:
        """Return upper-triangle cluster pairs above *cosine_threshold*, sorted by similarity desc.

        Args:
            names: Ordered list of cluster names with centroids.
            centroids: Per-cluster centroid vectors.

        Returns:
            List of ``(name_a, name_b, similarity)`` tuples.
        """
        mat = np.array([centroids[n] for n in names])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = mat / norms
        sim_mat = normed @ normed.T

        rows, cols = np.triu_indices(len(names), k=1)
        pairs = [
            (names[r], names[c], float(sim_mat[r, c]))
            for r, c in zip(rows, cols)
            if sim_mat[r, c] >= self.cosine_threshold
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    async def _llm_confirm_merges(
        self,
        candidate_pairs: list[tuple[str, str, float]],
        representative_logs: dict[str, list[str]],
    ) -> dict[str, str]:
        """Send candidate pairs to the LLM and collect confirmed merge mappings.

        Args:
            candidate_pairs: Sorted list of ``(name_a, name_b, similarity)`` tuples.
            representative_logs: Per-cluster log lists for prompt context.

        Returns:
            Dict mapping old cluster name → canonical name for confirmed merges.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.NEAR_DUPLICATE_CLUSTER_SYS_MESSAGE),
                ("human", prompts.NEAR_DUPLICATE_CLUSTER_LOG_MESSAGE),
            ]
        )
        chain = prompt_template | QgenieModels.azure_gpt_5_2 | _near_dup_parser

        rename_map: dict[str, str] = {}

        def resolve(name: str) -> str:
            seen: set[str] = set()
            while name in rename_map and name not in seen:
                seen.add(name)
                name = rename_map[name]
            return name

        batches = list(self._chunked(candidate_pairs, self.pairs_per_batch))
        concurrency = 8
        sem = asyncio.Semaphore(concurrency)
        progress = {"done": 0}

        async def _process_batch(batch: list[tuple[str, str, float]]):
            pairs_block = self._build_pairs_block(batch, representative_logs)
            async with sem:
                try:
                    result = await chain.ainvoke({"pairs_block": pairs_block})
                except Exception as exc:
                    logger.warning(f"[Deduplicator] LLM batch failed: {exc}")
                    result = []
                progress["done"] += 1
                if progress["done"] % 10 == 0 or progress["done"] == len(batches):
                    logger.info(f"[Deduplicator] {progress['done']}/{len(batches)} batches done")
                return result

        logger.info(f"[Deduplicator] Submitting {len(batches)} batches with concurrency={concurrency}")
        responses = await asyncio.gather(*[_process_batch(b) for b in batches])

        for response in responses:
            items = response.get("results") if isinstance(response, dict) else response
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                if not item.get("is_duplicate"):
                    continue
                a = resolve(item.get("cluster_a", ""))
                b = resolve(item.get("cluster_b", ""))
                keep = item.get("keep_name") or a
                drop = b if keep == a else a
                if drop and keep and drop != keep:
                    rename_map[drop] = keep
                    logger.info(f"[Deduplicator] LLM confirmed: '{drop}' → '{keep}'")

        return rename_map

    def _build_pairs_block(
        self,
        pairs: list[tuple[str, str, float]],
        representative_logs: dict[str, list[str]],
    ) -> str:
        """Build the formatted text block for a batch of candidate pairs.

        Args:
            pairs: List of ``(name_a, name_b, similarity)`` tuples.
            representative_logs: Per-cluster log lists.

        Returns:
            Formatted multi-line string for the LLM prompt.
        """
        blocks = []
        for num, (a, b, sim) in enumerate(pairs, start=1):
            logs_a = "\n".join(f"  [{i}] {log}" for i, log in enumerate(representative_logs.get(a, []), 1))
            logs_b = "\n".join(f"  [{i}] {log}" for i, log in enumerate(representative_logs.get(b, []), 1))
            blocks.append(
                f"Pair {num} (cosine similarity: {sim:.3f}):\n"
                f'  Cluster A: "{a}"\n  Cluster A logs:\n{logs_a}\n\n'
                f'  Cluster B: "{b}"\n  Cluster B logs:\n{logs_b}'
            )
        return "\n\n" + ("-" * 60 + "\n\n").join(blocks)

    @staticmethod
    def _chunked(seq: list, size: int, overlap: int = 0):
        """Yield successive overlapping chunks from *seq*."""
        start = 0
        while start < len(seq):
            yield seq[start : start + size]
            start = start + size - overlap


# ---------------------------------------------------------------------------
# Functional API — module-level functions for backward compatibility
# and direct use by pipeline layers
# ---------------------------------------------------------------------------


@execution_timer
async def detect_and_merge_near_duplicate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and merge near-duplicate clusters via cosine similarity + LLM.

    Thin wrapper around :meth:`Deduplicator.detect_and_merge`.

    Args:
        df: Clustered DataFrame containing ``DataFrameKeys.cluster_name`` and
            ``DataFrameKeys.embeddings_key`` columns.

    Returns:
        DataFrame with near-duplicate cluster names unified.
    """
    return await Deduplicator().detect_and_merge(df)


@execution_timer
async def _analyze_cluster(cluster_df: pd.DataFrame) -> dict:
    """Ask the LLM to name a cluster and identify misclassified rows."""

    def chunk_logs(logs, chunk_size=50, overlap=3):
        start = 0
        while start < len(logs):
            end = start + chunk_size
            yield logs[start:end]
            start = end - overlap

    async def process_chunk(chunk):
        return await chain.ainvoke({"error_logs": chunk})

    error_logs = [
        {"id": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in cluster_df.iterrows()
    ]
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTERING_SYS_MESSAGE), ("human", prompts.CLUSTERING_LOG_MESSAGE)]
    )
    chain = prompt_template | QgenieModels.azure_o3 | _cluster_parser
    tasks = [process_chunk(chunk) for chunk in chunk_logs(error_logs)]
    responses = await asyncio.gather(*tasks)
    if not responses:
        responses = [{}]
    misclassified_ids = [
        id_ for response in responses if response.get("misclassified_ids") for id_ in response["misclassified_ids"]
    ]
    return {"cluster_name": responses[0].get("cluster_name"), "misclassified_ids": list(set(misclassified_ids))}


@execution_timer
async def _async_merge_clusters(
    df: pd.DataFrame,
    cluster_id_a: int = None,
    cluster_id_b: int = None,
    cluster_name_a: str = None,
    cluster_name_b: str = None,
) -> dict:
    """Async version of cluster merging via LLM."""

    def chunk_logs(logs, chunk_size=50, overlap=3):
        start = 0
        while start < len(logs):
            end = start + chunk_size
            yield logs[start:end]
            start = end - overlap

    async def process_chunk(chunk, id_a, id_b):
        return await chain.ainvoke({"id_a": id_a, "logs_a": chunk["logs_a"], "id_b": id_b, "logs_b": chunk["logs_b"]})

    if cluster_id_a and cluster_id_b:
        df_a = df[df[DataFrameKeys.cluster_type_int] == cluster_id_a]
        df_b = df[df[DataFrameKeys.cluster_type_int] == cluster_id_b]
    else:
        df_a = df[df[DataFrameKeys.cluster_name] == cluster_name_a]
        df_b = df[df[DataFrameKeys.cluster_name] == cluster_name_b]

    logs_a = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in df_a.iterrows()
    ]
    logs_b = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in df_b.iterrows()
    ]
    chain = ChatPromptTemplate.from_template(prompts.MERGE_PROMPT_TEMPLATE) | QgenieModels.azure_o3 | _merge_parser
    logs_a_chunks = list(chunk_logs(logs_a))
    logs_b_chunks = list(chunk_logs(logs_b))
    tasks = []
    for chunk_a in logs_a_chunks:
        for chunk_b in logs_b_chunks:
            tasks.append(process_chunk({"logs_a": chunk_a, "logs_b": chunk_b}, cluster_id_a, cluster_id_b))
    responses = await asyncio.gather(*tasks)
    name = ""
    indices: set = set()
    if responses and len(responses) > 1:
        name = responses[0].get("merged_name")
        for response in responses:
            indices.update([int(idx) for idx in response.get("outlier_indices", [])])
    return {"merged_name": name, "outlier_indices": list(indices)}


@execution_timer
async def _get_clusters_name_and_misclassified_errors(df: pd.DataFrame) -> dict:
    unique_cluster_ids = [
        cid for cid in df[DataFrameKeys.cluster_type_int].unique() if cid != ClusterSpecificKeys.non_grouped_key
    ]

    async def _process(cluster_id):
        cluster_df = df[df[DataFrameKeys.cluster_type_int] == cluster_id]
        return int(cluster_id), await _analyze_cluster(cluster_df)

    pairs = await asyncio.gather(*[_process(cid) for cid in unique_cluster_ids])
    return dict(pairs)


def _get_duplicate_clusters(results: dict) -> dict:
    name_to_clusters: dict = defaultdict(list)
    for cluster_id, result in results.items():
        name_to_clusters[result["cluster_name"]].append(cluster_id)
    return {name: ids for name, ids in name_to_clusters.items() if len(ids) > 1}


@execution_timer
async def _merge_duplicate_clusters(
    df: pd.DataFrame, duplicate_clusters: dict, cluster_results: dict
) -> tuple[pd.DataFrame, dict]:
    for duplicate_name, cluster_ids in duplicate_clusters.items():
        base_cluster_id = cluster_ids[0]
        for next_cluster_id in cluster_ids[1:]:
            response = await _async_merge_clusters(df, cluster_id_a=base_cluster_id, cluster_id_b=next_cluster_id)
            cluster_results[base_cluster_id]["cluster_name"] = response["merged_name"]
            df.loc[df[DataFrameKeys.cluster_type_int] == next_cluster_id, DataFrameKeys.cluster_type_int] = (
                base_cluster_id
            )
            outlier_indices = [int(i) for i in response.get("outlier_indices", [])]
            df.loc[outlier_indices, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key
            if next_cluster_id in cluster_results:
                del cluster_results[next_cluster_id]
    return df, cluster_results


@execution_timer
def _give_cluster_names_and_reassign_misc_clusters(df: pd.DataFrame, cluster_results: dict) -> pd.DataFrame:
    for cluster_id, result in cluster_results.items():
        misclassified_ids = result.get("misclassified_ids", [])
        if misclassified_ids:
            df.loc[df.index.isin(misclassified_ids), DataFrameKeys.cluster_type_int] = (
                ClusterSpecificKeys.non_grouped_key
            )
        cluster_name = result.get("cluster_name")
        if cluster_name:
            df.loc[df[DataFrameKeys.cluster_type_int] == cluster_id, DataFrameKeys.cluster_name] = cluster_name
    return df


@execution_timer
async def _recluster_with_context(df: pd.DataFrame) -> pd.DataFrame:
    unclustered_df = df[df[DataFrameKeys.cluster_type_int] == ClusterSpecificKeys.non_grouped_key]
    error_logs = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]}
        for idx, row in unclustered_df.iterrows()
    ]
    chain = ChatPromptTemplate.from_template(prompts.RECLUSTERING_PROMPT) | QgenieModels.azure_o3 | _recluster_parser
    outliers_recluster_results = await chain.ainvoke({"error_logs": error_logs})
    df_index_set = set(df.index)
    for result in outliers_recluster_results:
        name, indices = result.get("cluster_name"), result.get("log_indices")
        if name and indices:
            if not isinstance(indices, list):
                try:
                    indices = [int(indices)]
                except (ValueError, TypeError):
                    continue
            else:
                indices = [int(idx) for idx in indices if isinstance(idx, (int, float))]
            valid_indices = [idx for idx in indices if idx in df_index_set]
            if valid_indices:
                df.loc[valid_indices, DataFrameKeys.cluster_name] = name
                df.loc[valid_indices, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.default_cluster_key
    return df


@execution_timer
async def subcluster_verifier_failed(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-cluster VerifierFailed logs using the LLM in batches.

    Args:
        df: DataFrame filtered to ``VerifierFailed`` cluster rows.

    Returns:
        DataFrame with updated ``cluster_name`` assignments, or an empty
        DataFrame if *df* is ``None`` or empty.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    logs = [{"index": int(idx), "error log": row[DataFrameKeys.error_reason]} for idx, row in df.iterrows()]

    def chunk_logs(items, size=30):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.SUBCLUSTER_VERIFIER_FAILED_SYS_MESSAGE),
            ("human", prompts.SUBCLUSTER_VERIFIER_FAILED_LOG_MESSAGE),
        ]
    )
    chain = prompt_template | QgenieModels.azure_o3 | _subcluster_verifier_parser

    previous_clusters_agg: dict[str, set[int]] = {}
    df_index_set = set(int(i) for i in df.index)

    for batch in chunk_logs(logs):
        if not batch:
            continue
        batch_index_set = {int(item["index"]) for item in batch}
        clustered_in_batch: set[int] = set()
        max_iters = 5
        for _ in range(max_iters):
            global_assigned_indices = set()
            for indices in previous_clusters_agg.values():
                global_assigned_indices.update(indices)
            unassigned_in_batch = [item for item in batch if int(item["index"]) not in global_assigned_indices]
            if not unassigned_in_batch:
                break
            try:
                response = await chain.ainvoke(
                    {
                        "logs": unassigned_in_batch,
                        "previous_clusters": {k: list(v) for k, v in previous_clusters_agg.items()},
                    }
                )
                if not isinstance(response, dict):
                    break
                new_indices_added = set()
                clusters = response.get("clusters", [])
                if not isinstance(clusters, list):
                    break
                for cluster_info in clusters:
                    name = cluster_info.get("cluster_name")
                    indices_raw = cluster_info.get("indices", [])
                    if not name or not indices_raw:
                        continue
                    indices = {int(i) for i in indices_raw if str(i).isdigit() and int(i) in df_index_set}
                    if not indices:
                        continue
                    if name not in previous_clusters_agg:
                        previous_clusters_agg[name] = set()
                    new_for_cluster = indices - previous_clusters_agg[name]
                    previous_clusters_agg[name].update(indices)
                    new_indices_added.update(new_for_cluster)
                    clustered_in_batch.update(indices & batch_index_set)
                if not new_indices_added:
                    break
            except Exception as e:
                logger.warning(f"[subcluster_verifier_failed] LLM call failed: {e}")
                break

    if not previous_clusters_agg:
        return pd.DataFrame()

    rows = []
    for cluster_name, indices in previous_clusters_agg.items():
        valid_indices = [i for i in indices if i in df_index_set]
        for idx in valid_indices:
            row = df.loc[idx].copy()
            row[DataFrameKeys.cluster_name] = cluster_name
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = result_df[~result_df.index.duplicated(keep="first")]
    return result_df


@execution_timer
async def qgenie_post_processing(df: pd.DataFrame) -> pd.DataFrame:
    """LLM-based post-processing: cluster naming, deduplication, and reclustering.

    Args:
        df: DataFrame after HDBSCAN clustering with ``cluster_type_int`` labels.

    Returns:
        DataFrame with updated ``cluster_name`` assignments.
    """
    try:
        analyzed_results = await _get_clusters_name_and_misclassified_errors(df)
        if analyzed_results:
            duplicate_clusters = _get_duplicate_clusters(analyzed_results)
            df, analyzed_results = await _merge_duplicate_clusters(df, duplicate_clusters, analyzed_results)
            df = _give_cluster_names_and_reassign_misc_clusters(df, analyzed_results)
        df = detect_cluster_outlier(df)
        df = reassign_unclustered_logs(df)
        df = await _recluster_with_context(df)
    except Exception as e:
        logger.error(f"Exception in qgenie_post_processing: {e}")
        traceback.print_exc()
    logger.info(f"qgenie_post_processing clusters: {df[DataFrameKeys.cluster_name].unique()}")
    return df
