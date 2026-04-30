"""Pre-grouping pipeline: fuzzy + SPLADE grouping before HDBSCAN.

Provides :class:`PreGroupingPipeline`, which wraps the two pre-grouping
passes that run *before* the main HDBSCAN clustering step:

1. **Fuzzy grouping** — rapidfuzz string similarity on short error texts
   to catch near-identical errors across test cases.
2. **SPLADE pre-grouping** — sparse neural similarity to catch semantically
   equivalent errors with different surface forms.

Both passes only operate on rows whose ``cluster_name`` is still
:attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`.

Layering
--------
This module imports from ``src.clustering``, ``src.embeddings``, ``src.llm``,
``src.constants``, ``src.logger``, and ``src.utils``.  It must **not** import
from ``src.pipeline.cluster_pipeline`` (no circular deps within the package).
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.clustering.splade_encoder import SPLADEEncoder
from src.constants import ClusterSpecificKeys, DataFrameKeys, SPLADEConfigurations, regex_based_filteration_patterns
from src.custom_clustering import CustomEmbeddingCluster
from src.embeddings import FallbackEmbeddings
from src.llm.cluster_namer import generate_cluster_name
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "PreGroupingPipeline",
    "group_similar_errors",
    "fuzzy_cluster_grouping",
    "splade_pregroup",
    "check_if_issue_alread_grouped",
    "update_rows_by_regex_patterns",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def update_rows_by_regex_patterns(df: pd.DataFrame) -> pd.DataFrame:
    for name, pattern in regex_based_filteration_patterns.items():
        matched_df = df[
            df[DataFrameKeys.preprocessed_text_key].astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        ]
        logger.debug(f"Found occurence of match: {name}: {matched_df.shape[0]}")
        if not matched_df.empty:
            if "verifier failed" in pattern:
                cluster_name = {"cluster_name": "VerifierFailed"}
            else:
                cluster_name = await generate_cluster_name(matched_df)
            df.loc[matched_df.index, DataFrameKeys.cluster_name] = cluster_name["cluster_name"]


@execution_timer
def group_similar_errors(df: pd.DataFrame, column: str, threshold) -> list:
    groups = []
    assigned = set()

    if df.empty:
        return groups

    index_list = df.index.tolist()
    if len(index_list) <= 1:
        return groups

    for i, base_idx in enumerate(index_list):
        if base_idx in assigned:
            continue

        base = df.at[base_idx, column]
        group = [base_idx]

        for right_idx in index_list[i + 1 :]:
            ratio = fuzz.ratio(base, df.at[right_idx, column])
            if right_idx not in assigned and ratio >= threshold:
                group.append(right_idx)

        if len(group) > 1:
            groups.append(group)
            assigned.update(group)

    return groups


@execution_timer
async def fuzzy_cluster_grouping(
    failures_dataframe: pd.DataFrame, threshold=100, bin_intervals=[[0, 50], [50, 110]]
) -> pd.DataFrame:
    failures_dataframe.loc[:, DataFrameKeys.error_logs_length] = failures_dataframe[
        DataFrameKeys.preprocessed_text_key
    ].apply(len)

    # Step 1: Create bin column if not already present
    if DataFrameKeys.bins not in failures_dataframe.columns:
        bin_edges = sorted({edge for interval in bin_intervals for edge in interval})
        failures_dataframe.loc[:, DataFrameKeys.bins] = pd.cut(
            failures_dataframe[DataFrameKeys.error_logs_length], bins=bin_edges, right=True
        )

    for i, j in bin_intervals:
        target_bin = pd.Interval(left=i, right=j, closed="right")
        # Group similar errors
        grouped_indices = group_similar_errors(
            failures_dataframe[failures_dataframe[DataFrameKeys.bins] == target_bin],
            DataFrameKeys.preprocessed_text_key,
            threshold,
        )

        # Process groups in parallel using asyncio.gather
        if grouped_indices:
            # Create tasks for each group
            tasks = [generate_cluster_name(failures_dataframe.iloc[group]) for group in grouped_indices]
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            # Update the dataframe with results
            for group, result in zip(grouped_indices, results):
                failures_dataframe.loc[group, DataFrameKeys.cluster_name] = result["cluster_name"]

    # regex based common errors mapping
    await update_rows_by_regex_patterns(failures_dataframe)
    return failures_dataframe


@execution_timer
async def check_if_issue_alread_grouped(df: pd.DataFrame) -> pd.DataFrame:
    # Identify rows that are not yet grouped
    mask = df[DataFrameKeys.cluster_name].isin([ClusterSpecificKeys.non_grouped_key, np.nan])
    ungrouped_df = df[mask]

    if not ungrouped_df.empty:
        # Get cluster names using FAISS — use unmasked embedding text for better similarity
        new_cluster_names, class_names, embeddings = await CustomEmbeddingCluster().batch_search(
            type_=ungrouped_df.iloc[0]["type"],  # assuming same type for batch
            queries=ungrouped_df[DataFrameKeys.preprocessed_text_key].tolist(),
        )
        # Update the original DataFrame
        df.loc[mask, DataFrameKeys.cluster_name] = new_cluster_names
        df.loc[mask, DataFrameKeys.cluster_class] = class_names
        df.loc[mask, DataFrameKeys.embeddings_key] = pd.Series(embeddings.tolist(), index=df.index[mask])

        # Create a boolean mask for rows that were successfully grouped (not non_grouped_key)
        successfully_grouped_mask = mask & df[DataFrameKeys.cluster_name].ne(ClusterSpecificKeys.non_grouped_key)
        if any(successfully_grouped_mask):
            df.loc[successfully_grouped_mask, DataFrameKeys.grouped_from_faiss] = True

    return df


@execution_timer
async def splade_pregroup(df: pd.DataFrame, type_: str = None) -> pd.DataFrame:
    """
    SPLADE-based pre-grouping pass.
    Groups errors with high SPLADE sparse-vector similarity before HDBSCAN clustering.
    Only processes rows with cluster_name == non_grouped_key.

    For each discovered group:
      1. Embeds all member texts (one batch for all groups) and computes a centroid.
      2. Searches CustomEmbeddingCluster with that centroid — if an existing cluster
         matches, assigns its name directly with no LLM call.
      3. Only calls the LLM naming API for groups that did not match the DB.

    Returns df with cluster_name updated for discovered groups.
    """
    if not SPLADEConfigurations.enabled:
        logger.debug(f"[SPLADEPregroup] type={type_}: SPLADE disabled, skipping pre-grouping")
        return df

    SPLADE_THRESHOLD = SPLADEConfigurations.pregroup_threshold
    MIN_GROUP_SIZE = 2

    cluster_col = DataFrameKeys.cluster_name
    text_col = DataFrameKeys.preprocessed_text_key

    mask = df[cluster_col] == ClusterSpecificKeys.non_grouped_key
    unclustered_count = mask.sum()

    if unclustered_count < MIN_GROUP_SIZE:
        logger.info(
            f"[SPLADEPregroup] type={type_}: only {unclustered_count} unclustered rows "
            f"(< min_group_size={MIN_GROUP_SIZE}), skipping"
        )
        return df

    logger.info(
        f"[SPLADEPregroup] type={type_}: starting SPLADE pre-grouping on "
        f"{unclustered_count} unclustered rows (threshold={SPLADE_THRESHOLD})"
    )

    work_df = df[mask].copy()
    indices = list(work_df.index)
    texts = work_df[text_col].astype(str).tolist()

    encoder = SPLADEEncoder()
    if not encoder.is_available:
        logger.warning(f"[SPLADEPregroup] type={type_}: SPLADE encoder unavailable, skipping pre-grouping")
        return df

    # Run CPU-bound SPLADE inference in a thread executor so the event loop stays responsive.
    # Without this, the ~6s torch forward pass blocks all pending network I/O (QGenie, etc.).
    loop = asyncio.get_running_loop()
    sparse_vecs = await loop.run_in_executor(None, encoder.encode, texts)
    if sparse_vecs is None:
        logger.warning(f"[SPLADEPregroup] type={type_}: encoding returned None, skipping pre-grouping")
        return df

    # Pairwise SPLADE dot products → (N, N) dense matrix
    sim_matrix = (sparse_vecs @ sparse_vecs.T).toarray()

    assigned = [False] * len(indices)
    groups = []

    for anchor_pos in range(len(indices)):
        if assigned[anchor_pos]:
            continue

        similarity_row = sim_matrix[anchor_pos].copy()
        min_sim, max_sim = similarity_row.min(), similarity_row.max()
        if max_sim - min_sim > 1e-9:
            normalized_row = (similarity_row - min_sim) / (max_sim - min_sim)
        else:
            normalized_row = np.zeros(len(indices))

        group = [anchor_pos]
        for candidate_pos in range(len(indices)):
            if (
                candidate_pos != anchor_pos
                and not assigned[candidate_pos]
                and normalized_row[candidate_pos] >= SPLADE_THRESHOLD
            ):
                group.append(candidate_pos)

        if len(group) >= MIN_GROUP_SIZE:
            groups.append(group)
            for member_pos in group:
                assigned[member_pos] = True

    if not groups:
        logger.info(
            f"[SPLADEPregroup] type={type_}: no groups found above threshold={SPLADE_THRESHOLD}, "
            f"all {unclustered_count} rows remain unclustered"
        )
        return df

    group_sizes = [len(g) for g in groups]
    logger.info(
        f"[SPLADEPregroup] type={type_}: found {len(groups)} candidate groups "
        f"(sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
        f"total grouped={sum(group_sizes)}/{unclustered_count}); "
        f"attempting DB lookup before LLM"
    )

    # --- Batch embed all group texts at once, then compute per-group centroids ---
    all_group_texts = []
    group_text_ranges: list[tuple[int, int]] = []  # (start, count) into all_group_texts
    for group_positions in groups:
        range_start = len(all_group_texts)
        all_group_texts.extend([texts[pos] for pos in group_positions])
        group_text_ranges.append((range_start, len(group_positions)))

    embeddings_list = await FallbackEmbeddings().aembed(all_group_texts)
    embeddings_array = np.array(embeddings_list)

    # --- For each group: try DB lookup first, collect misses for LLM ---
    custom_cluster = CustomEmbeddingCluster()
    groups_needing_llm: list[list[int]] = []

    for group_positions, (emb_start, emb_count) in zip(groups, group_text_ranges):
        group_embeddings = embeddings_array[emb_start : emb_start + emb_count]
        centroid = group_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        db_cluster_name = ClusterSpecificKeys.non_grouped_key
        if type_:
            db_cluster_name, _ = custom_cluster._search_with_embedding(type_, centroid)

        if db_cluster_name != ClusterSpecificKeys.non_grouped_key:
            group_df_indices = [indices[pos] for pos in group_positions]
            df.loc[group_df_indices, cluster_col] = db_cluster_name
            logger.debug(f"[SPLADEPregroup] DB hit: group of {len(group_positions)} rows → '{db_cluster_name}'")
        else:
            groups_needing_llm.append(group_positions)

    db_hit_count = len(groups) - len(groups_needing_llm)

    # --- LLM fallback only for groups the DB did not recognise ---
    if groups_needing_llm:
        logger.info(
            f"[SPLADEPregroup] type={type_}: {db_hit_count}/{len(groups)} groups matched DB — "
            f"calling LLM for remaining {len(groups_needing_llm)}"
        )
        llm_tasks = [generate_cluster_name(work_df.iloc[group_positions]) for group_positions in groups_needing_llm]
        llm_results = await asyncio.gather(*llm_tasks)

        for group_positions, result in zip(groups_needing_llm, llm_results):
            if result and "cluster_name" in result:
                group_df_indices = [indices[pos] for pos in group_positions]
                cluster_name = result["cluster_name"]
                df.loc[group_df_indices, cluster_col] = cluster_name
                logger.debug(f"[SPLADEPregroup] LLM: group of {len(group_positions)} rows → '{cluster_name}'")
            else:
                logger.warning(
                    f"[SPLADEPregroup] type={type_}: LLM returned no cluster name for "
                    f"group of size {len(group_positions)}, rows remain unclustered"
                )
    else:
        logger.info(f"[SPLADEPregroup] type={type_}: all {len(groups)} groups matched DB — no LLM calls needed")

    named_count = (df.loc[list(work_df.index), cluster_col] != ClusterSpecificKeys.non_grouped_key).sum()
    logger.info(
        f"[SPLADEPregroup] type={type_}: complete — "
        f"{named_count}/{unclustered_count} rows grouped into {len(groups)} clusters "
        f"({db_hit_count} from DB, {len(groups_needing_llm)} from LLM)"
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class PreGroupingPipeline:
    """Runs fuzzy and SPLADE pre-grouping passes on a DataFrame of error logs.

    Both passes only mutate rows whose ``cluster_name`` column equals
    :attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`.  Rows that
    are already grouped (e.g. from the vector-DB lookup) are left untouched.

    Args:
        fuzzy_threshold: Minimum fuzz.ratio score (0–100) to consider two
            error logs as identical.  Defaults to 100 (exact match after
            normalisation).
        splade_threshold: Minimum normalised SPLADE dot-product score (0–1)
            to group two errors together.  Defaults to
            :attr:`~src.constants.SPLADEConfigurations.pregroup_threshold`.

    Example::

        pipeline = PreGroupingPipeline()
        df = await pipeline.run(df, cluster_type="quantizer")
    """

    def __init__(
        self,
        fuzzy_threshold: int = 100,
        splade_threshold: float | None = None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.splade_threshold = splade_threshold or SPLADEConfigurations.pregroup_threshold

    @execution_timer
    async def run(self, df: pd.DataFrame, cluster_type: str | None = None) -> pd.DataFrame:
        """Run both pre-grouping passes sequentially.

        Args:
            df: DataFrame of failure records.  Must include
                ``DataFrameKeys.preprocessed_text_key`` and
                ``DataFrameKeys.cluster_name`` columns.
            cluster_type: Test-type identifier (e.g. ``"quantizer"``).  Used
                for vector-DB lookup during SPLADE grouping.  Pass ``None``
                to skip the DB lookup.

        Returns:
            Updated DataFrame with ``cluster_name`` assigned for all
            successfully grouped rows.
        """
        df = await self._fuzzy_grouping(df)
        df = await self._splade_pregroup(df, cluster_type=cluster_type)
        return df

    @execution_timer
    async def _fuzzy_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rapidfuzz-based pre-grouping pass.

        Args:
            df: Input DataFrame.

        Returns:
            Updated DataFrame.
        """
        return await fuzzy_cluster_grouping(df, threshold=self.fuzzy_threshold)

    @execution_timer
    async def _splade_pregroup(self, df: pd.DataFrame, cluster_type: str | None) -> pd.DataFrame:
        """SPLADE sparse-vector pre-grouping pass.

        Args:
            df: Input DataFrame.
            cluster_type: Test-type identifier for DB lookup.

        Returns:
            Updated DataFrame.
        """
        if not SPLADEConfigurations.enabled:
            logger.debug(f"[PreGroupingPipeline] SPLADE disabled for type={cluster_type}")
            return df

        return await splade_pregroup(df, type_=cluster_type)
