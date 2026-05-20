"""Cluster quality analysis — ranking and cohesion detection.

Provides two independent, stateless analysis classes:

- :class:`ClusterRanker` — ranks cluster members by cosine similarity to the
  cluster centroid and marks core members.
- :class:`ClusterCohesionAnalyzer` — detects loose, poorly-formed clusters
  using mean pairwise intra-cluster cosine similarity.

Both classes add new columns to the input DataFrame and return a copy.
They use only the precomputed ``embeddings`` column — no additional model
inference is triggered.

Layering
--------
Imports from ``src.constants``, ``src.logger``, and standard library /
NumPy / sklearn.  No imports from ``src.helpers``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor

from src.constants import ClusterSpecificKeys, DataFrameKeys, SPLADEConfigurations
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = [
    "ClusterRanker",
    "ClusterCohesionAnalyzer",
    "detect_cluster_outlier",
    "reassign_unclustered_logs",
    "merge_similar_clusters",
    "update_labels_with_merged_clusters",
]


class ClusterRanker:
    """Ranks cluster members by representativeness (cosine similarity to centroid).

    Added DataFrame columns
    -----------------------
    - ``rank`` — integer rank within the cluster (1 = most representative).
    - ``representativeness_score`` — float in [0, 1] (min-max normalised cosine
      similarity to cluster centroid; 1 = closest).
    - ``is_core_member`` — ``True`` for the top
      :attr:`core_percentile` fraction of members.

    Args:
        core_percentile: Fraction of members per cluster to mark as core.
            Default: 0.50 (top 50 %).

    Example:
        >>> ranker = ClusterRanker()
        >>> df_ranked = ranker.rank_dataframe(df)
    """

    def __init__(self, core_percentile: float = SPLADEConfigurations.core_member_percentile) -> None:
        self.core_percentile = core_percentile

    def rank_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ranking columns to *df*.

        Processes each cluster independently.  Rows with missing embeddings
        within a cluster are skipped (their rank columns retain the defaults).

        Args:
            df: Input DataFrame.  Must contain
                :attr:`~src.constants.DataFrameKeys.embeddings_key` and
                :attr:`~src.constants.DataFrameKeys.cluster_name` columns.

        Returns:
            Copy of *df* with ``rank``, ``representativeness_score``, and
            ``is_core_member`` columns added.
        """
        df = df.copy()
        df["rank"] = -1
        df["representativeness_score"] = 0.0
        df["is_core_member"] = False

        emb_col = DataFrameKeys.embeddings_key
        cluster_col = DataFrameKeys.cluster_name

        if emb_col not in df.columns or cluster_col not in df.columns:
            logger.warning("[ClusterRanker] Missing required columns; skipping ranking.")
            return df

        cluster_groups = list(df.groupby(cluster_col))
        logger.info(f"[ClusterRanker] Ranking {len(cluster_groups)} clusters ({len(df)} rows).")

        for cluster_name, group in cluster_groups:
            valid = group[group[emb_col].notna()]
            if valid.empty:
                continue

            embeddings = np.vstack(valid[emb_col].values)
            centroid = embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(embeddings, centroid).flatten()

            # Min-max normalise to [0, 1]
            lo, hi = sims.min(), sims.max()
            norm_scores = (sims - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones(len(sims))

            # Rank descending (1 = highest similarity)
            n = len(sims)
            ranks = n - np.argsort(np.argsort(sims))

            core_threshold = np.percentile(norm_scores, (1 - self.core_percentile) * 100)

            for idx, score, rank in zip(valid.index, norm_scores, ranks):
                df.at[idx, "rank"] = int(rank)
                df.at[idx, "representativeness_score"] = float(score)
                df.at[idx, "is_core_member"] = bool(score >= core_threshold)

            logger.debug(f"[ClusterRanker] '{cluster_name}': {n} members, core_threshold={core_threshold:.3f}.")

        total_core = int(df["is_core_member"].sum())
        logger.info(f"[ClusterRanker] Done: {total_core}/{len(df)} rows marked as core members.")
        return df


class ClusterCohesionAnalyzer:
    """Detects loose clusters using mean pairwise intra-cluster cosine similarity.

    A cluster is considered *loose* when:
    - Its mean pairwise cosine similarity is below :attr:`low_cohesion_threshold`
    - AND it contains more than 5 members.

    Added DataFrame columns
    -----------------------
    - ``cluster_cohesion_score`` — float in [0, 1]; mean pairwise cosine
      similarity within the cluster.
    - ``is_loose_cluster`` — ``True`` if the cluster is flagged as loose.

    Args:
        low_cohesion_threshold: Cohesion score below which a cluster is
            considered loose.  Default: 0.35.

    Example:
        >>> analyzer = ClusterCohesionAnalyzer()
        >>> df_analyzed = analyzer.analyze_dataframe(df)
    """

    def __init__(self, low_cohesion_threshold: float = SPLADEConfigurations.low_cohesion_threshold) -> None:
        self.low_cohesion_threshold = low_cohesion_threshold

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cohesion columns to *df*.

        The computation is O(n²) per cluster in the number of cluster members.
        For very large clusters this may be slow; set a size cap in the calling
        code if needed.

        Args:
            df: Input DataFrame.  Must contain
                :attr:`~src.constants.DataFrameKeys.embeddings_key` and
                :attr:`~src.constants.DataFrameKeys.cluster_name` columns.

        Returns:
            Copy of *df* with ``cluster_cohesion_score`` and
            ``is_loose_cluster`` columns added.
        """
        df = df.copy()
        df["cluster_cohesion_score"] = 1.0
        df["is_loose_cluster"] = False

        emb_col = DataFrameKeys.embeddings_key
        cluster_col = DataFrameKeys.cluster_name

        if emb_col not in df.columns or cluster_col not in df.columns:
            logger.warning("[CohesionAnalyzer] Missing required columns; skipping cohesion analysis.")
            return df

        cluster_groups = list(df.groupby(cluster_col))
        logger.info(f"[CohesionAnalyzer] Analyzing {len(cluster_groups)} clusters.")

        loose: list[str] = []
        max_sample = 500
        for cluster_name, group in cluster_groups:
            valid = group[group[emb_col].notna()]
            n = len(valid)
            if n < 2:
                continue  # single-member: cohesion trivially 1.0

            if n > max_sample:
                valid = valid.sample(n=max_sample, random_state=42)
                n = max_sample

            embeddings = np.vstack(valid[emb_col].values)
            sim_matrix = cosine_similarity(embeddings)  # (n, n)
            off_diag = sim_matrix[~np.eye(n, dtype=bool)]
            cohesion = float(np.clip(off_diag.mean(), 0.0, 1.0)) if len(off_diag) > 0 else 1.0
            is_loose = cohesion < self.low_cohesion_threshold and n > 5

            df.loc[group.index, "cluster_cohesion_score"] = cohesion
            df.loc[group.index, "is_loose_cluster"] = is_loose

            logger.debug(
                f"[CohesionAnalyzer] '{cluster_name}': size={n}, " f"cohesion={cohesion:.3f}, loose={is_loose}."
            )
            if is_loose:
                loose.append(str(cluster_name))

        if loose:
            logger.warning(
                f"[CohesionAnalyzer] {len(loose)} loose clusters detected "
                f"(threshold={self.low_cohesion_threshold}): {loose}"
            )
        else:
            logger.info(f"[CohesionAnalyzer] All {len(cluster_groups)} clusters have acceptable cohesion.")

        return df


def detect_cluster_outlier(df: pd.DataFrame) -> pd.DataFrame:
    """Mark outlier members of each cluster by reassigning them to the ungrouped key.

    Uses Local Outlier Factor + cosine similarity to centroid to identify
    cluster members that are both geometrically distant and flagged as
    density outliers.

    Args:
        df: DataFrame with ``embeddings`` and ``cluster_name`` columns.

    Returns:
        DataFrame with outlier rows moved to
        :attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`.
    """
    clusters = set(df[DataFrameKeys.cluster_name]) - {-1}
    for cluster in clusters:
        cluster_df = df[df[DataFrameKeys.cluster_name] == cluster]
        embedding_matrix = np.vstack(cluster_df[DataFrameKeys.embeddings_key].values)

        if len(embedding_matrix) < 2:
            continue

        centroid = np.mean(embedding_matrix, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embedding_matrix, centroid).flatten()
        cluster_df = cluster_df.copy()
        cluster_df["cosine_similarity_to_centroid"] = similarities

        k = min(15, len(embedding_matrix) - 1)
        lof = LocalOutlierFactor(n_neighbors=k, metric="cosine", n_jobs=-1)
        outlier_flags = lof.fit_predict(embedding_matrix)
        cluster_df["lof_outlier"] = outlier_flags

        similarity_threshold = 0.7
        outliers = cluster_df[
            (cluster_df["cosine_similarity_to_centroid"] < similarity_threshold) & (cluster_df["lof_outlier"] == -1)
        ]

        logger.info(f"For {cluster} Outlier log IDs: {outliers['reason'].tolist()}")
        if not outliers.empty:
            df.loc[outliers.index, DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key

    return df


def merge_similar_clusters(embeddings, labels, threshold=0.87):
    """Merge clusters whose centroids have cosine similarity above *threshold*.

    Args:
        embeddings: Sequence of embedding vectors aligned with *labels*.
        labels: Cluster label per embedding (``-1`` = noise/ungrouped).
        threshold: Minimum cosine similarity to merge two cluster centroids.

    Returns:
        Dict mapping each surviving parent label → list of merged child labels.
    """
    unique_labels = set(labels) - {-1}
    centroids = {
        label: np.mean([embeddings[i] for i in range(len(labels)) if labels[i] == label], axis=0)
        for label in unique_labels
    }

    merged = {}
    used = set()
    for label1 in unique_labels:
        if label1 in used:
            continue
        merged[label1] = [label1]
        for label2 in unique_labels:
            if label1 != label2 and label2 not in used:
                sim = cosine_similarity([centroids[label1]], [centroids[label2]])[0][0]
                if sim >= threshold:
                    merged[label1].append(label2)
                    used.add(label2)
        used.add(label1)
    return merged


def update_labels_with_merged_clusters(df: pd.DataFrame, merged_clusters: dict, label_col: str) -> pd.DataFrame:
    """Relabel *df[label_col]* so that all merged children adopt the parent label.

    Args:
        df: DataFrame whose *label_col* will be updated in-place.
        merged_clusters: Output of :func:`merge_similar_clusters` — maps
            parent label → list of child labels.
        label_col: Column name containing integer cluster labels.

    Returns:
        *df* with *label_col* updated.
    """
    label_map = {}
    for parent, children in merged_clusters.items():
        for child in children:
            label_map[int(child)] = int(parent)
    df[label_col] = df[label_col].map(label_map).fillna(df[label_col])
    return df


def reassign_unclustered_logs(df: pd.DataFrame, threshold: float = 0.88) -> pd.DataFrame:
    """Reassign ungrouped (cluster=-1) rows to the nearest valid cluster.

    Computes per-cluster centroids from already-clustered rows, then
    reassigns each ungrouped row to the cluster whose centroid has the
    highest cosine similarity (above *threshold*).

    Args:
        df: DataFrame with ``embeddings`` and ``cluster_name`` columns.
        threshold: Minimum cosine similarity required to reassign a row.

    Returns:
        DataFrame with previously ungrouped rows reassigned where possible.
    """
    valid_clusters = set(df[DataFrameKeys.cluster_name]) - {-1}
    logger.info(f"Valid clusters for reassigning unclustered logs: {valid_clusters}")
    if not valid_clusters:
        return df

    unclustered_mask = df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
    unclustered = df[unclustered_mask]
    if unclustered.empty:
        return df

    valid_embed_mask = unclustered[DataFrameKeys.embeddings_key].apply(lambda v: isinstance(v, (list, np.ndarray)))
    unclustered = unclustered[valid_embed_mask]
    if unclustered.empty:
        return df

    labels = list(valid_clusters)
    centroids = np.stack(
        [
            np.mean(np.stack(df[df[DataFrameKeys.cluster_name] == label][DataFrameKeys.embeddings_key].values), axis=0)
            for label in labels
        ]
    )
    log_matrix = np.stack([np.asarray(v) for v in unclustered[DataFrameKeys.embeddings_key].values])

    sims = cosine_similarity(log_matrix, centroids)  # (U, K)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    to_assign = best_sim >= threshold

    if to_assign.any():
        labels_arr = np.array(labels, dtype=object)
        target_idx = unclustered.index[to_assign]
        df.loc[target_idx, DataFrameKeys.cluster_name] = labels_arr[best_idx[to_assign]]
        for idx, label, sim in zip(target_idx, labels_arr[best_idx[to_assign]], best_sim[to_assign]):
            logger.info(f"Best label: {label}  Best Similarity: {float(sim)}")

    return df
