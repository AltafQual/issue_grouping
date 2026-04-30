"""HDBSCAN clustering wrapper.

:class:`HDBSCANClusterer` wraps the HDBSCAN algorithm with sensible defaults
for error-log clustering and exposes a clean, type-annotated interface.  All
HDBSCAN configuration comes from constructor parameters — no hard-coded magic
numbers inside the class.

Layering
--------
Imports from ``src.constants``, ``src.core``, ``src.logger``, and standard
library / NumPy / hdbscan.  No imports from ``src.helpers``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None

from src.constants import ClusterSpecificKeys, DataFrameKeys
from src.core.exceptions import ClusteringError
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["HDBSCANClusterer"]


class HDBSCANClusterer:
    """Wraps HDBSCAN for error-log cluster assignment.

    Args:
        min_cluster_size: Minimum number of samples to form a cluster.
            Smaller values → more (smaller) clusters; default 5.
        min_samples: Number of samples in a neighbourhood for a point to be
            considered a core point.  ``None`` defaults to *min_cluster_size*.
        cluster_selection_epsilon: Distance threshold below which clusters are
            not split further; 0.0 disables this behaviour.
        metric: Distance metric passed to HDBSCAN.

    Example:
        >>> clusterer = HDBSCANClusterer(min_cluster_size=5)
        >>> labels = clusterer.fit(embeddings_array)
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Run HDBSCAN on *embeddings* and return integer cluster labels.

        Labels are HDBSCAN's native integer labels, where ``-1`` indicates
        noise (unclustered) points — matching
        :attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`.

        Args:
            embeddings: Float32 array of shape ``[N, dim]`` — one row per
                error log.

        Returns:
            Integer array of shape ``[N]`` with cluster labels.

        Raises:
            ClusteringError: If HDBSCAN is unavailable or clustering fails.
        """
        if len(embeddings) == 0:
            return np.array([], dtype=int)

        if hdbscan_lib is None:
            raise ClusteringError(
                "hdbscan package is not installed. Install it with: uv add hdbscan",
                cause=ImportError("hdbscan"),
            )

        try:
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=self.metric,
            )
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            logger.info(
                f"[HDBSCAN] Clustered {len(embeddings)} samples → " f"{n_clusters} clusters, {n_noise} noise points."
            )
            return labels
        except Exception as exc:
            raise ClusteringError(
                f"HDBSCAN clustering failed: {exc}",
                cause=exc,
            ) from exc

    def fit_dataframe(self, df: pd.DataFrame, embeddings_col: str | None = None) -> pd.DataFrame:
        """Apply HDBSCAN to *df* and write integer cluster labels to a column.

        Args:
            df: Input DataFrame with an embeddings column.
            embeddings_col: Name of the embeddings column.  Defaults to
                :attr:`~src.constants.DataFrameKeys.embeddings_key`.

        Returns:
            Copy of *df* with
            :attr:`~src.constants.DataFrameKeys.cluster_type_int` column added.

        Raises:
            ClusteringError: If clustering fails.
        """
        col = embeddings_col or DataFrameKeys.embeddings_key
        if col not in df.columns:
            raise ClusteringError(
                f"Embeddings column '{col}' not found in DataFrame.",
                stage="hdbscan",
            )

        valid = df[df[col].notna()]
        if valid.empty:
            df = df.copy()
            df[DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key
            return df

        embeddings = np.vstack(valid[col].values)
        labels = self.fit(embeddings)

        df = df.copy()
        df[DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key
        df.loc[valid.index, DataFrameKeys.cluster_type_int] = labels
        return df
