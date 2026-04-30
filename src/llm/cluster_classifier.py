"""Cluster type classification via LLM.

:class:`ClusterClassifier` asks the LLM to assign a class label
(``environment_issue``, ``setup_issue``, ``sdk_issue``) to a cluster based on
representative error logs.  It replaces the module-level
``classify_cluster_based_of_type()`` function and the ``assign_cluster_class()``
orchestration function from the old ``helpers.py``.

Layering
--------
Imports from ``src.llm.client``, ``src.constants``, ``src.logger``,
``src.prompts``, and standard library / pandas.  No imports from
``src.helpers``.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.constants import ClusterSpecificKeys, DataFrameKeys, ErrorLogConfigurations
from src.llm import prompts
from src.llm.client import ClassifyClusterGroup, QgenieModels
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["ClusterClassifier", "classify_cluster_based_of_type", "get_cluster_class", "assign_cluster_class"]

_classify_parser = JsonOutputParser(pydantic_object=ClassifyClusterGroup)


class ClusterClassifier:
    """Classifies error-log clusters into coarse fault categories.

    The LLM returns a :class:`~src.llm.client.ClassifyClusterGroup` response
    with boolean flags for ``environment_issue``, ``setup_issue``, and
    ``sdk_issue``.  The first ``True`` flag becomes the cluster's class label;
    if none are set the default is ``"sdk_issue"``.

    Args:
        concurrency: Maximum number of simultaneous LLM classification calls.
            Default: 6 (matches the semaphore in the original code).

    Example:
        >>> classifier = ClusterClassifier()
        >>> df_classified = await classifier.classify_dataframe(df)
    """

    def __init__(self, concurrency: int = 6) -> None:
        self.concurrency = concurrency

    async def classify_cluster(self, logs: list[str], cluster_name: str = "") -> dict:
        """Ask the LLM to classify a single cluster.

        Args:
            logs: List of preprocessed log strings (up to 5 recommended).
            cluster_name: Optional cluster name for additional context.

        Returns:
            Dict with boolean keys ``environment_issue``, ``setup_issue``,
            ``sdk_issue``.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.CLASSIFY_CLUSTER_TYPE_SYS_MESSAGE),
                ("human", prompts.CLASSIFY_CLUSTER_TYPE_LOG_MESSAGE),
            ]
        )
        chain = prompt_template | QgenieModels.azure_gpt_5_4_mini | _classify_parser
        return await chain.ainvoke({"logs": logs, "cluster_name": cluster_name})

    async def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify every unclassified cluster in *df*.

        Skips clusters that already have a class, the empty/no-error sentinel
        clusters, and the noise cluster.  Applies classification results back
        to all rows sharing the same cluster name.

        Args:
            df: DataFrame with ``clusters`` and ``preprocessed_reason`` columns.

        Returns:
            Copy of *df* with :attr:`~src.constants.DataFrameKeys.cluster_class`
            column populated.
        """
        if df is None or df.empty:
            return df

        if DataFrameKeys.cluster_class not in df.columns:
            df = df.copy()
            df[DataFrameKeys.cluster_class] = pd.array([np.nan] * len(df), dtype=object)

        exclude = {ErrorLogConfigurations.empty_error, ErrorLogConfigurations.no_error}
        clusters_needing_class = []

        for name in df[DataFrameKeys.cluster_name].dropna().unique():
            if name in exclude:
                continue
            existing_class = df[df[DataFrameKeys.cluster_name] == name].iloc[0][DataFrameKeys.cluster_class]
            if (
                pd.isna(existing_class)
                or existing_class == ClusterSpecificKeys.non_grouped_key
                or (isinstance(existing_class, str) and not existing_class.strip())
            ):
                clusters_needing_class.append(name)

        logger.info(f"[ClusterClassifier] {len(clusters_needing_class)} clusters need classification.")
        if not clusters_needing_class:
            return df

        semaphore = asyncio.Semaphore(self.concurrency)
        results: dict[str, str] = {}

        async def _classify_one(cluster_name: str) -> None:
            async with semaphore:
                cluster_df = df[df[DataFrameKeys.cluster_name] == cluster_name]
                if cluster_df.empty:
                    return
                logs = (
                    cluster_df[DataFrameKeys.preprocessed_text_key]
                    .dropna()
                    .apply(lambda x: x if isinstance(x, str) else "")
                    .tolist()
                )
                logs = [s for s in logs if s.strip()][:5]
                if not logs:
                    return
                try:
                    result = await self.classify_cluster(logs, cluster_name)
                    results[cluster_name] = self._extract_class_label(result)
                    logger.info(f"[ClusterClassifier] '{cluster_name}' → '{results[cluster_name]}'")
                except Exception as exc:
                    logger.error(f"[ClusterClassifier] Failed for '{cluster_name}': {exc}")

        await asyncio.gather(*[_classify_one(n) for n in clusters_needing_class])

        df = df.copy()
        for cluster_name, class_label in results.items():
            df.loc[df[DataFrameKeys.cluster_name] == cluster_name, DataFrameKeys.cluster_class] = class_label

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_class_label(result: dict) -> str:
        """Convert the LLM boolean result dict to a single class string.

        Returns the first ``True`` key, or ``"sdk_issue"`` as default.

        Args:
            result: Dict with boolean classification flags.

        Returns:
            Class label string.
        """
        if not isinstance(result, dict):
            return "sdk_issue"
        for key in ("environment_issue", "setup_issue", "sdk_issue"):
            if result.get(key):
                return key
        return "sdk_issue"


@execution_timer
async def classify_cluster_based_of_type(cluster_logs: list[str], cluster_name: str = "") -> dict:
    """Classify a cluster into a fault category via LLM.

    Args:
        cluster_logs: Sample error log strings for this cluster.
        cluster_name: Human-readable cluster name (used as context).

    Returns:
        Dict with boolean keys ``environment_issue``, ``setup_issue``,
        ``sdk_issue``.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLASSIFY_CLUSTER_TYPE_SYS_MESSAGE), ("human", prompts.CLASSIFY_CLUSTER_TYPE_LOG_MESSAGE)]
    )
    chain = prompt_template | QgenieModels.azure_gpt_5_4_mini | _classify_parser
    result = await chain.ainvoke({"logs": cluster_logs, "cluster_name": cluster_name})
    return result


def get_cluster_class(cluster_dict):
    if not isinstance(cluster_dict, dict):
        return "sdk_issue"
    for cluster_class in cluster_dict:
        if cluster_dict[cluster_class]:
            return cluster_class
    return "sdk_issue"


@execution_timer
async def assign_cluster_class(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df is None or df.empty:
            return df
        if DataFrameKeys.cluster_class not in df.columns:
            df[DataFrameKeys.cluster_class] = pd.array([np.nan] * len(df), dtype=object)
        exclude_names = {
            ErrorLogConfigurations.empty_error,
            ErrorLogConfigurations.no_error,
        }
        unique_clusters = []
        cluster_names = df[DataFrameKeys.cluster_name].dropna().unique().tolist()
        for c in cluster_names:
            if c in exclude_names:
                continue
            cluster_class = df[df[DataFrameKeys.cluster_name] == c].iloc[0][DataFrameKeys.cluster_class]
            if (
                pd.isna(cluster_class)
                or cluster_class == ClusterSpecificKeys.non_grouped_key
                or (isinstance(cluster_class, str) and not cluster_class.strip())
            ):
                unique_clusters.append(c)
        logger.info(f"Found {len(unique_clusters)} clusters that need classification")
        semaphore = asyncio.Semaphore(6)

        async def classify_cluster(cluster_name):
            async with semaphore:
                logger.info(f"Processing classification for cluster: {cluster_name}")
                cluster_df = df[df[DataFrameKeys.cluster_name] == cluster_name]
                if cluster_df.empty:
                    logger.warning(f"Empty cluster dataframe for {cluster_name}")
                    return None, None
                sample_logs = cluster_df[DataFrameKeys.preprocessed_text_key].dropna().tolist()
                sample_logs = [s for s in sample_logs if isinstance(s, str) and s.strip()][:5]
                if not sample_logs:
                    logger.warning(f"Invalid representative log for cluster {cluster_name}")
                    return None, None
                try:
                    logger.info(f"Sending classification request for cluster: {cluster_name}")
                    result = await classify_cluster_based_of_type(sample_logs, cluster_name=cluster_name)
                    class_str = get_cluster_class(result)
                    logger.info(f"Received classification for {cluster_name}: {class_str}")
                    return cluster_df.index, class_str
                except Exception as e:
                    logger.error(f"Failed to classify cluster '{cluster_name}': {e}")
                    return None, None

        tasks = [classify_cluster(cluster_name) for cluster_name in unique_clusters]
        results = await asyncio.gather(*tasks)
        for result in results:
            if not result:
                continue
            indices, class_str = result
            df.loc[indices, DataFrameKeys.cluster_class] = class_str
            logger.info(f"Updated {len(indices)} rows with class '{class_str}'")
    except Exception as e:
        logger.error(f"assign_cluster_class error: {e}")
        logger.error(traceback.format_exc())
    return df
