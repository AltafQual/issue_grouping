"""Cluster naming via LLM.

:class:`ClusterNamer` encapsulates the logic for asking the LLM to produce a
short, descriptive name for a cluster of error logs.  It replaces the
module-level ``generate_cluster_name()`` and ``analyze_cluster()`` functions
in the old ``qgenie_llm_calls.py``.

Layering
--------
Imports from ``src.llm.client``, ``src.constants``, ``src.logger``, and
``src.prompts``.  No imports from ``src.helpers``.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.constants import ClusterSpecificKeys, DataFrameKeys
from src.llm import prompts
from src.llm.client import ClusteringResult, NameClusteringResult, QgenieModels, get_exponential_backoff_delay
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["ClusterNamer", "generate_cluster_name", "generate_cluster_name_for_single_rows"]

_name_parser = JsonOutputParser(pydantic_object=NameClusteringResult)
_cluster_parser = JsonOutputParser(pydantic_object=ClusteringResult)


class ClusterNamer:
    """Generates concise names for error-log clusters via the LLM.

    Wraps two naming strategies:

    * :meth:`name_cluster` — fast single-call naming using up to 5 representative
      logs (replaces ``generate_cluster_name``).
    * :meth:`analyze_and_name_cluster` — deeper analysis that also identifies
      misclassified entries within a cluster (replaces ``analyze_cluster``).

    Example:
        >>> namer = ClusterNamer()
        >>> result = await namer.name_cluster(cluster_df)
        >>> print(result["cluster_name"])
        "SegFault During HTP Graph Execute"
    """

    async def name_cluster(self, cluster_df: pd.DataFrame) -> dict:
        """Generate a short name for the cluster represented by *cluster_df*.

        Uses up to 5 preprocessed log samples to keep the prompt focused.

        Args:
            cluster_df: DataFrame slice containing the cluster's rows.  Must
                include :attr:`~src.constants.DataFrameKeys.preprocessed_text_key`.

        Returns:
            Dict with key ``"cluster_name"`` (str).
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompts.CLUSTER_NAMING_SYS_MESSAGE), ("human", prompts.CLUSTER_NAMING_LOG_MESSAGE)]
        )
        logs = cluster_df[DataFrameKeys.preprocessed_text_key]
        logs = logs.tolist() if isinstance(logs, pd.Series) else [logs]
        logs = logs[:5]

        chain = prompt_template | QgenieModels.azure_o3_mini | _name_parser
        return await chain.ainvoke({"logs": logs})

    async def analyze_and_name_cluster(self, cluster_df: pd.DataFrame) -> dict:
        """Deeply analyse *cluster_df* to produce a name and identify outliers.

        Processes error logs in overlapping chunks of 50 to stay within token
        limits, then aggregates results.

        Args:
            cluster_df: DataFrame slice containing the cluster's rows.

        Returns:
            Dict with:
            - ``"cluster_name"`` (str) — proposed name
            - ``"misclassified_ids"`` (list[int]) — indices of outlier rows
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompts.CLUSTERING_SYS_MESSAGE), ("human", prompts.CLUSTERING_LOG_MESSAGE)]
        )
        chain = prompt_template | QgenieModels.azure_o3 | _cluster_parser

        error_logs = [
            {"id": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]}
            for idx, row in cluster_df.iterrows()
        ]

        async def _process_chunk(chunk):
            return await chain.ainvoke({"error_logs": chunk})

        chunks = list(self._chunked(error_logs, size=50, overlap=3))
        responses = await asyncio.gather(*[_process_chunk(c) for c in chunks])

        if not responses:
            responses = [{}]

        misclassified_ids = [id_ for r in responses if r.get("misclassified_ids") for id_ in r["misclassified_ids"]]
        return {
            "cluster_name": responses[0].get("cluster_name"),
            "misclassified_ids": list(set(misclassified_ids)),
        }

    async def name_all_clusters(self, df: pd.DataFrame) -> dict:
        """Analyse every cluster in *df* and return names + misclassified IDs.

        Iterates over unique cluster integer IDs, skipping the noise cluster
        (:attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`).  Adds a
        5-second pause after each LLM call when there are more than 5 clusters
        to avoid rate-limit throttling.

        Args:
            df: DataFrame with cluster integer assignments
                (:attr:`~src.constants.DataFrameKeys.cluster_type_int`).

        Returns:
            Dict mapping cluster integer ID → analysis result dict.
        """
        results: dict = {}
        unique_ids = df[DataFrameKeys.cluster_type_int].unique()
        logger.info(f"[ClusterNamer] Naming {len(unique_ids)} clusters.")

        for cid in unique_ids:
            if cid == ClusterSpecificKeys.non_grouped_key:
                continue
            cluster_df = df[df[DataFrameKeys.cluster_type_int] == cid]
            results[int(cid)] = await self.analyze_and_name_cluster(cluster_df)
            if len(unique_ids) > 5:
                await asyncio.sleep(5)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunked(seq: list, size: int, overlap: int = 0):
        """Yield successive overlapping chunks from *seq*."""
        start = 0
        while start < len(seq):
            end = start + size
            yield seq[start:end]
            start = end - overlap


@execution_timer
async def generate_cluster_name(grouped_cluster: pd.DataFrame) -> dict:
    """Generate a cluster name for a single cluster DataFrame via LLM.

    Args:
        grouped_cluster: DataFrame slice for one cluster; must contain
            :attr:`~src.constants.DataFrameKeys.preprocessed_text_key`.

    Returns:
        Dict with ``"cluster_name"`` key.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTER_NAMING_SYS_MESSAGE), ("human", prompts.CLUSTER_NAMING_LOG_MESSAGE)]
    )
    logs = grouped_cluster[DataFrameKeys.preprocessed_text_key]
    if isinstance(logs, pd.Series):
        logs = logs.tolist()
    else:
        logs = [logs]
    logs = logs[:5] if len(logs) > 5 else logs
    chain = prompt_template | QgenieModels.azure_o3_mini | _name_parser
    response = await chain.ainvoke({"logs": logs})
    return response


@execution_timer
async def generate_cluster_name_for_single_rows(df_subset):
    async def process_row(row):
        result = await generate_cluster_name(row)
        return result["cluster_name"]

    semaphore = asyncio.Semaphore(3)
    results = []

    async def process_with_semaphore(row):
        async with semaphore:
            return await process_row(row)

    tasks = [process_with_semaphore(row) for _, row in df_subset.iterrows()]
    results = await asyncio.gather(*tasks)
    return results
