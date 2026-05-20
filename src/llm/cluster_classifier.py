"""Cluster type classification via LLM.

Provides :func:`assign_cluster_class` which asks the LLM to assign a class
label (``environment_issue``, ``setup_issue``, ``sdk_issue``) to each cluster
in a DataFrame using a representative subset of error logs.

Layering
--------
Imports from ``src.llm.client``, ``src.constants``, ``src.logger``,
``src.prompts``, and standard library / pandas.  No imports from
``src.helpers``.
"""

from __future__ import annotations

import asyncio
import traceback

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

__all__ = ["classify_cluster_based_of_type", "get_cluster_class", "assign_cluster_class"]

_classify_parser = JsonOutputParser(pydantic_object=ClassifyClusterGroup)


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
        semaphore = asyncio.Semaphore(15)

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
