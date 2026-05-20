"""Cluster naming via LLM.

Provides :func:`generate_cluster_name` for naming a single cluster, and
:func:`generate_cluster_name_for_single_rows` for naming many single-row
clusters concurrently with bounded parallelism.

Layering
--------
Imports from ``src.llm.client``, ``src.constants``, ``src.logger``, and
``src.prompts``.  No imports from ``src.helpers``.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.constants import DataFrameKeys
from src.llm import prompts
from src.llm.client import NameClusteringResult, QgenieModels
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["generate_cluster_name", "generate_cluster_name_for_single_rows"]

_name_parser = JsonOutputParser(pydantic_object=NameClusteringResult)


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
    return await chain.ainvoke({"logs": logs})


@execution_timer
async def generate_cluster_name_for_single_rows(df_subset):
    semaphore = asyncio.Semaphore(10)

    async def process_with_semaphore(row):
        async with semaphore:
            result = await generate_cluster_name(row)
            return result["cluster_name"]

    tasks = [process_with_semaphore(row) for _, row in df_subset.iterrows()]
    return await asyncio.gather(*tasks)
