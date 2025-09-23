import asyncio
import logging
import time
import traceback
from collections import defaultdict
from typing import Any

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, Field

from qgenie.integrations.langchain import QGenieChat
from src import helpers, prompts
from src.constants import QGENEIE_API_KEY, ClusterSpecificKeys, DataFrameKeys
from src.execution_timer_log import execution_timer

logger = logging.getLogger(__name__)


class CustomQGenieChat(QGenieChat):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        retry_count = 3
        while retry_count:
            try:
                message_dicts, params = self._create_message_dicts(messages)
                params = {**params, **kwargs}
                params.pop("stream", "")
                response = self.client.chat(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                logger.error(traceback.format_exc())
                retry_count -= 1
                time.sleep(4)
                continue

        return self._create_chat_result({})

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        retry_count = 3
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        params.pop("stream", "")

        while retry_count:
            try:
                response = await self.async_client.chat(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                logger.error(traceback.format_exc())
                retry_count -= 1
                await asyncio.sleep(4)
                continue
        return self._create_chat_result({})


model = CustomQGenieChat(model="Pro", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=200)


class ClusteringResult(BaseModel):
    cluster_name: str = Field(description="Name to the whole cluster")
    misclassified_ids: str = Field(description="Misclassified erros ids in that cluster")


class NameClusteringResult(BaseModel):
    cluster_name: str = Field(description="Name to the whole cluster")


class MergeResult(BaseModel):
    merged_name: str = Field(description="Name for the merged cluster")
    outlier_indices: list[int] = Field(description="Indices of logs that do not belong in the merged cluster")


class ReclusterResult(BaseModel):
    clusters: list[dict] = Field(description="List of clusters with name and log indices")


parser = JsonOutputParser(pydantic_object=ClusteringResult)
merge_parser = JsonOutputParser(pydantic_object=MergeResult)
recluster_parser = JsonOutputParser(pydantic_object=ReclusterResult)
nameparser = JsonOutputParser(pydantic_object=NameClusteringResult)


@execution_timer
def generate_cluster_name(grouped_cluster: pd.DataFrame) -> dict:
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTER_NAMING_SYS_MESSAGE), ("human", prompts.CLUSTER_NAMING_LOG_MESSAGE)]
    )
    logs = grouped_cluster[DataFrameKeys.preprocessed_text_key].tolist()
    logs = logs[:5] if len(logs) > 5 else logs
    chain = prompt_template | model | nameparser
    response = chain.invoke({"logs": logs})
    return response


@execution_timer
async def analyze_cluster(cluster_df: pd.DataFrame) -> dict:
    def chunk_logs(logs, chunk_size=30, overlap=10):
        start = 0
        while start < len(logs):
            end = start + chunk_size
            yield logs[start:end]
            start = end - overlap  # move back by `overlap` logs

    async def process_chunk(chunk):
        return await chain.ainvoke({"error_logs": chunk})

    error_logs = [
        {"id": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in cluster_df.iterrows()
    ]

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTERING_SYS_MESSAGE), ("human", prompts.CLUSTERING_LOG_MESSAGE)]
    )

    chain = prompt_template | model | parser

    tasks = [process_chunk(chunk) for chunk in chunk_logs(error_logs)]
    responses = await asyncio.gather(*tasks)
    logger.info(f"Analyze clusters response: {responses}")
    if not responses:
        responses = [{}]

    # due to overlap there can be duplicate ids, so using set of list to get only unique of those ids

    misclassified_ids = [
        id_ for response in responses if response.get("misclassified_ids") for id_ in response["misclassified_ids"]
    ]

    responses = {"cluster_name": responses[0].get("cluster_name"), "misclassified_ids": list(set(misclassified_ids))}

    return responses


@execution_timer
def merge_clusters(
    df: pd.DataFrame,
    cluster_id_a: int = None,
    cluster_id_b: int = None,
    cluster_name_a: str = None,
    cluster_name_b: str = None,
) -> MergeResult:
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

    chain = ChatPromptTemplate.from_template(prompts.MERGE_PROMPT_TEMPLATE) | model | merge_parser
    response = chain.invoke({"id_a": cluster_id_a, "logs_a": logs_a, "id_b": cluster_id_b, "logs_b": logs_b})

    return response


@execution_timer
async def get_clusters_name_and_misclassified_errors(df: pd.DataFrame) -> dict:
    results = {}
    for cluster_id in df[DataFrameKeys.cluster_type_int].unique():
        # avoid the miscellaneous cluster, that will be dealt with later
        if cluster_id == ClusterSpecificKeys.non_grouped_key:
            continue

        cluster_df = df[df[DataFrameKeys.cluster_type_int] == cluster_id]
        result = await analyze_cluster(cluster_df)
        results[int(cluster_id)] = result
    return results


def get_duplicate_clusters(results: dict) -> dict:
    # Group cluster IDs by their assigned names
    name_to_clusters = defaultdict(list)
    for cluster_id, result in results.items():
        name_to_clusters[result["cluster_name"]].append(cluster_id)

    # Filter only names with duplicates
    duplicate_names = {name: ids for name, ids in name_to_clusters.items() if len(ids) > 1}
    return duplicate_names


@execution_timer
def merge_duplicate_clusters(
    df: pd.DataFrame, duplicate_clusters: dict, cluster_results: dict
) -> tuple[pd.DataFrame, dict]:
    for duplicate_name, cluster_ids in duplicate_clusters.items():
        # Start with the first cluster as base
        base_cluster_id = cluster_ids[0]

        for next_cluster_id in cluster_ids[1:]:
            print(f"Merging {base_cluster_id} and {next_cluster_id} for name '{duplicate_name}'")
            response = merge_clusters(df, cluster_id_a=base_cluster_id, cluster_id_b=next_cluster_id)

            # Update base cluster name
            cluster_results[base_cluster_id]["cluster_name"] = response["merged_name"]

            # Move all rows from next_cluster to base_cluster
            df.loc[df[DataFrameKeys.cluster_type_int] == next_cluster_id, DataFrameKeys.cluster_type_int] = (
                base_cluster_id
            )

            # Move outliers to cluster -1
            outlier_indices = [int(index) for index in response.get("outlier_indices")]
            df.loc[outlier_indices, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key

            # Remove next_cluster from results
            if next_cluster_id in cluster_results:
                del cluster_results[next_cluster_id]

    return df, cluster_results


@execution_timer
def give_cluster_names_and_reassign_misc_clusters(df: pd.DataFrame, cluster_results: dict) -> pd.DataFrame:
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
def recluster_with_context(df: pd.DataFrame) -> pd.DataFrame:
    unclustered_df = df[df[DataFrameKeys.cluster_type_int] == ClusterSpecificKeys.non_grouped_key]
    error_logs = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]}
        for idx, row in unclustered_df.iterrows()
    ]
    chain = ChatPromptTemplate.from_template(prompts.RECLUSTERING_PROMPT) | model | recluster_parser
    outliers_recluster_results = chain.invoke({"error_logs": error_logs})
    # Create a set of valid indices from the DataFrame for efficient lookup
    df_index_set = set(df.index)

    for result in outliers_recluster_results:
        name, indices = result.get("cluster_name"), result.get("log_indices")
        if name and indices:
            if not isinstance(indices, list):
                # If it's a single value, ensure it's convertible to int and put in a list
                try:
                    indices = [int(indices)]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid single index received: {indices}")
                    continue
            else:
                # Filter out non-integer values and convert to int
                indices = [
                    int(idx) for idx in indices if isinstance(idx, (int, float))
                ]  # ensure is int or float before conversion

            # Filter out indices that are not present in the DataFrame's current index
            valid_indices = [idx for idx in indices if idx in df_index_set]

            if valid_indices:  # Only proceed if there are valid indices to update
                df.loc[valid_indices, DataFrameKeys.cluster_name] = name
                # 200 specifies the reclustered grouped using gpt in which each cluster doesn't belong to new cluster id
                # instead same cluster id is defined but cluster names will be different
                df.loc[valid_indices, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.default_cluster_key
            else:
                logger.warning(f"No valid indices found in DataFrame for recluster result: {result}")
        else:
            logger.warning(f"Missing name or indices in recluster result: {result}")

    return df


# TODO: Test this and fix the input token exceed issue as well
@execution_timer
def inter_cluster_merging(df: pd.DataFrame) -> pd.DataFrame:
    unique_clusters = df[DataFrameKeys.cluster_name].unique()
    clusters_to_consider = [c for c in unique_clusters if c != ClusterSpecificKeys.non_grouped_key]

    # Build cluster dictionary with top logs
    cluster_dict = {
        cluster_name: {
            "name": cluster_name,
            "logs": df[df[DataFrameKeys.cluster_name] == cluster_name][DataFrameKeys.preprocessed_text_key]
            .head()
            .tolist(),
        }
        for cluster_name in clusters_to_consider
    }

    processed_clusters = set()

    for current_cluster_name in list(cluster_dict.keys()):
        if current_cluster_name in processed_clusters:
            continue

        # Prepare other clusters excluding the current one
        other_clusters = {
            name: data
            for name, data in cluster_dict.items()
            if name != current_cluster_name and name not in processed_clusters
        }

        if not other_clusters:
            continue

        try:
            # Call external function to decide merge target
            response = cluster_merge_qgenie(current_cluster_name=current_cluster_name, other_clusters=other_clusters)

            merge_target_name = response.get("cluster_name")

            if merge_target_name and merge_target_name != current_cluster_name:
                print(f"Merging '{current_cluster_name}' into '{merge_target_name}'")

                # Perform the merge
                merge_response = merge_clusters(df, current_cluster_name, merge_target_name)

                # Update cluster labels in df
                df.loc[df[DataFrameKeys.cluster_name] == current_cluster_name, DataFrameKeys.cluster_name] = (
                    merge_response["merged_name"]
                )

                # Handle outliers
                outlier_indices = merge_response.get("outlier_indices", [])
                if outlier_indices:
                    df.loc[outlier_indices, DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key

                # Update merged cluster info
                updated_logs = (
                    df[df[DataFrameKeys.cluster_name] == merge_target_name][DataFrameKeys.preprocessed_text_key]
                    .head()
                    .tolist()
                )
                cluster_dict[merge_target_name]["logs"] = updated_logs
                cluster_dict[merge_target_name]["name"] = merge_response.get("merged_name", merge_target_name)

                # Mark both clusters as processed
                processed_clusters.update({current_cluster_name, merge_target_name})

                # Remove merged-from cluster
                cluster_dict.pop(current_cluster_name, None)

        except Exception as e:
            logger.error(f"Error merging cluster '{current_cluster_name}': {e}")
            traceback.print_exc()

    return df


@execution_timer
async def qgenie_post_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Perform post-processing on the dataframe to handle outlier clusters.

    This function identifies outlier clusters, retrieves GPT responses for them,
    and updates the dataframe accordingly.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data.

    Returns:
    - pd.DataFrame: The dataframe with post-processed data.
    """
    try:
        analyzed_results = await get_clusters_name_and_misclassified_errors(df)
        if analyzed_results:
            duplicate_clusters = get_duplicate_clusters(analyzed_results)
            df, analyzed_results = merge_duplicate_clusters(df, duplicate_clusters, analyzed_results)
            df = give_cluster_names_and_reassign_misc_clusters(df, analyzed_results)

        df = helpers.detect_cluster_outlier(df)
        df = helpers.reassign_unclustered_logs(df)
        df = recluster_with_context(df)
    except Exception as e:
        logger.error(f"Exception in Qgenie post processing: {e}")
        traceback.print_exc()

    logger.info(f"Qgenie Post Processing clusters: {df[DataFrameKeys.cluster_name].unique()}")
    return df
