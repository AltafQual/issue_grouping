import asyncio
import json
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
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)


def get_exponential_backoff_delay(attempt: int, base_delay: int = 1, max_delay: int = 60) -> int:
    """
    Returns the number of seconds to wait before the next retry using exponential backoff.

    Parameters:
    - attempt: The current retry attempt (starting from 1).
    - base_delay: The initial delay in seconds.
    - max_delay: The maximum delay allowed.

    Returns:
    - An integer number of seconds to wait.
    """
    delay = base_delay * (2 ** (attempt - 1))
    return min(delay, max_delay)


class CustomQGenieChat(QGenieChat):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        retry_count = 5
        while retry_count:
            try:
                message_dicts, params = self._create_message_dicts(messages)
                params = {**params, **kwargs}
                params.pop("stream", "")
                response = self.client.chat(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"for messages: {message_dicts}")
                retry_count -= 1
                time.sleep(get_exponential_backoff_delay(attempt=retry_count))
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
        retry_count = 5
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        params.pop("stream", "")

        while retry_count:
            try:
                response = await self.async_client.chat(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"for messages: {message_dicts}")
                retry_count -= 1
                await asyncio.sleep(get_exponential_backoff_delay(attempt=retry_count))
                continue
        return self._create_chat_result({})


# pro_model = CustomQGenieChat(model="Pro", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=200)
gemini_pro_model = CustomQGenieChat(
    model="vertexai::gemini-2.5-pro", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=200
)
model = CustomQGenieChat(
    model="vertexai::gemini-2.5-flash", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=200
)


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


class ClassifyClusterGroup(BaseModel):
    environment_issue: bool = Field(description="Whether this cluster is an environment issue or not")
    code_failure: bool = Field(description="Whether this cluster is a code failure or not")
    sdk_issue: bool = Field(description="Whether this is a sdk related issue or not")


class SubClusterVerifierFailed(BaseModel):
    cluster_name: str = Field(description="name of the subcluster")
    indices: list[int] = Field(description="index of the logs that belong to that cluster")
    previous_clusters: dict = Field(description="json of existing verifier failed clusters for regrouping")


parser = JsonOutputParser(pydantic_object=ClusteringResult)
merge_parser = JsonOutputParser(pydantic_object=MergeResult)
recluster_parser = JsonOutputParser(pydantic_object=ReclusterResult)
nameparser = JsonOutputParser(pydantic_object=NameClusteringResult)
classify_cluster_based_on_type = JsonOutputParser(pydantic_object=ClassifyClusterGroup)
subcluster_verifer_failed = JsonOutputParser(pydantic_object=SubClusterVerifierFailed)


@execution_timer
def classify_cluster_based_of_type(cluster_logs: list[str]) -> dict:
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLASSIFY_CLUSTER_TYPE_SYS_MESSAGE), ("human", prompts.CLASSIFY_CLUSTER_TYPE_LOG_MESSAGE)]
    )
    chain = prompt_template | model | classify_cluster_based_on_type
    result = chain.invoke({"logs": cluster_logs})
    return result


@execution_timer
def generate_cluster_name(grouped_cluster: pd.DataFrame) -> dict:
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTER_NAMING_SYS_MESSAGE), ("human", prompts.CLUSTER_NAMING_LOG_MESSAGE)]
    )

    logs = grouped_cluster[DataFrameKeys.preprocessed_text_key]
    if isinstance(logs, pd.Series):
        logs = logs.tolist()
    else:
        logs = [logs]
    logs = logs[:5] if len(logs) > 5 else logs
    chain = prompt_template | model | nameparser
    response = chain.invoke({"logs": logs})
    return response


@execution_timer
async def analyze_cluster(cluster_df: pd.DataFrame) -> dict:
    def chunk_logs(logs, chunk_size=50, overlap=3):
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

    chain = prompt_template | gemini_pro_model | parser

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

    chain = ChatPromptTemplate.from_template(prompts.MERGE_PROMPT_TEMPLATE) | gemini_pro_model | merge_parser
    response = chain.invoke({"id_a": cluster_id_a, "logs_a": logs_a, "id_b": cluster_id_b, "logs_b": logs_b})

    return response


@execution_timer
async def async_merge_clusters(
    df: pd.DataFrame,
    cluster_id_a: int = None,
    cluster_id_b: int = None,
    cluster_name_a: str = None,
    cluster_name_b: str = None,
) -> MergeResult:
    def chunk_logs(logs, chunk_size=50, overlap=3):
        start = 0
        while start < len(logs):
            end = start + chunk_size
            yield logs[start:end]
            start = end - overlap

    async def process_chunk(chunk, id_a, id_b):
        return await chain.ainvoke({"id_a": id_a, "logs_a": chunk["logs_a"], "id_b": id_b, "logs_b": chunk["logs_b"]})

    # Select clusters
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

    # Prepare prompt chain
    chain = ChatPromptTemplate.from_template(prompts.MERGE_PROMPT_TEMPLATE) | gemini_pro_model | merge_parser

    # Chunk logs to avoid token overflow
    logs_a_chunks = list(chunk_logs(logs_a))
    logs_b_chunks = list(chunk_logs(logs_b))

    tasks = []
    for chunk_a in logs_a_chunks:
        for chunk_b in logs_b_chunks:
            tasks.append(process_chunk({"logs_a": chunk_a, "logs_b": chunk_b}, cluster_id_a, cluster_id_b))

    responses = await asyncio.gather(*tasks)

    logger.info(f"Merged cluster response: {responses}")
    name = ""
    indices = set()
    if responses and len(responses) > 1:
        name = responses[0].get("merged_name")
        for response in responses:
            indices.update([int(idx) for idx in response.get("outlier_indices", [])])

    indices = list(indices)
    return {"merged_name": name, "outlier_indices": indices}


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
async def merge_duplicate_clusters(
    df: pd.DataFrame, duplicate_clusters: dict, cluster_results: dict
) -> tuple[pd.DataFrame, dict]:
    for duplicate_name, cluster_ids in duplicate_clusters.items():
        # Start with the first cluster as base
        base_cluster_id = cluster_ids[0]

        for next_cluster_id in cluster_ids[1:]:
            print(f"Merging {base_cluster_id} and {next_cluster_id} for name '{duplicate_name}'")

            # TODO: getting input token exceed error, have to update this
            response = await async_merge_clusters(df, cluster_id_a=base_cluster_id, cluster_id_b=next_cluster_id)

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
def subcluster_verifier_failed(df: pd.DataFrame):
    """
    Iteratively sub-cluster logs in batches of n using the SUBCLUSTER_VERIFIER_FAILED prompts.
    Maintains a previous_clusters registry across batches, and within each batch continues to
    refine subclusters until no new indices are added or a safety cap is hit.

    Returns:
    - dict[str, list[int]]: final consolidated mapping of subcluster name -> indices
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    logs = [{"index": int(idx), "error log": row[DataFrameKeys.error_reason]} for idx, row in df.iterrows()]

    # Helper to chunk logs into non-overlapping batches of 50
    def chunk_logs(items, size=30):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    # Build the prompt chain once
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.SUBCLUSTER_VERIFIER_FAILED_SYS_MESSAGE),
            ("human", prompts.SUBCLUSTER_VERIFIER_FAILED_LOG_MESSAGE),
        ]
    )
    chain = prompt_template | model | subcluster_verifer_failed

    # previous_clusters as a dict[str, set[int]] for consolidation
    previous_clusters_agg: dict[str, set[int]] = {}

    # For validation
    df_index_set = set(int(i) for i in df.index)

    # Process in 50-log batches
    for batch in chunk_logs(logs):
        if not batch:
            continue

        batch_index_set = {int(item["index"]) for item in batch}
        clustered_in_batch: set[int] = set()

        # Safety cap: at most 5 model iterations per batch
        max_iters = 5
        for _ in range(max_iters):
            # Compute globally assigned indices (across all previous batches)
            global_assigned_indices = set()
            for idxs in previous_clusters_agg.values():
                global_assigned_indices.update(idxs)

            # Remaining indices to consider in this batch
            remaining_in_batch = batch_index_set - clustered_in_batch - global_assigned_indices
            if not remaining_in_batch:
                break  # no work left in this batch

            # Prepare current batch logs excluding already-assigned indices
            current_batch_logs = [item for item in batch if int(item["index"]) in remaining_in_batch]

            previous_clusters_json = json.dumps(
                {name: sorted(list(indices)) for name, indices in previous_clusters_agg.items()},
                indent=2,
            )
            batch_logs_json = json.dumps(current_batch_logs, ensure_ascii=False, indent=2)

            try:
                result = chain.invoke({"previous_clusters": previous_clusters_json, "error_logs": batch_logs_json})
            except Exception as e:
                logger.error(f"Subcluster verifier failed invocation error: {e}")
                break

            cluster_name = (result or {}).get("cluster_name") or ""
            indices = (result or {}).get("indices") or []
            returned_prev = (result or {}).get("previous_clusters") or {}

            # Coerce indices to ints and validate
            try:
                indices_int = {int(i) for i in indices}
            except Exception:
                indices_int = set()

            # Only keep indices that are valid for this df and not already assigned globally,
            # and are still remaining in this batch
            indices_int = {
                i
                for i in indices_int
                if i in df_index_set and i in remaining_in_batch and i not in global_assigned_indices
            }

            # If no progress (no indices or no cluster name), still merge structural updates and exit iteration
            new_indices = indices_int - clustered_in_batch
            if not cluster_name or not new_indices:
                if isinstance(returned_prev, dict):
                    for name, idxs in returned_prev.items():
                        try:
                            idxs_int = {int(i) for i in idxs if int(i) in df_index_set}
                        except Exception:
                            idxs_int = set()
                        # Exclude globally assigned duplicates
                        idxs_int -= global_assigned_indices
                        if idxs_int:
                            previous_clusters_agg.setdefault(name, set()).update(idxs_int)
                break

            # Merge returned_prev after filtering out already assigned indices
            if isinstance(returned_prev, dict):
                for name, idxs in returned_prev.items():
                    try:
                        idxs_int = {int(i) for i in idxs if int(i) in df_index_set}
                    except Exception:
                        idxs_int = set()
                    idxs_int -= global_assigned_indices
                    if idxs_int:
                        previous_clusters_agg.setdefault(name, set()).update(idxs_int)

            # Ensure the chosen cluster_name reflects the new indices as well
            previous_clusters_agg.setdefault(cluster_name, set()).update(new_indices)

            # Mark progress within this batch — assigned indices won't be reconsidered next iterations
            clustered_in_batch.update(new_indices)

            # If we've covered the whole batch, stop iterating
            if clustered_in_batch == batch_index_set:
                break

    # Finalize: identify any indices that were never assigned to any subcluster and assign to "-1"
    all_assigned = set()
    for idxs in previous_clusters_agg.values():
        all_assigned.update(idxs)

    # Consider the whole df index set as the universe for this function
    unassigned = sorted(list(df_index_set - all_assigned))
    if unassigned:
        previous_clusters_agg.setdefault(str(ClusterSpecificKeys.non_grouped_key), set()).update(unassigned)

    finalized = {name: sorted(list(indices)) for name, indices in previous_clusters_agg.items()}
    logger.info(f"Subcluster verifier finalized clusters: {list(finalized.keys())}")

    for cluster_name, indices in finalized.items():
        if cluster_name != str(ClusterSpecificKeys.non_grouped_key):
            df.loc[indices, DataFrameKeys.cluster_name] = cluster_name
        else:
            df.loc[indices, DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key

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
            df, analyzed_results = await merge_duplicate_clusters(df, duplicate_clusters, analyzed_results)
            df = give_cluster_names_and_reassign_misc_clusters(df, analyzed_results)

        df = helpers.detect_cluster_outlier(df)
        df = helpers.reassign_unclustered_logs(df)
        df = recluster_with_context(df)
    except Exception as e:
        logger.error(f"Exception in Qgenie post processing: {e}")
        traceback.print_exc()

    logger.info(f"Qgenie Post Processing clusters: {df[DataFrameKeys.cluster_name].unique()}")
    return df
