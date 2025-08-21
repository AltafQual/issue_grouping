import asyncio
from collections import defaultdict

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from qgenie.integrations.langchain import QGenieChat
from src import helpers, prompts
from src.constants import QGENEIE_API_KEY, ClusterSpecificKeys, DataFrameKeys
from src.execution_timer_log import execution_timer

model = QGenieChat(model="Pro", api_key=QGENEIE_API_KEY, temperature=0.3)


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
async def generate_cluster_name(grouped_cluster: list) -> dict:
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTER_NAMING_SYS_MESSAGE), ("human", prompts.CLUSTER_NAMING_LOG_MESSAGE)]
    )
    logs = grouped_cluster[DataFrameKeys.preprocessed_text_key].tolist()
    logs = logs[:5] if len(logs) > 5 else logs
    chain = prompt_template | model | nameparser
    response = await chain.ainvoke({"logs": logs})
    return response


@execution_timer
async def analyze_cluster(cluster_df: pd.DataFrame) -> dict:
    # Prepare input text
    error_logs = [
        {"id": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in cluster_df.iterrows()
    ]

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTERING_SYS_MESSAGE), ("human", prompts.CLUSTERING_LOG_MESSAGE)]
    )

    chain = prompt_template | model | parser
    response = await chain.ainvoke({"error_logs": error_logs})
    return response


@execution_timer
async def merge_clusters(df: pd.DataFrame, cluster_id_a: int, cluster_id_b: int) -> MergeResult:
    df_a = df[df[DataFrameKeys.cluster_type_int] == cluster_id_a]
    df_b = df[df[DataFrameKeys.cluster_type_int] == cluster_id_b]

    logs_a = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in df_a.iterrows()
    ]
    logs_b = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]} for idx, row in df_b.iterrows()
    ]

    chain = ChatPromptTemplate.from_template(prompts.MERGE_PROMPT_TEMPLATE) | model | merge_parser
    response = await chain.ainvoke({"id_a": cluster_id_a, "logs_a": logs_a, "id_b": cluster_id_b, "logs_b": logs_b})

    return response


@execution_timer
async def get_clusters_name_and_misclassified_errors(df: pd.DataFrame) -> dict:
    tasks = []
    cluster_ids = []

    for cluster_id in df[DataFrameKeys.cluster_type_int].unique():
        # avoid the miscellaneous cluster, that will be dealt with later
        if cluster_id == ClusterSpecificKeys.non_grouped_key:
            continue

        cluster_df = df[df[DataFrameKeys.cluster_type_int] == cluster_id]
        tasks.append(analyze_cluster(cluster_df))
        cluster_ids.append(int(cluster_id))

    results_list = await asyncio.gather(*tasks)
    results = {cluster_id: result for cluster_id, result in zip(cluster_ids, results_list)}
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

        merge_tasks = []
        for next_cluster_id in cluster_ids[1:]:
            print(f"Merging {base_cluster_id} and {next_cluster_id} for name '{duplicate_name}'")
            merge_tasks.append((next_cluster_id, merge_clusters(df, base_cluster_id, next_cluster_id)))

        merge_results = await asyncio.gather(*[task[1] for task in merge_tasks])

        for (next_cluster_id, _), response in zip(merge_tasks, merge_results):
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
async def recluster_with_context(df: pd.DataFrame) -> pd.DataFrame:
    unclustered_df = df[df[DataFrameKeys.cluster_type_int] == ClusterSpecificKeys.non_grouped_key]
    error_logs = [
        {"index": int(idx), "error log": row[DataFrameKeys.preprocessed_text_key]}
        for idx, row in unclustered_df.iterrows()
    ]
    chain = ChatPromptTemplate.from_template(prompts.RECLUSTERING_PROMPT) | model | recluster_parser
    outliers_recluster_results = await chain.ainvoke({"error_logs": error_logs})
    for result in outliers_recluster_results:
        name, indices = result.get("cluster_name"), result.get("log_indices")
        if name and indices:
            if not type(indices) == list:
                indices = [int(indices)]
            else:
                indices = [int(index) for index in indices]

        if name and indices:
            df.loc[indices, DataFrameKeys.cluster_name] = name
            # 200 specifies the reclustered grouped using gpt in which each cluster doesn't belong to new cluster id
            # instead same cluster id is defined but cluster names will be different
            df.loc[indices, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.default_cluster_key

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
    analyzed_results = await get_clusters_name_and_misclassified_errors(df)
    duplicate_clusters = get_duplicate_clusters(analyzed_results)
    df, analyzed_results = await merge_duplicate_clusters(df, duplicate_clusters, analyzed_results)
    df = give_cluster_names_and_reassign_misc_clusters(df, analyzed_results)
    df = await recluster_with_context(df)

    # rename -1 cluster to be as Others
    df.loc[df[DataFrameKeys.cluster_type_int] == ClusterSpecificKeys.non_grouped_key, DataFrameKeys.cluster_name] = (
        "Others"
    )
    return df
