from collections import defaultdict

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from qgenie.integrations.langchain import QGenieChat
from src import prompts
from src.constants import QGENEIE_API_KEY

model = QGenieChat(model="Pro", api_key=QGENEIE_API_KEY, temperature=0.3)


class ClusteringResult(BaseModel):
    cluster_name: str = Field(description="Name to the whole cluster")
    misclassified_ids: str = Field(description="Misclassified erros ids in that cluster")


class MergeResult(BaseModel):
    merged_name: str = Field(description="Name for the merged cluster")
    outlier_indices: list[int] = Field(description="Indices of logs that do not belong in the merged cluster")


class ReclusterResult(BaseModel):
    clusters: list[dict] = Field(description="List of clusters with name and log indices")


parser = JsonOutputParser(pydantic_object=ClusteringResult)
merge_parser = JsonOutputParser(pydantic_object=MergeResult)
recluster_parser = JsonOutputParser(pydantic_object=ReclusterResult)


def analyze_cluster(cluster_df):
    # Prepare input text
    error_logs = [{"id": int(idx), "error log": row["preprocessed_reason"]} for idx, row in cluster_df.iterrows()]
    # Sample prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.CLUSTERING_SYS_MESSAGE), ("human", prompts.CLUSTERING_LOG_MESSAGE)]
    )

    # Get response
    chain = prompt_template | model | parser
    response = chain.invoke({"error_logs": error_logs})
    return response


def merge_clusters(df: pd.DataFrame, cluster_id_a: int, cluster_id_b: int):
    df_a = df[df["cosine_label_grouped"] == cluster_id_a]
    df_b = df[df["cosine_label_grouped"] == cluster_id_b]

    logs_a = [{"index": int(idx), "log": row["preprocessed_reason"]} for idx, row in df_a.iterrows()]
    logs_b = [{"index": int(idx), "log": row["preprocessed_reason"]} for idx, row in df_b.iterrows()]

    chain = prompts.MERGE_PROMPT_TEMPLATE | model | merge_parser
    response = chain.invoke({"id_a": cluster_id_a, "logs_a": logs_a, "id_b": cluster_id_b, "logs_b": logs_b})

    return response


def get_clusters_name_and_misclassified_errors(df: pd.DataFrame):
    results = {}
    for cluster_id in df["cosine_label_grouped"].unique():
        # avoid the miscellaneous cluster, that will be dealt with later
        if cluster_id == -1:
            continue
        cluster_df = df[df["cosine_label_grouped"] == cluster_id]

        result_json = analyze_cluster(cluster_df)
        results[int(cluster_id)] = result_json
    return results


def get_duplicate_clusters(result):
    # Group cluster IDs by their assigned names
    name_to_clusters = defaultdict(list)
    for cluster_id, result in result.items():
        name_to_clusters[result["cluster_name"]].append(cluster_id)

    # Filter only names with duplicates
    duplicate_names = {name: ids for name, ids in name_to_clusters.items() if len(ids) > 1}
    return duplicate_names


def merge_duplicate_clusters(df: pd.DataFrame, duplicate_clusters: dict, cluster_results: dict):
    for duplicate_name, cluster_ids in duplicate_clusters.items():
        # Start with the first cluster as base
        base_cluster_id = cluster_ids[0]

        for next_cluster_id in cluster_ids[1:]:
            print(f"Merging {base_cluster_id} and {next_cluster_id} for name '{duplicate_name}'")

            # Get GPT response
            response = merge_clusters(base_cluster_id, next_cluster_id)

            # Update base cluster name
            cluster_results[base_cluster_id]["cluster_name"] = response["merged_name"]

            # Move all rows from next_cluster to base_cluster
            df.loc[df["cosine_label_grouped"] == next_cluster_id, "cosine_label_grouped"] = base_cluster_id

            # Move outliers to cluster -1
            outlier_indices = response["outlier_indices"]
            df.loc[df.index.isin(outlier_indices), "cosine_label_grouped"] = -1

            # Remove next_cluster from results
            if next_cluster_id in cluster_results:
                del cluster_results[next_cluster_id]

    return df


def give_cluster_names_and_reassign_misc_clusters(df: pd.DataFrame, cluster_results: dict):
    for cluster_id, result in cluster_results.items():
        misclassified_ids = result.get("misclassified_ids", [])
        if misclassified_ids:
            df.loc[df.index.isin(misclassified_ids), "cosine_label_grouped"] = -1

        cluster_name = result.get("cluster_name")
        if cluster_name:
            df.loc[df["cosine_label_grouped"] == cluster_id, "cosine_label_grouped"] = cluster_name

    return df


def recluster_with_context(df: pd.DataFrame) -> list[dict]:
    unclustered_df = df[df["cosine_label_grouped"] == -1]
    error_logs = [
        {"index": int(idx), "error log": row["preprocessed_reason"]} for idx, row in unclustered_df.iterrows()
    ]
    chain = prompts.RECLUSTERING_PROMPT | model | recluster_parser
    outliers_recluster_results = chain.invoke({"error_logs": error_logs})

    for result in outliers_recluster_results:
        name, indices = result["cluster_name"], result["log_indices"]
        if indices:
            df.loc[df.index.isin(indices), "cosine_label_grouped"] = name

    return df
