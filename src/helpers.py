import asyncio
import hashlib
import logging
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
from io import BytesIO

import numpy as np
import pandas as pd
import swifter
from apscheduler.schedulers.background import BackgroundScheduler
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor

from src.constants import ClusterSpecificKeys, DataFrameKeys, ErrorLogConfigurations, regex_based_filteration_patterns
from src.db_connections import ConnectToMySql
from src.execution_timer_log import execution_timer
from src.faiss_db import FaissIVFFlatIndex, SearchInExistingFaiss
from src.qgenie import generate_cluster_name

logger = logging.getLogger(__name__)
sql_connection = ConnectToMySql()
faiss_runner = FaissIVFFlatIndex()

# in-memory tc id data cache
_tc_id_cache = {}
parquet_file = "run_ids.parquet"


def add_hashed_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)

    def generate_hash(row):
        raw_string = f"{random.randint(0, df.shape[0])}_{row['type']}_{row['tc_uuid']}_{row['runtime']}_{row['soc_name']}".lower()
        return hashlib.md5(raw_string.encode()).hexdigest()

    df[DataFrameKeys.index] = df.apply(generate_hash, axis=1)
    return df


@execution_timer
def preprocess_error_log(log: str) -> str:
    # Step 1: Remove timestamps like "1077.9ms"
    log = re.sub(r"\b\d{1,4}(\.\d+)?ms\b", "", log)

    # Step 2: Remove line numbers like "9:" or "15 -"
    log = re.sub(r"^\d+[:\-]\s*", "", log, flags=re.MULTILINE)

    # Step 3: Replace newlines in the middle with space
    log = re.sub(r"(?<=\S)\n(?=\S)", " ", log)

    # Step 4: Remove build/version info
    log = re.sub(r"build version:.*?(,|$)", "", log, flags=re.IGNORECASE)
    log = re.sub(r"version:.*?\\b", "", log, flags=re.IGNORECASE)

    # Step 5: Remove brackets, quotes
    log = re.sub(r"[\[\]\'\"`]", " ", log)

    # Step 6: Remove special characters and digits from the start until a letter is found
    log = re.sub(r"^[^a-zA-Z]+", "", log)

    # Step 7: Normalize whitespace
    log = re.sub(r"\s+", " ", log).strip()

    # Step 8: Lowercase for consistency
    return log.lower()


def is_empty_error_log(s):
    if s is None or pd.isna(s) or (isinstance(s, str) and s.lower() in {"null", "nan", "none"}):
        return ErrorLogConfigurations.no_error
    if not bool(re.search(r"[a-zA-Z]", s)):
        return ErrorLogConfigurations.empty_error

    return ClusterSpecificKeys.non_grouped_key


def mask_numbers(text: str) -> str:
    # Mask time formats like 01h, 30m, 20s, 500ms
    time_masked = re.sub(r"\b\d+(?:\.\d+)?\s*(h|hr|hrs|m|min|s|sec|ms)\b", "<TIME>", text, flags=re.IGNORECASE)

    # Mask standalone numbers not part of time units
    final_masked = re.sub(r"(?<!\w)(\d+(\.\d+)?)(?!\w)", "<NUM>", time_masked)

    return final_masked


@execution_timer
def remove_empty_and_misc_rows(df: pd.DataFrame, errors: list, error_column_name: str):

    df[error_column_name] = errors
    # Apply filters
    df = df[
        ~df[error_column_name]
        .str.strip()
        .str.lower()
        .str.startswith("limiting reason to 3000 chars")  # not starting with that phrase
    ]
    # add
    df.loc[:, DataFrameKeys.cluster_name] = df[error_column_name].apply(is_empty_error_log)
    df.loc[:, error_column_name] = df[error_column_name].apply(mask_numbers)
    df = df.reset_index(drop=True)
    return df


def update_rows_by_regex_patterns(df: pd.DataFrame) -> pd.DataFrame:
    for name, pattern in regex_based_filteration_patterns.items():
        matched_df = df[
            df[DataFrameKeys.preprocessed_text_key].astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        ]
        print(f"Found occurence of match: {name}: {matched_df.shape[0]}")
        if not matched_df.empty:
            cluster_name = generate_cluster_name(matched_df)
            df.loc[matched_df.index, DataFrameKeys.cluster_name] = cluster_name["cluster_name"]


@execution_timer
def merge_similar_clusters(embeddings, labels, threshold=0.95):
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


@execution_timer
def reassign_unclustered_logs(df: pd.DataFrame, threshold=0.95):
    # Step 1: Compute centroids for all valid clusters
    valid_clusters = set(df[DataFrameKeys.cluster_name]) - {-1}
    logger.info(f"Valid clusters for reassigning: {valid_clusters}")
    if valid_clusters:
        centroids = {
            label: np.mean(
                np.stack(df[df[DataFrameKeys.cluster_name] == label][DataFrameKeys.embeddings_key].values), axis=0
            )
            for label in valid_clusters
        }
        # Step 2: Reassign -1 cluster logs based on highest similarity
        for idx, row in df[df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key].iterrows():
            log_embedding = row[DataFrameKeys.embeddings_key].reshape(1, -1)
            best_label = None
            best_similarity = -1
            for label, centroid in centroids.items():
                similarity = cosine_similarity(log_embedding, centroid.reshape(1, -1))[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_label = label
            if best_similarity >= threshold:
                logger.info(f"Best label: {best_label}  Best Similarity: {best_similarity}")
                df.at[idx, DataFrameKeys.cluster_name] = best_label

    return df


@execution_timer
def update_labels_with_merged_clusters(df: pd.DataFrame, merged_clusters: dict, label_col: str) -> pd.DataFrame:
    # Create a mapping from old label to new (parent) label
    label_map = {}
    for parent, children in merged_clusters.items():
        for child in children:
            label_map[int(child)] = int(parent)

    # Apply the mapping to the DataFrame
    df[label_col] = df[label_col].map(label_map).fillna(df[label_col])
    return df


def trim(log, head_ratio=0.3, max_length=1000):
    try:
        log_str = str(log)
        if len(log_str) > max_length:
            head_length = int(max_length * head_ratio)
            tail_length = max_length - head_length
            head = log_str[:head_length]
            tail = log_str[-tail_length:]
            return head + tail
        else:
            return log_str
    except Exception as e:
        print(f"Error processing log: {log}, Error: {e}")
        return ""


@execution_timer
def trim_error_logs(df: pd.DataFrame, column=DataFrameKeys.preprocessed_text_key) -> pd.DataFrame:
    df[column] = df[column].apply(trim)
    return df


@execution_timer
def group_similar_errors(df: pd.DataFrame, column: str, threshold) -> list:
    groups = []
    assigned = set()

    if df.empty:
        return groups

    index_list = df.index.tolist()
    if len(index_list) <= 1:
        return groups

    for i, base_idx in enumerate(index_list):
        if base_idx in assigned:
            continue

        base = df.at[base_idx, column]
        group = [base_idx]

        for right_idx in index_list[i + 1 :]:
            ratio = fuzz.ratio(base, df.at[right_idx, column])
            if right_idx not in assigned and ratio >= threshold:
                group.append(right_idx)

        if len(group) > 1:
            groups.append(group)
            assigned.update(group)

    return groups


@execution_timer
def fuzzy_cluster_grouping(
    failures_dataframe: pd.DataFrame, threshold=100, bin_intervals=[[0, 50], [50, 110]]
) -> pd.DataFrame:
    failures_dataframe.loc[:, DataFrameKeys.error_logs_length] = failures_dataframe[
        DataFrameKeys.preprocessed_text_key
    ].apply(len)

    # Step 1: Create bin column if not already present
    if DataFrameKeys.bins not in failures_dataframe.columns:
        bin_edges = sorted({edge for interval in bin_intervals for edge in interval})
        failures_dataframe.loc[:, DataFrameKeys.bins] = pd.cut(
            failures_dataframe[DataFrameKeys.error_logs_length], bins=bin_edges, right=True
        )

    for i, j in bin_intervals:
        target_bin = pd.Interval(left=i, right=j, closed="right")
        # Group similar errors
        grouped_indices = group_similar_errors(
            failures_dataframe[failures_dataframe[DataFrameKeys.bins] == target_bin],
            DataFrameKeys.preprocessed_text_key,
            threshold,
        )

        for group in grouped_indices:
            result = generate_cluster_name(failures_dataframe.iloc[group])
            failures_dataframe.loc[group, DataFrameKeys.cluster_name] = result["cluster_name"]

    # regex based common errors mapping
    update_rows_by_regex_patterns(failures_dataframe)
    return failures_dataframe


@execution_timer
def create_excel_with_clusters(df: pd.DataFrame, cluster_column: str, columns_to_include=None) -> pd.ExcelFile:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for cluster in df[cluster_column].unique():
            sheet_name = str(cluster)[:31]
            cluster_df = df[df[cluster_column] == cluster]
            if columns_to_include:
                cluster_df = cluster_df[columns_to_include]
            cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output


def detect_cluster_outlier(df: pd.DataFrame) -> pd.DataFrame:
    clusters = set(df[DataFrameKeys.cluster_name]) - {-1}
    for cluster in clusters:
        cluster_df = df[df[DataFrameKeys.cluster_name] == cluster]
        embedding_matrix = np.vstack(cluster_df[DataFrameKeys.embeddings_key].values)

        # Step 1: Compute cosine similarity to cluster centroid
        centroid = np.mean(embedding_matrix, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embedding_matrix, centroid).flatten()
        cluster_df["cosine_similarity_to_centroid"] = similarities

        # Step 2: Apply Local Outlier Factor
        lof = LocalOutlierFactor(n_neighbors=15, metric="cosine", n_jobs=-1)
        outlier_flags = lof.fit_predict(embedding_matrix)
        cluster_df["lof_outlier"] = outlier_flags  # -1 indicates outlier

        # Step 3: Combine both metrics to identify strong outliers
        # Criteria: low cosine similarity and flagged by LOF
        similarity_threshold = 0.7
        outliers = cluster_df[
            (cluster_df["cosine_similarity_to_centroid"] < similarity_threshold) & (cluster_df["lof_outlier"] == -1)
        ]

        # Output the IDs of the outlier logs
        logger.info(f"For {cluster} Outlier log IDs: {outliers['reason'].tolist()}")
        if not outliers.empty:
            df.loc[outliers.index, DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key

    return df


async def check_if_issue_alread_grouped(df: pd.DataFrame) -> pd.DataFrame:
    # Identify rows that are not yet grouped
    mask = df[DataFrameKeys.cluster_name].isin([ClusterSpecificKeys.non_grouped_key, np.nan])
    ungrouped_df = df[mask]

    if not ungrouped_df.empty:
        # Get cluster names using FAISS
        new_cluster_names = await SearchInExistingFaiss().batch_search(
            type=ungrouped_df.iloc[0]["type"],  # assuming same type for batch
            query=ungrouped_df[DataFrameKeys.preprocessed_text_key].tolist(),
        )

        # Update the original DataFrame
        df.loc[mask, DataFrameKeys.cluster_name] = new_cluster_names
        df.loc[mask, DataFrameKeys.grouped_from_faiss] = True

    return df


@execution_timer
def get_tc_ids_from_sql():
    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        return df
    else:
        run_ids = sql_connection.fetch_runids()
        run_ids.to_parquet(parquet_file)
        return run_ids


@execution_timer
def update_error_map_qgenie_table(df):
    sql_connection.update_qgenie_error_map_table(df)


@execution_timer
def get_error_group_id(error_type: str, runtime: str, cluster_name: str) -> str:
    return sql_connection.get_error_group_id(error_type, runtime, cluster_name)

@execution_timer
def find_regressions_between_two_tests(tc_id_a: str, tc_id_b: str) -> str:
    return sql_connection.get_regressions(tc_id_a, tc_id_b)

def cache_tc_id(func):
    @wraps(func)
    def wrapper(tc_id: str):
        if tc_id in _tc_id_cache:
            return _tc_id_cache[tc_id]
        result = func(tc_id)
        _tc_id_cache[tc_id] = result
        return result

    return wrapper


@execution_timer
@cache_tc_id
def get_tc_id_df(tc_id: str):
    return sql_connection.fetch_result_based_on_runid(tc_id)


def tc_id_scheduler():
    def update_tc_ids():
        logger.info("Running tc ids updation background job")
        run_ids = sql_connection.fetch_runids()
        run_ids.to_parquet(parquet_file)
        logger.info("Background task updated Parquet file")

    scheduler = BackgroundScheduler()
    scheduler.add_job(update_tc_ids, "interval", hours=6)
    scheduler.start()


async def async_process_by_type(df, update_faiss_and_sql=False):
    from src.failure_analyzer import FailureAnalyzer

    results = {}
    analyzer = FailureAnalyzer()

    async def process_group(t, group_df):
        group_df = group_df.reset_index(drop=True)
        result = await analyzer.analyze(dataframe=group_df)
        results[t] = result

    logger.info(f"All types in data: {df.type.unique()}")
    tasks = [process_group(t, group_df) for t, group_df in df.groupby("type")]
    await asyncio.gather(*tasks)

    if update_faiss_and_sql:
        clustered_df = pd.concat(
            [df.assign(cluster_type=cluster_name) for cluster_name, df in results.items()],
            ignore_index=True,
        )
        analyzer.save_as_faiss(faiss_runner, clustered_df)

    return results


def run_analysis_in_process(group_df):
    from src.failure_analyzer import FailureAnalyzer

    async def run():
        analyzer = FailureAnalyzer()
        return await analyzer.analyze(dataframe=group_df.reset_index(drop=True))

    return asyncio.run(run())


async def concurrent_process_by_type(df, update_faiss_and_sql=False):
    results = {}

    logger.info(f"All types in data: {df.type.unique()}")

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, run_analysis_in_process, group_df) for t, group_df in df.groupby("type")
        ]
        analysis_results = await asyncio.gather(*tasks)

    # Map results back to type
    for (t, _), result in zip(df.groupby("type"), analysis_results):
        results[t] = result

    if update_faiss_and_sql:
        from src.failure_analyzer import FailureAnalyzer

        clustered_df = pd.concat(
            [df.assign(cluster_type=cluster_name) for cluster_name, df in results.items()],
            ignore_index=True,
        )
        analyzer = FailureAnalyzer()
        analyzer.save_as_faiss(faiss_runner, clustered_df)

    return results
