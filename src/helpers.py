import asyncio
import hashlib
import json
import os
import queue
import random
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import wraps
from io import BytesIO
from queue import Queue

import numpy as np
import pandas as pd
import swifter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor

from src.constants import (
    ClusterSpecificKeys,
    DataFrameKeys,
    ErrorLogConfigurations,
    FaissConfigurations,
    regex_based_filteration_patterns
)
from src.custom_clustering import CustomEmbeddingCluster
from src.db_connections import ConnectToMySql
from src.execution_timer_log import execution_timer
from src.faiss_db import FaissIVFFlatIndex, SearchInExistingFaiss
from src.logger import AppLogger
from src.qgenie import classify_cluster_based_of_type, generate_cluster_name

logger = AppLogger().get_logger(__name__)
sql_connection = ConnectToMySql()
faiss_runner = FaissIVFFlatIndex()
faiss_update_queue = Queue()
scheduler = BackgroundScheduler()

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


def get_cluster_class(cluster_dict):
    if not isinstance(cluster_dict, dict):
        return ""

    for cluster_class in cluster_dict:
        if cluster_dict[cluster_class]:
            return cluster_class

    # by default mark the class for error as sdk_issue (can be changed)
    return "sdk_issue"


@execution_timer
async def assign_cluster_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a class to each cluster by classifying a single representative log per cluster.
    - Groups by cluster_name
    - Sends only one preprocessed log to the classifier
    - Converts the classifier's boolean result to a single class string via get_cluster_class
    - Assigns that class to all rows with the same cluster_name
    - Uses semaphores to limit concurrent requests to 5 at a time

    Returns the updated DataFrame.
    """
    try:
        if df is None or df.empty:
            return df

        # Ensure the target column exists
        if DataFrameKeys.cluster_class not in df.columns:
            df.loc[:, DataFrameKeys.cluster_class] = np.nan

        # Exclude non-grouped and empty/no-error clusters
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

        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(3)

        async def classify_cluster(cluster_name):
            async with semaphore:
                logger.info(f"Processing classification for cluster: {cluster_name}")
                cluster_df = df[df[DataFrameKeys.cluster_name] == cluster_name]
                if cluster_df.empty:
                    logger.warning(f"Empty cluster dataframe for {cluster_name}")
                    return None, None
                # Use a single representative preprocessed log
                representative_log = cluster_df.iloc[0][DataFrameKeys.preprocessed_text_key]
                if not isinstance(representative_log, str) or representative_log.strip() == "":
                    logger.warning(f"Invalid representative log for cluster {cluster_name}")
                    return None, None
                try:

                    logger.info(f"Sending classification request for cluster: {cluster_name}")
                    result = await classify_cluster_based_of_type([representative_log])
                    class_str = get_cluster_class(result)
                    logger.info(f"Received classification for {cluster_name}: {class_str}")

                    return cluster_df.index, class_str
                except Exception as e:
                    logger.error(f"Failed to classify cluster '{cluster_name}': {e}")
                    return None, None

        # Process clusters in batches of 5
        tasks = [classify_cluster(cluster_name) for cluster_name in unique_clusters]
        results = await asyncio.gather(*tasks)

        # Update the DataFrame with classification results
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


def load_cached_model(model_name="BAAI/bge-m3", models_dir="models"):
    try:
        # Convert model name to Hugging Face cache format
        model_folder_name = f"models--{model_name.replace('/', '--')}"

        # Construct full path to the model cache
        cwd = os.getcwd()
        model_base_path = os.path.join(cwd, models_dir, model_folder_name, "snapshots")

        if not os.path.exists(model_base_path):
            raise FileNotFoundError(f"No cached model found at {model_base_path}")

        # Get the first snapshot folder
        snapshots = os.listdir(model_base_path)
        if not snapshots:
            raise FileNotFoundError(f"No snapshot folders found in {model_base_path}")

        model_path = os.path.join(model_base_path, snapshots[0])
        print(f"Loading model from: {model_path}")

        return SentenceTransformer(model_path)
    except Exception as e:
        print(f"Exception occured while loading cached model: {e}")
        return None


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


async def update_rows_by_regex_patterns(df: pd.DataFrame) -> pd.DataFrame:
    for name, pattern in regex_based_filteration_patterns.items():
        matched_df = df[
            df[DataFrameKeys.preprocessed_text_key].astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        ]
        print(f"Found occurence of match: {name}: {matched_df.shape[0]}")
        if not matched_df.empty:
            if "verifier failed" in pattern:
                cluster_name = {"cluster_name": "VerifierFailed"}
            else:
                cluster_name = await generate_cluster_name(matched_df)
            df.loc[matched_df.index, DataFrameKeys.cluster_name] = cluster_name["cluster_name"]


@execution_timer
def merge_similar_clusters(embeddings, labels, threshold=0.93):
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
def reassign_unclustered_logs(df: pd.DataFrame, threshold=0.93):
    # Step 1: Compute centroids for all valid clusters
    valid_clusters = set(df[DataFrameKeys.cluster_name]) - {-1}
    logger.info(f"Valid clusters for reassigning unclustered logs: {valid_clusters}")
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
async def fuzzy_cluster_grouping(
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

        # Process groups in parallel using asyncio.gather
        if grouped_indices:
            # Create tasks for each group
            tasks = [generate_cluster_name(failures_dataframe.iloc[group]) for group in grouped_indices]
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            # Update the dataframe with results
            for group, result in zip(grouped_indices, results):
                failures_dataframe.loc[group, DataFrameKeys.cluster_name] = result["cluster_name"]

    # regex based common errors mapping
    await update_rows_by_regex_patterns(failures_dataframe)
    return failures_dataframe


@execution_timer
def create_excel_with_clusters(df: pd.DataFrame, cluster_column: str, columns_to_include=None) -> pd.ExcelFile:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for cluster in df[cluster_column].unique():
            if isinstance(cluster, str) and cluster.strip():
                sheet_name = str(cluster)[:31]
                cluster_df = df[df[cluster_column] == cluster]
                if columns_to_include:
                    cluster_df = cluster_df[columns_to_include]
                cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output


@execution_timer
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


@execution_timer
async def check_if_issue_alread_grouped(df: pd.DataFrame) -> pd.DataFrame:
    # Identify rows that are not yet grouped
    mask = df[DataFrameKeys.cluster_name].isin([ClusterSpecificKeys.non_grouped_key, np.nan])
    ungrouped_df = df[mask]

    if not ungrouped_df.empty:
        # Get cluster names using FAISS
        new_cluster_names, class_names = await CustomEmbeddingCluster().batch_search(
            type_=ungrouped_df.iloc[0]["type"],  # assuming same type for batch
            queries=ungrouped_df[DataFrameKeys.preprocessed_text_key].tolist(),
        )

        # Update the original DataFrame
        df.loc[mask, DataFrameKeys.cluster_name] = new_cluster_names
        df.loc[mask, DataFrameKeys.cluster_class] = class_names

        # Create a boolean mask for rows that were successfully grouped (not non_grouped_key)
        successfully_grouped_mask = mask & df[DataFrameKeys.cluster_name].ne(ClusterSpecificKeys.non_grouped_key)
        if any(successfully_grouped_mask):
            df.loc[successfully_grouped_mask, DataFrameKeys.grouped_from_faiss] = True

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
def find_regressions_between_two_tests(tc_id_a: str, tc_id_b: str) -> pd.DataFrame:
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
        logger.info("Running TC IDs updation background job")
        run_ids = sql_connection.fetch_runids()
        run_ids.to_parquet(parquet_file)
        logger.info("Background task updated Parquet file")
        # NOTE: currently disabling the nightly processing background job, will do it manually for now
        # asyncio.run(process_tc_ids_async_bg_job(run_ids))

    job_id = "update_tc_ids_job"

    # Check if scheduler is running
    if scheduler.state != STATE_RUNNING:
        scheduler.start()

    # Check if job is already scheduled
    existing_job = scheduler.get_job(job_id)
    if existing_job is None:
        scheduler.add_job(update_tc_ids, "interval", hours=5, id=job_id)
        logger.info("Scheduled TC ID update job.")
    else:
        logger.info("TC ID update job is already scheduled. Skipping re-scheduling.")


@execution_timer
async def process_tc_ids_async_bg_job(run_ids):
    from src.failure_analyzer import FailureAnalyzer

    logger.info("processing the parquet and updating faiss as background job")

    run_ids_list = run_ids["testplan_id"].tolist()
    for run_id in run_ids_list:
        try:
            processed_run_ids_path = os.path.join(FaissConfigurations.base_path, "processed_runids.json")
            processed_run_ids = []
            if os.path.isfile(processed_run_ids_path):
                processed_run_ids = json.loads(open(processed_run_ids_path).read())

            if run_id not in processed_run_ids:
                logger.info(f"Processing: {run_id}")
                await async_sequential_process_by_type(
                    FailureAnalyzer().load_data(tc_id=run_id), update_faiss_and_sql=True, run_id=run_id
                )
                await asyncio.sleep(15)
            else:
                logger.info(f"Skipping processing: {run_id} already processed")
        except Exception as e:
            logger.error(f"Error occured while processing: {run_id}: \n{e}")

            # Append full traceback to a log file
            error_log_path = os.path.join(FaissConfigurations.base_path, "failed_processing_runids_log.txt")
            with open(error_log_path, "a") as log_file:
                log_file.write(f"\nRun ID: {run_id}\n")
                log_file.write(f"\nFailed with error: {e}\n")
                log_file.write(traceback.format_exc())
                log_file.write("\n" + "-" * 80 + "\n")

            continue

    logger.info("Finished background job processing of TC IDs")


@execution_timer
async def async_process_by_type(df, update_faiss_and_sql=False, run_id=None):
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

    if results and update_faiss_and_sql:
        clustered_df = pd.concat(
            [df.assign(cluster_type=cluster_name) for cluster_name, df in results.items()],
            ignore_index=True,
        )
        faiss_update_queue.put((clustered_df, run_id, "update"))

    return results


@execution_timer
def run_analysis_in_process(group_df):
    from src.failure_analyzer import FailureAnalyzer

    async def run():
        analyzer = FailureAnalyzer()
        return await analyzer.analyze(dataframe=group_df.reset_index(drop=True))

    return asyncio.run(run())


@execution_timer
async def concurrent_process_by_type(df, update_faiss_and_sql=False, run_id=None):
    results = {}

    logger.info(f"All types in data: {df.type.unique()}")

    loop = asyncio.get_running_loop()
    grouped_df = df.groupby("type")
    with ProcessPoolExecutor(max_workers=5) as executor:
        tasks = [loop.run_in_executor(executor, run_analysis_in_process, group_df) for t, group_df in grouped_df]
        analysis_results = await asyncio.gather(*tasks)

    # Map results back to type
    for (t, _), result in zip(grouped_df, analysis_results):
        results[t] = result

    if results and update_faiss_and_sql:
        clustered_df = pd.concat(
            [df.assign(cluster_type=cluster_name) for cluster_name, df in results.items()],
            ignore_index=True,
        )
        faiss_update_queue.put((clustered_df, run_id, "update"))

    return results


@execution_timer
async def async_sequential_process_by_type(df, update_faiss_and_sql=False, run_id=None):
    from src.failure_analyzer import FailureAnalyzer

    results = {}
    analyzer = FailureAnalyzer()

    async def process_group(t, group_df):
        group_df = group_df.reset_index(drop=True)
        result = await analyzer.analyze(dataframe=group_df)
        results[t] = result

    logger.info(f"All types in data: {df.type.unique()}")
    for t, group_df in df.groupby("type"):
        logger.info(f"Processing type: {t}  with {len(group_df)} rows")
        await process_group(t, group_df)

    if results and update_faiss_and_sql:
        clustered_df = pd.concat(
            [df.assign(cluster_type=cluster_name) for cluster_name, df in results.items()],
            ignore_index=True,
        )
        faiss_update_queue.put((clustered_df, run_id, "update"))

    return results


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


def faissdb_update_worker():
    logger.info("Starting faiss db update background job")
    from threading import Lock
    save_lock = Lock()

    while True:
        try:
            task = faiss_update_queue.get(timeout=5)
        except queue.Empty:
            # No task available, continue waiting
            continue

        try:
            clustered_df, run_id, _ = task
            logger.info(f"Running faiss db updation task for run id: {run_id} with types: {clustered_df.type.unique()}")
            from src.failure_analyzer import FailureAnalyzer

            analyzer = FailureAnalyzer()

            # Use a lock to ensure only one thread is saving at a time
            with save_lock:
                analyzer.save_as_faiss(faiss_runner, clustered_df, run_id=run_id)

            logger.info(f"Successfully processed run_id: {run_id}")
        except Exception as e:
            logger.error(f"Error in FAISS update: {e}")
            logger.error(traceback.format_exc())

            error_log_path = os.path.join(FaissConfigurations.base_path, "failed_processing_runids_log.txt")
            with open(error_log_path, "a") as log_file:
                log_file.write(f"\nRun ID: {run_id} Failed While Saving to FAISS \n")
                log_file.write(f"\nFailed with error: {e}\n")
                log_file.write(traceback.format_exc())
                log_file.write("\n" + "-" * 80 + "\n")

        finally:
            faiss_update_queue.task_done()


def filter_and_sort_by_embedded_datetime(strings: list[str], n_remove: int = 0) -> list[str]:
    """
    Extract an embedded datetime from each string (supports 12-digit yymmddHHMMSS and 14-digit yyyymmddHHMMSS),
    sort by that datetime, remove the first `n_remove` items (in the chosen order), and return the rest.
    Strings without a parseable datetime are dropped.

    Parameters:
    - strings: list of input strings.
    - n_remove: number of earliest (or latest if ascending is False) items to remove after sorting.
    - ascending: sort order for the extracted datetime.

    Returns:
    - List of strings filtered and sorted by the extracted datetime.
    """

    def parse_candidate(token: str):
        # Try 14-digit YYYYmmddHHMMSS first (if present)
        if len(token) == 14:
            try:
                return datetime.strptime(token, "%Y%m%d%H%M%S")
            except ValueError:
                pass
        # Fallback to 12-digit yymmddHHMMSS (try both ends if token is longer)
        if len(token) >= 12:
            for cand in (token[:12], token[-12:]):
                try:
                    return datetime.strptime(cand, "%y%m%d%H%M%S")
                except ValueError:
                    continue
        return None

    def extract_datetime(s: str):
        # Find tokens of length 12 to 14; return the first that parses
        for token in re.findall(r"\d{12,14}", str(s)):
            dt = parse_candidate(token)
            if dt is not None:
                return dt
        return None

    dated = [(extract_datetime(s), s) for s in strings]
    dated = [t for t in dated if t[0] is not None]

    # Stable, deterministic ordering: by datetime, then by string as tiebreaker
    dated.sort(key=lambda x: (x[0], x[1]))

    # Drop the first n_remove in the chosen order
    return [s for _, s in dated[n_remove:]]


def get_bu_name(soc_name: str) -> str:
    auto = {
        "Lemans8775IVI",
        "Lemans_QOS224Q",
        "LemansIVI",
        "LemansQ",
        "Makena8295Q",
        "MakenaIVI",
        "MakenaQ",
        "Monaco7775Q",
        "Monaco8620LE",
        "Monaco_qnx710Q",
        "MonacoQ",
        "NordLE",
        "QCM8538",
        "AIC100_x86",
    }

    cbn = {"KobukLE"}

    compute = {"GlymurW", "HamoaW", "KodiakW", "MakenaW", "PoipuW", "PurwaW", "windowshost"}

    host = {"host"}

    iot = {
        "ClarenceIOT",
        "DivarIOT",
        "KamortaIOT",
        "KodiakIOT",
        "KodiakIOTU",
        "KodiakIOTU2",
        "KodiakWIOT",
        "LemansLE",
        "MilosIOT",
        "QCM2290",
        "QCM4290A1",
        "QCM4290A3",
        "QCM6125",
        "QCM6490K2L",
        "QCM6490LE",
        "QCS410",
        "QCS410LE2",
        "QCS610",
        "QCS610LE",
        "QCS610LE2",
        "QCS610LE3",
        "QCS615LE",
        "QCS7230LE",
        "QCS8250",
        "QCS8300K2L",
        "QCS8300LE",
        "QCS8550",
        "QCS8550LE",
        "QCS8550N",
        "QCS8550U",
        "QCS8625",
        "QCS8625LE",
        "QCS9100K2L",
        "QCS9100LE",
        "QCS9100LEC1",
        "QRB4210LE",
        "QRB5165LE2",
        "QRB5165U2",
    }

    mobile = {
        "Bitra",
        "Bonito",
        "Camano",
        "Clarence",
        "Divar",
        "Eliza",
        "Fillmore",
        "Fraser",
        "Kaanapali",
        "Kailua",
        "Kalpeni",
        "Kam",
        "Kamorta",
        "Kodiak",
        "Kona",
        "Lahaina",
        "LahainaPro",
        "Lamma",
        "Lanai",
        "Mannar",
        "Milos",
        "Molokai",
        "Netrani",
        "Pakala",
        "Palawan",
        "Palima",
        "Strait",
        "Tofino",
        "Waipio",
    }

    wearables = {"AspenLAW", "AspenLW"}

    xr = {"Aurora", "AuroraLE", "AuroraLE2", "Balsam", "Halliday", "Halliday2", "Luna", "Matrix", "WaipioLE"}

    if soc_name in auto:
        return "AUTO"
    elif soc_name in cbn:
        return "CBN"
    elif soc_name in compute:
        return "Compute"
    elif soc_name in host:
        return "Tools"
    elif soc_name in iot:
        return "IOT"
    elif soc_name in mobile:
        return "Mobile"
    elif soc_name in wearables:
        return "Wearables"
    elif soc_name in xr:
        return "XR"
    else:
        return "Unknown"
