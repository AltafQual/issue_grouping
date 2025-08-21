import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import DataFrameKeys
from src.qgenie import generate_cluster_name


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


def remove_empty_and_misc_rows(df: pd.DataFrame, errors: list, error_column_name: str):
    def has_alphabets(s):
        if not bool(re.search(r"[a-zA-Z]", s)):
            return "NoErrorLog"
        return -1

    def mask_numbers(text):
        # Replace floating point numbers and integers with a placeholder
        return re.sub(r"\d+(\.\d+)?", "<NUM>", text)

    df[error_column_name] = errors
    # Apply filters
    df = df[
        ~df[error_column_name]
        .str.strip()
        .str.lower()
        .str.startswith("limiting reason to 3000 chars")  # not starting with that phrase
    ]
    # add
    df.loc[:, DataFrameKeys.cluster_name] = df[error_column_name].apply(has_alphabets)
    df.loc[:, error_column_name] = df[error_column_name].apply(mask_numbers)
    df = df.reset_index(drop=True)
    return df


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


def update_labels_with_merged_clusters(df, merged_clusters, label_col):
    # Create a mapping from old label to new (parent) label
    label_map = {}
    for parent, children in merged_clusters.items():
        for child in children:
            label_map[int(child)] = int(parent)

    # Apply the mapping to the DataFrame
    df[label_col] = df[label_col].map(label_map).fillna(df[label_col])
    return df


def trim_error_logs(df: pd.DataFrame, column=DataFrameKeys.preprocessed_text_key, max_length=1000):
    def trim(log):
        try:
            log_str = str(log)
            return log_str[-max_length:] if len(log_str) > max_length else log_str
        except Exception as e:
            print(f"Error processing log: {log}, Error: {e}")
            return ""

    df[column] = df[column].apply(trim)
    return df


def group_similar_errors(df: pd.DataFrame, column: str, threshold):
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


def fuzzy_cluster_grouping(failures_dataframe, threshold=100, bin_intervals=[[0, 50], [50, 100]]):
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

        # Assign cluster names
        for group in grouped_indices:
            cluster_details_dict = generate_cluster_name(failures_dataframe.iloc[group])
            failures_dataframe.loc[group, DataFrameKeys.cluster_name] = cluster_details_dict["cluster_name"]

    return failures_dataframe


def create_excel_with_clusters(df, cluster_column, columns_to_include):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for cluster in df[cluster_column].unique():
            sheet_name = str(cluster)[:31]
            cluster_df = df[df[cluster_column] == cluster][columns_to_include]
            cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output
