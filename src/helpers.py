import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


def remove_empty_and_misc_rows(df: pd.DataFrame, errors: list, error_column_name="preprocessed_reason"):
    def has_alphabets(s):
        return bool(re.search(r"[a-zA-Z]", s))

    df[error_column_name] = errors
    # Apply filters
    df = df[
        ~df[error_column_name]
        .str.strip()
        .str.lower()
        .str.startswith("limiting reason to 3000 chars")  # not starting with that phrase
        & df[error_column_name].apply(lambda x: has_alphabets(str(x).strip()))  # has alphabets
    ]
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
