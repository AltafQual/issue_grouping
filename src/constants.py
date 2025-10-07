import os
from dataclasses import dataclass

QGENEIE_API_KEY = os.getenv("QGENIE_API_KEY")
regex_based_filteration_patterns = {
    "Verifier Failed": r"verifier failed",
    "DLC Handle": r"failed to create dlc handle with code",
    "Wait timeout": r"wait timeout",
    "Batch Timer Expired": r"batchtimerexpired",
    "Scheduled on Same worker": r"scheduledonsameworker",
}


@dataclass
class DataFrameKeys:
    embeddings_key: str = "embeddings"
    preprocessed_text_key: str = "preprocessed_reason"
    cluster_type_int: str = "int_cluster"
    cluster_name: str = "clusters"
    bins: str = "bins"
    error_logs_length: str = "logs_length"
    index: str = "unique_id"
    grouped_from_faiss: str = "issue_already_occured"


@dataclass
class ClusterSpecificKeys:
    non_grouped_key: int = -1
    default_cluster_key: int = 200


@dataclass
class FaissConfigurations:
    base_path: str = "test_faiss_issue_grouping_db"
    default_db_name: str = "issue_grouping_db"


@dataclass
class ErrorLogConfigurations:
    empty_error: str = "EmptyErrorLog"
    no_error: str = "NoErrorLog"
