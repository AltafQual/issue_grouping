import os
from dataclasses import dataclass

QGENEIE_API_KEY = os.getenv("QGENIE_API_KEY")


@dataclass
class DataFrameKeys:
    embeddings_key: str = "embeddings"
    preprocessed_text_key: str = "preprocessed_reason"
    cluster_type_int: str = "int_cluster"
    cluster_name: str = "clusters"
    bins: str = "bins"
    error_logs_length: str = "logs_length"


@dataclass
class ClusterSpecificKeys:
    non_grouped_key: int = -1
    default_cluster_key: int = 200


@dataclass
class FaissConfigurations:
    base_path: str = "faiss_issue_grouping_db"
    default_db_name: str = "issue_grouping_db"
