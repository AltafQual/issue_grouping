import os
from dataclasses import dataclass

QGENEIE_API_KEY = os.getenv("QGENIE_API_KEY")
regex_based_filteration_patterns = {
    "Verifier Failed": r"verifier failed",
    "DLC Handle": r"failed to create dlc handle with code",
    "Wait timeout": r"wait timeout",
    "Batch Timer Expired": r"batchtimerexpired",
    "Scheduled on Same worker": r"scheduledonsameworker",
    "Device creation Failure": r"device creation failure",
}


@dataclass
class DataFrameKeys:
    embeddings_key: str = "embeddings"
    error_reason: str = "reason"
    preprocessed_text_key: str = "preprocessed_reason"
    cluster_type_int: str = "int_cluster"
    cluster_name: str = "clusters"
    bins: str = "bins"
    error_logs_length: str = "logs_length"
    index: str = "unique_id"
    grouped_from_faiss: str = "issue_already_occured"
    cluster_class: str = "cluster_class"


@dataclass
class ClusterSpecificKeys:
    non_grouped_key: int = -1
    default_cluster_key: int = 200


@dataclass
class FaissConfigurations:
    base_path: str = "issue_grouping_db"


@dataclass
class FaissDBPath:
    local: str = os.path.join(
        "/prj/qct/webtech_scratch29/altaf/issue_grouping/issue_grouping/", FaissConfigurations.base_path
    )
    prod: str = os.path.join(
        "/prj/qct/webtech_scratch29/altaf/issue_grouping/issue_grouping/issue_grouping_hosting/issue_grouping",
        FaissConfigurations.base_path,
    )


@dataclass
class ErrorLogConfigurations:
    empty_error: str = "EmptyErrorLog"
    no_error: str = "NoErrorLog"


GERRIT_INFO_PATH = "/prj/qct/webtech_hyd11/pgbs/output/tag_info.json"


@dataclass
class GERRIT_API_CONFIG:
    host: str = "https://review.qualcomm.com"
    user_name: str = os.getenv("GERRIT_USER_NAME")
    http_password: str = os.getenv("GERRIT_HTTP_PASSWORD")
