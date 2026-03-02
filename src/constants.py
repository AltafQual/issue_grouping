import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Tuple

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
    extracted_error_log: str = "extracted_error_log"


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


@dataclass
class GERRIT_CONFIGURATION:
    gerrit_info_path: str = "/prj/qct/webtech_hyd11/pgbs/output/tag_info.json"
    gerrit_backend_configuration: ClassVar[Dict[Tuple[str, ...], str]] = {
        "cpu": ("qnn-cpu",),
        "htp_fp16": ("mlg-infra", "qnn-htp", "manifest", "genie"),
        "htp": ("mlg-infra", "qnn-htp", "manifest", "genie"),
        "gpu": ("qnn-gpu",),
        "gpu_fp16": ("qnn-gpu",),
        "lpai": ("qnn-eai",),
        "common": ("qnn", "qnn_qti", "mlg", "ml", "api"),
        "converter": ("modeltools",),
        "quantizer": ("modeltools", "qnn-cpu"),
        "mcp": ("mlg-infra", "qnn-htp", "manifest", "genie"),
        "mcp_x86": ("mlg-infra", "qnn-htp", "manifest", "genie"),
        "hta": ("hta",),
        "gpu:cpu": ("qnn-cpu", "qnn-gpu"),
    }


@dataclass
class GERRIT_API_CONFIG:
    host: str = "https://review.qualcomm.com"
    user_name: str = os.getenv("GERRIT_USER_NAME")
    http_password: str = os.getenv("GERRIT_HTTP_PASSWORD")


@dataclass
class CONSOLIDATED_REPORTS:
    path: str = "/prj/qct/webtech_hyd19/CONSOLIDATED_REPORTS/"
    sheet_name: str = "Consolidated Report"

    # NOTE: hard coding previous version release info, currently have to update everymonth unless automated
    prev_release_info: str = "v2.42.0.251225135753_193295"
    prev_release_rc_number: str = "RC4"

    prev_run_id_generation_script_path = "/prj/mlgqipl/Satyam/Scripts/get_previous_testplan_id.py"
    qa2_config_file_path = "/prj/qct/webtech_hyd7/qa2_web/config/config-prod.yaml"
    PROCESSING_JSON = "./consolidate_report_assests/processing_ids.json"
