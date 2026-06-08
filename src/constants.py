import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Tuple

QGENIE_API_KEY = os.getenv("QGENIE_API_KEY")
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
    embedding_text_key: str = "embedding_reason"
    cluster_type_int: str = "int_cluster"
    cluster_name: str = "clusters"
    bins: str = "bins"
    error_logs_length: str = "logs_length"
    index: str = "unique_id"
    grouped_from_faiss: str = "issue_already_occured"
    cluster_class: str = "cluster_class"
    extracted_error_log: str = "extracted_error_log"
    t2t_garbage_cluster: str = "T2TGarbageResponse"


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
class SPLADEConfigurations:
    enabled: bool = True
    use_quantized: bool = True  # use rasyosef/splade-small (~17 MB) instead of the 440 MB model
    model_name: str = "naver/splade-cocondenser-ensembledistil"
    quantized_model_name: str = "rasyosef/splade-small"
    pregroup_threshold: float = 0.80
    low_cohesion_threshold: float = 0.35
    core_member_percentile: float = 0.50
    max_inference_chunk: int = 64  # server-side outer chunk; large requests are split internally
    splade_batch_size: int = 16  # inner SparseEncoder batch_size; bounds (batch * seq_len * vocab) tensor on GPU
    splade_pooling_chunk_size: int = 32  # SpladePooling chunk along seq-len; caps peak GPU tensor in max-pool
    splade_max_seq_length: int = 256  # truncation length applied to both encoder paths


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
    # Default points at the host NFS path on Qualcomm-internal hosts. Inside
    # the container the path doesn't exist, so override via the
    # QA2_CONFIG_FILE_PATH env var (or bind-mount /prj/qct/webtech_hyd7 into
    # the container at the same path). When unreachable,
    # iterate_db_get_testplan returns an empty result instead of raising.
    qa2_config_file_path = os.getenv("QA2_CONFIG_FILE_PATH", "/prj/qct/webtech_hyd7/qa2_web/config/config-prod.yaml")
    PROCESSING_JSON = "./consolidate_report_assests/processing_ids.json"


@dataclass
class NIGHTLY_EXECUTION:
    DAG_API_BASE = "https://aisw-hyd.qualcomm.com/dag-api/job-service/v1/jobs/public/jobs"
    DAG_API_DEFAULT_QUERY = 'status="RUNNING"'


@dataclass
class StabilityReportConfig:
    # Failure analysis
    FAILURE_THRESHOLD: float = 0.50  # flag test types with ≥ this failure rate
    DETAIL_ROW_CAP: int = 100  # max failure rows shown per type in the report
    REASON_MAX_CHARS: int = 300  # truncate reason column to this length

    SENDER: str = "mlg_user_admin@qti.qualcomm.com"
    RECIPIENT: str = "aisw.qipl.auto.qa@qti.qualcomm.com"
    SEND_EMAIL: bool = True
    TEAMS_WEBHOOK_URL: str = os.getenv("TEAMS_WEBHOOK_URL", "")

    # Idempotency tracking for the Jenkins-invoked nightly-stability-report job.
    PROCESSED_STABILITY_RUN_IDS_PATH: str = "stability_assets/processed_stability_runids.json"


@dataclass
class EmbeddingConfigurations:
    ASYNC_BATCH_TIMEOUT_S: int = 300
    ASYNC_TIMEOUT_RETRIES: int = 1
    ASYNC_TIMEOUT_BACKOFF_S: int = 5
    MAX_CONCURRENT_BATCHES: int = 2
