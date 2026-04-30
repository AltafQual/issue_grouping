"""Domain exception hierarchy for the Issue Grouping system.

All application-level exceptions derive from ``IssueGroupingError``, enabling
callers to catch the full hierarchy with a single clause while still allowing
fine-grained handling where needed.

Hierarchy
---------
IssueGroupingError
├── EmbeddingError        – failures in vector embedding generation
├── ClusteringError       – failures during HDBSCAN / centroid clustering
├── VectorStoreError      – failures reading/writing the on-disk vector index
├── LLMError              – failures communicating with the QGenie / Vertex AI API
├── DatabaseError         – failures communicating with MySQL
├── NormalizationError    – failures during log text normalization/preprocessing
└── PipelineError         – orchestration-level failures that span multiple stages
"""

__all__ = [
    "IssueGroupingError",
    "EmbeddingError",
    "ClusteringError",
    "VectorStoreError",
    "LLMError",
    "DatabaseError",
    "NormalizationError",
    "PipelineError",
]


class IssueGroupingError(Exception):
    """Base exception for all Issue Grouping errors.

    Catch this class to handle any error raised by the issue-grouping pipeline
    without needing to enumerate every sub-type.

    Args:
        message: Human-readable description of the failure.
        cause: Optional original exception that triggered this error (for
            exception chaining when ``raise ... from ...`` is not available).
    """

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.cause:
            return f"{base} (caused by: {type(self.cause).__name__}: {self.cause})"
        return base


class EmbeddingError(IssueGroupingError):
    """Raised when embedding generation fails after all retries are exhausted.

    This covers both QGenie API failures and local BGE-M3 fallback failures.

    Args:
        message: Description of what went wrong.
        batch_size: Number of texts in the batch that failed (optional, for
            diagnostics).
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        batch_size: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.batch_size = batch_size


class ClusteringError(IssueGroupingError):
    """Raised when the clustering algorithm produces an invalid or unusable result.

    Examples: HDBSCAN receives fewer samples than ``min_cluster_size``; all
    points are labelled as noise; centroid computation encounters a zero-length
    embedding.

    Args:
        message: Description of the clustering failure.
        cluster_type: The test-type being clustered (e.g. ``"quantizer"``).
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        cluster_type: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.cluster_type = cluster_type


class VectorStoreError(IssueGroupingError):
    """Raised when reading from or writing to the on-disk vector index fails.

    The vector index consists of ``centroids.npy``, ``metadata.json``,
    ``splade_vectors.npz``, and ``splade_cluster_names.json`` per type
    directory under ``issue_grouping_db/``.

    Args:
        message: Description of the storage failure.
        index_path: Path to the index directory that caused the error.
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        index_path: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.index_path = index_path


class LLMError(IssueGroupingError):
    """Raised when the LLM (QGenie / Vertex AI Gemini) call fails after all retries.

    Args:
        message: Description of the LLM failure.
        model: Name of the model that was called (optional).
        attempts: Number of attempts made before giving up.
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        attempts: int = 0,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.model = model
        self.attempts = attempts


class DatabaseError(IssueGroupingError):
    """Raised when a MySQL database operation fails.

    This wraps both connection errors and query execution errors so the caller
    does not need to depend on ``mysql.connector`` directly.

    Args:
        message: Description of the database failure.
        query: The SQL query that failed (optional; omit sensitive data).
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.query = query


class NormalizationError(IssueGroupingError):
    """Raised when error-log normalization or preprocessing fails.

    Args:
        message: Description of the normalization failure.
        input_text: The raw text that could not be normalized (truncated if
            very long).
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        input_text: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        # Truncate to avoid bloating log output for very long error logs
        self.input_text = (input_text[:200] + "…") if input_text and len(input_text) > 200 else input_text


class PipelineError(IssueGroupingError):
    """Raised when the clustering pipeline itself fails at the orchestration level.

    This is distinct from the lower-level errors above: a ``PipelineError``
    indicates that the overall run failed (e.g. wrong DataFrame schema, missing
    required columns, unexpected state) rather than a transient sub-system
    failure.

    Args:
        message: Description of the pipeline failure.
        stage: Name of the pipeline stage that failed (e.g. ``"pregroup"``,
            ``"embed"``, ``"hdbscan"``).
        cause: Original exception.
    """

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.stage = stage
