"""Unified error-log text normaliser.

Merges **four** previously scattered preprocessing implementations into one
class with clear, named methods:

1. ``preprocess_error_log()`` in ``src/helpers.py``
   → :meth:`ErrorNormalizer.normalize_for_embedding`

2. ``ErrorNormalizer.normalize()`` in ``src/splade_clustering.py``
   → :meth:`ErrorNormalizer.normalize_for_splade`

3. ``mask_numbers()`` in ``src/helpers.py``
   → :meth:`ErrorNormalizer.mask_numbers`

4. ``is_empty_error_log()`` in ``src/helpers.py``
   → :meth:`ErrorNormalizer.classify_empty`

Usage
-----
Create a single shared instance (e.g. in ``src/pipeline/cluster_pipeline.py``)
and pass it wherever normalisation is needed::

    normalizer = ErrorNormalizer()

    # For dense embedding input:
    cleaned = normalizer.normalize_for_embedding(raw_log)

    # For SPLADE sparse encoding:
    cleaned = normalizer.normalize_for_splade(raw_log)

Layering
--------
This module imports only from ``src.core`` and the standard library.
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd

from src.constants import ClusterSpecificKeys, DataFrameKeys, ErrorLogConfigurations
from src.core.exceptions import NormalizationError
from src.core.interfaces import INormalizer
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "ErrorNormalizer",
    "preprocess_error_log",
    "is_empty_error_log",
    "is_t2t_garbage_output",
    "t2t_garbage_output_class_assign",
    "mask_numbers",
    "trim",
    "trim_error_logs",
    "clean_excel_string",
]

# ---------------------------------------------------------------------------
# Pre-compiled patterns shared by both normalisation strategies
# ---------------------------------------------------------------------------

# Timestamp prefix on log lines:  "2024-01-15T13:45:22.123 "
_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*")

# Error marker patterns used by extract_error_lines
_ERROR_MARKER = re.compile(r"<[eE]>|\[ ?[Ee][Rr][Rr][Oo][Rr] ?\]", re.IGNORECASE)

# ---------------------------------------------------------------------------
# SPLADE-focused patterns (aggressive noise removal for sparse encoding)
# ---------------------------------------------------------------------------

_SPLADE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # CamelCase → "Camel Case"  (run before other subs so SPLADE sees words)
    (re.compile(r"([a-z])([A-Z])"), r"\1 \2"),
    # snake_case → "snake case"
    (re.compile(r"_"), " "),
    # "Limiting Reason To 3000 chars|" prefix
    (re.compile(r"Limiting Reason To \d+ chars\|", re.IGNORECASE), ""),
    # Leading exit-code prefix like "9: ", "139: "
    (re.compile(r"^\d+:\s+"), ""),
    # Absolute paths
    (re.compile(r"/prj/[^\s,;]+"), "<PATH>"),
    (re.compile(r"/tmp/AiswTest_[A-Za-z0-9_]+[^\s,;]*"), "<TMPPATH>"),
    (re.compile(r"/data/local/[^\s,;]+"), "<PATH>"),
    (re.compile(r"/[a-zA-Z][a-zA-Z0-9_\-/]+\.[a-zA-Z]{1,5}(?=\s|,|;|$)"), "<FILEPATH>"),
    # PIDs
    (re.compile(r"pid:\s*\d+", re.IGNORECASE), ""),
    # Version strings  v2.36.0.250610144245_123137-auto
    (re.compile(r"v\d+\.\d+\.\d+\.\d+[_\-][a-zA-Z0-9_\-]+"), "<VERSION>"),
    # Result/Run numbers
    (re.compile(r"\bResult_\d+\b"), "<RESULT>"),
    (re.compile(r"\bRun\d+\b"), "<RUN>"),
    # Random temp dir names  AiswTest_Abc123
    (re.compile(r"AiswTest_[A-Za-z0-9]+"), "<TMPDIR>"),
    # Timing values  "1234.56ms" or "within 3.14 secs"
    (re.compile(r"\b\d+\.\d+ms\b"), "<TIMING>"),
    (re.compile(r"within\s+[\d\.]+\s+secs?", re.IGNORECASE), "within <TIME> secs"),
    # Hex addresses
    (re.compile(r"\b0x[0-9a-fA-F]{4,}\b"), "<ADDR>"),
    # Long digit sequences (timestamps, port numbers, etc.)
    (re.compile(r"\b\d{6,}\b"), "<NUM>"),
    # Repeated whitespace
    (re.compile(r"\s+"), " "),
]

# ---------------------------------------------------------------------------
# Embedding-focused patterns (lighter cleanup; preserves more context)
# ---------------------------------------------------------------------------

_EMBED_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Timing values like "1077.9ms"
    (re.compile(r"\b\d{1,4}(\.\d+)?ms\b"), ""),
    # Line numbers "9:" or "15 -"
    (re.compile(r"^\d+[:\-]\s*", re.MULTILINE), ""),
    # Inline newline between words → space
    (re.compile(r"(?<=\S)\n(?=\S)"), " "),
    # Build/version info lines
    (re.compile(r"build version:.*?(,|$)", re.IGNORECASE), ""),
    (re.compile(r"version:.*?\\b", re.IGNORECASE), ""),
    # Brackets and quotes
    (re.compile(r"[\[\]\'\"`]"), " "),
    # Leading non-alpha characters
    (re.compile(r"^[^a-zA-Z]+"), ""),
    # Normalise whitespace
    (re.compile(r"\s+"), " "),
]

# Max raw log length before splitting head+tail for embedding normalisation
_MAX_EMBED_RAW_LEN = 20_000


class ErrorNormalizer(INormalizer):
    """Unified error-log text normaliser supporting two distinct strategies.

    Two normalisation modes are available:

    * **embedding** — lighter cleanup that preserves enough context for
      dense vector embeddings.  Lowercase output.  Caps input at 20k chars.
    * **splade** — aggressive noise removal (paths, PIDs, version strings,
      hex addresses) optimised for SPLADE sparse encoding where vocabulary
      precision matters.

    Use :meth:`normalize` (which delegates to
    :meth:`normalize_for_embedding`) to satisfy the :class:`INormalizer`
    contract.

    Example:
        >>> n = ErrorNormalizer()
        >>> n.normalize_for_embedding("[ ERROR ] pid:1234 /prj/foo/bar.so failed")
        'failed'
        >>> n.normalize_for_splade("[ ERROR ] pid:1234 /prj/foo/bar.so failed")
        '[ error ] <PATH> failed'
    """

    # ------------------------------------------------------------------
    # INormalizer contract
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """Normalise *text* using the embedding strategy.

        This method satisfies :class:`~src.core.interfaces.INormalizer` and
        is used as the default normalisation path throughout the pipeline.
        For SPLADE-specific normalisation use :meth:`normalize_for_splade`.

        Args:
            text: Raw error log string.

        Returns:
            Cleaned, lowercase string suitable for dense embedding.

        Raises:
            NormalizationError: On unexpected failures.
        """
        return self.normalize_for_embedding(text)

    # ------------------------------------------------------------------
    # Embedding normalisation
    # ------------------------------------------------------------------

    def normalize_for_embedding(self, log: str) -> str:
        """Light-touch normalisation suitable as input to dense embeddings.

        Steps (in order):
        1. Truncate very long logs to 6k head + 14k tail (preserves context).
        2. Remove short timing values (e.g. ``1077.9ms``).
        3. Remove leading line-number prefixes.
        4. Collapse inline newlines between words to a single space.
        5. Remove build/version info lines.
        6. Remove brackets and quote characters.
        7. Strip leading non-alphabetic characters.
        8. Normalise whitespace and lowercase.

        Args:
            log: Raw error log text.

        Returns:
            Cleaned, lowercase string (≤ 5 000 chars after trimming).

        Raises:
            NormalizationError: On unexpected failures.
        """
        try:
            if len(log) > _MAX_EMBED_RAW_LEN:
                log = log[:6_000] + " ... " + log[-14_000:]
            for pattern, replacement in _EMBED_PATTERNS:
                log = pattern.sub(replacement, log)
            return log.strip().lower()
        except Exception as exc:
            raise NormalizationError("normalize_for_embedding failed", input_text=log, cause=exc) from exc

    # ------------------------------------------------------------------
    # SPLADE normalisation
    # ------------------------------------------------------------------

    def normalize_for_splade(self, text: str) -> str:
        """Aggressive normalisation optimised for SPLADE sparse encoding.

        Strips all run-specific noise (absolute paths, PIDs, version strings,
        hex addresses, timestamps) and expands CamelCase/snake_case so the
        SPLADE vocabulary aligns well with the training distribution.

        Args:
            text: Raw error log text.

        Returns:
            Noise-stripped string suitable for SPLADE tokenisation.

        Raises:
            NormalizationError: On unexpected failures.
        """
        try:
            text = text.strip()
            for pattern, replacement in _SPLADE_PATTERNS:
                text = pattern.sub(replacement, text)
            return text.strip()
        except Exception as exc:
            raise NormalizationError("normalize_for_splade failed", input_text=text, cause=exc) from exc

    def tokenize_for_splade(self, text: str) -> List[str]:
        """Normalise with the SPLADE strategy then tokenise into lowercase words.

        Only returns tokens with at least 2 characters.

        Args:
            text: Raw error log text.

        Returns:
            List of lowercase word tokens.
        """
        normalized = self.normalize_for_splade(text)
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", normalized)
        return [t.lower() for t in tokens if len(t) >= 2]

    # ------------------------------------------------------------------
    # Batch convenience helpers
    # ------------------------------------------------------------------

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Apply :meth:`normalize_for_embedding` to every element of *texts*.

        Args:
            texts: List of raw error log strings.

        Returns:
            List of normalised strings in the same order.
        """
        return [self.normalize_for_embedding(t) for t in texts]

    def normalize_splade_batch(self, texts: list[str]) -> list[str]:
        """Apply :meth:`normalize_for_splade` to every element of *texts*.

        Args:
            texts: List of raw error log strings.

        Returns:
            List of SPLADE-normalised strings in the same order.
        """
        return [self.normalize_for_splade(t) for t in texts]

    # ------------------------------------------------------------------
    # Classification helpers (formerly standalone functions in helpers.py)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_empty(s: object) -> str | int:
        """Classify whether *s* represents a meaningful error log.

        Args:
            s: Value from the error-reason column (may be NaN, ``None``,
               or any string).

        Returns:
            * :attr:`~src.constants.ErrorLogConfigurations.no_error` — if
              *s* is ``None``, NaN, or a null-like string.
            * :attr:`~src.constants.ErrorLogConfigurations.empty_error` — if
              *s* contains no alphabetic characters.
            * :attr:`~src.constants.ClusterSpecificKeys.non_grouped_key`
              (``-1``) — if *s* is a valid (non-empty) error string.
        """
        if s is None or pd.isna(s) or (isinstance(s, str) and s.lower() in {"null", "nan", "none"}):
            return ErrorLogConfigurations.no_error
        if not bool(re.search(r"[a-zA-Z]", str(s))):
            return ErrorLogConfigurations.empty_error
        return ClusterSpecificKeys.non_grouped_key

    @staticmethod
    def mask_numbers(text: str) -> str:
        """Replace numeric tokens with ``<TIME>`` or ``<NUM>`` placeholders.

        Masks time-unit values (e.g. ``30ms``, ``5min``) as ``<TIME>`` and
        remaining standalone numbers as ``<NUM>``.  Used before fuzzy-matching
        so that differences in numeric values do not inflate edit distance.

        Args:
            text: Input string.

        Returns:
            String with numeric tokens replaced.
        """
        time_masked = re.sub(
            r"\b\d+(?:\.\d+)?\s*(h|hr|hrs|m|min|s|sec|ms)\b",
            "<TIME>",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"(?<!\w)(\d+(\.\d+)?)(?!\w)", "<NUM>", time_masked)

    @staticmethod
    def trim(log: object, head_ratio: float = 0.3, max_length: int = 5_000) -> str:
        """Truncate *log* to *max_length* characters, preserving head and tail.

        Args:
            log: Log value (will be converted to str).
            head_ratio: Fraction of *max_length* to take from the start.
            max_length: Maximum output length in characters.

        Returns:
            Possibly-truncated string with ``" ... "`` inserted at the cut
            point, or the original string if it is shorter than *max_length*.
        """
        try:
            log_str = str(log)
            if len(log_str) <= max_length:
                return log_str
            head_len = int(max_length * head_ratio)
            return log_str[:head_len] + " ... " + log_str[-(max_length - head_len) :]
        except (TypeError, ValueError):
            return ""


# ---------------------------------------------------------------------------
# Module-level standalone functions (ported from src/helpers.py)
# ---------------------------------------------------------------------------


@execution_timer
def preprocess_error_log(log: str) -> str:
    # Truncate early to avoid O(n) regex cost on multi-MB raw logs.
    if len(log) > 20000:
        log = log[:6000] + " ... " + log[-14000:]
    log = re.sub(r"\b\d{1,4}(\.\d+)?ms\b", "", log)
    log = re.sub(r"^\d+[:\-]\s*", "", log, flags=re.MULTILINE)
    log = re.sub(r"(?<=\S)\n(?=\S)", " ", log)
    log = re.sub(r"build version:.*?(,|$)", "", log, flags=re.IGNORECASE)
    log = re.sub(r"version:.*?\\b", "", log, flags=re.IGNORECASE)
    log = re.sub(r"[\[\]\'\"`]", " ", log)
    log = re.sub(r"^[^a-zA-Z]+", "", log)
    log = re.sub(r"\s+", " ", log).strip()
    return log.lower()


def is_empty_error_log(s):
    if s is None or pd.isna(s) or (isinstance(s, str) and s.lower() in {"null", "nan", "none"}):
        return ErrorLogConfigurations.no_error
    if not bool(re.search(r"[a-zA-Z]", s)):
        return ErrorLogConfigurations.empty_error
    return ClusterSpecificKeys.non_grouped_key


def is_t2t_garbage_output(row):
    if row.type == "transformer_run" and any(
        failure_reason in row["preprocessed_reason"]
        for failure_reason in ["context size was exceeded", "segmentation fault"]
    ):
        return DataFrameKeys.t2t_garbage_cluster
    return ClusterSpecificKeys.non_grouped_key


def t2t_garbage_output_class_assign(row):
    if row[DataFrameKeys.cluster_name] == DataFrameKeys.t2t_garbage_cluster:
        return "sdk_issue"
    return None


def mask_numbers(text: str) -> str:
    time_masked = re.sub(r"\b\d+(?:\.\d+)?\s*(h|hr|hrs|m|min|s|sec|ms)\b", "<TIME>", text, flags=re.IGNORECASE)
    final_masked = re.sub(r"(?<!\w)(\d+(\.\d+)?)(?!\w)", "<NUM>", time_masked)
    return final_masked


def trim(log, head_ratio=0.3, max_length=5000):
    try:
        log_str = str(log)
        if len(log_str) > max_length:
            head_length = int(max_length * head_ratio)
            tail_length = max_length - head_length
            head = log_str[:head_length]
            tail = log_str[-tail_length:]
            return head + " ... " + tail
        else:
            return log_str
    except Exception as e:
        logger.error(f"Error processing log: {log}, Error: {e}")
        return ""


@execution_timer
def trim_error_logs(df: pd.DataFrame, column=DataFrameKeys.preprocessed_text_key) -> pd.DataFrame:
    df[column] = df[column].apply(trim)
    return df


def clean_excel_string(text):
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    cleaned = re.sub(r"[\uFFFE\uFFFF]", "", cleaned)
    return cleaned
