"""Error-log extraction utilities.

Provides :class:`LogExtractor` which encapsulates all logic for reading raw
log files from disk and extracting meaningful error lines from them.  This
consolidates the previously scattered functions:

- ``extract_error_lines()`` in ``src/helpers.py``
- ``extract_errors()`` in ``src/helpers.py``
- ``get_log_tail()`` in ``src/helpers.py``
- ``replace_limiting_reason_with_actual_reason_concurrently()`` in ``src/helpers.py``

Layering
--------
This module imports only from ``src.core``, ``src.preprocessing.normalizer``,
and the standard library.  It must **not** import from ``src.helpers``.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from src.constants import ClusterSpecificKeys, DataFrameKeys
from src.core.exceptions import NormalizationError
from src.logger import AppLogger
from src.preprocessing.normalizer import (
    _ERROR_MARKER,
    _TIMESTAMP_PREFIX,
    ErrorNormalizer,
    is_empty_error_log,
    is_t2t_garbage_output,
    mask_numbers,
    preprocess_error_log
)
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "LogExtractor",
    "extract_errors",
    "get_log_tail",
    "extract_error_lines",
    "replace_limiting_reason_with_actual_reason_concurrently",
    "remove_empty_and_misc_rows",
]

# Patterns used during extraction
_LIST_QUOTED = re.compile(r"'([^']*?)'")
_ERROR_KEYWORD = re.compile(
    r"error.*|encountered error.*|command exited with non-zero status.*|traceback.*|filenotfounderror.*",
    re.IGNORECASE,
)


class LogExtractor:
    """Extracts and cleans error content from raw test-execution logs.

    All extraction methods are pure functions wrapped as instance methods so
    they can be injected or mocked in tests.  A shared :class:`ErrorNormalizer`
    is used for post-extraction preprocessing.

    Args:
        normalizer: Optional pre-built :class:`ErrorNormalizer` instance.
            A default one is created if not supplied.
        max_workers: Thread-pool size for :meth:`extract_from_dataframe`.
            Defaults to ``min(32, cpu_count * 5)``.

    Example:
        >>> extractor = LogExtractor()
        >>> clean = extractor.extract_error_lines(raw_log_string)
    """

    def __init__(
        self,
        normalizer: Optional[ErrorNormalizer] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self._normalizer = normalizer or ErrorNormalizer()
        self._max_workers = max_workers or min(32, (os.cpu_count() or 2) * 5)

    # ------------------------------------------------------------------
    # Single-string extraction
    # ------------------------------------------------------------------

    def extract_error_lines(self, error_msg: str) -> str:
        """Extract and deduplicate lines that contain error markers.

        Recognises ``<e>``, ``<E>``, ``[ error ]``, ``[ Error ]``,
        ``[ ERROR ]`` as error markers.  If no markers are found the original
        string is returned unchanged so the caller always has a non-empty
        fallback.

        The raw log is often stored as a Python list repr (e.g.
        ``"['line1', 'line2']"``).  This method handles both that format and
        plain multi-line strings.

        Args:
            error_msg: Raw error message from ``execution_results.json`` or
                a raw log string.

        Returns:
            Newline-joined string of unique error lines, or *error_msg*
            unchanged if no markers are found.
        """
        # Try to parse as Python list repr first
        lines = _LIST_QUOTED.findall(error_msg)
        if not lines:
            lines = error_msg.splitlines()

        error_lines = [line for line in lines if _ERROR_MARKER.search(line)]
        if not error_lines:
            return error_msg  # fallback: return original

        # Deduplicate by content (strip timestamp prefix for key comparison)
        seen: set[str] = set()
        unique: list[str] = []
        for line in error_lines:
            key = _TIMESTAMP_PREFIX.sub("", line).strip()
            if key not in seen:
                seen.add(key)
                unique.append(line)

        return "\n".join(unique)

    def extract_errors_from_text(self, raw_log: str, chunk_size: int = 250, overlap: int = 50) -> str:
        """Extract error-keyword lines from *raw_log* using a sliding-window scan.

        Splits *raw_log* into overlapping chunks and searches for known error
        pattern keywords in each chunk.  Used as a fallback when
        ``execution_results.json`` is not available.

        Args:
            raw_log: Raw console log text.
            chunk_size: Characters per chunk.
            overlap: Overlap between consecutive chunks.

        Returns:
            Space-joined unique error segments found in *raw_log*.
        """
        text = raw_log.replace("\\n", "\n")
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap

        seen_segments: set[str] = set()
        ordered: list[str] = []
        for chunk in chunks:
            for match in _ERROR_KEYWORD.findall(chunk):
                if match not in seen_segments:
                    seen_segments.add(match)
                    ordered.append(match)

        return " ".join(ordered)

    def get_log_tail(self, log_path: str, lines: int = 10_000) -> str:
        """Read the last *lines* lines of ``console.txt`` from *log_path*.

        Args:
            log_path: Directory containing ``console.txt``.
            lines: Number of tail lines to read.

        Returns:
            Content string, or ``""`` on any error.
        """
        try:
            result = subprocess.run(
                f"tail -n {lines} {log_path}/console.txt",
                shell=True,
                capture_output=True,
                text=True,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception as exc:
            logger.warning(f"get_log_tail failed for {log_path}: {exc}")
            return ""

    def extract_from_path(self, log_path: str) -> tuple[str, str]:
        """Extract (preprocessed, parsed) error text from a log directory.

        Tries ``execution_results.json`` first; falls back to
        :meth:`get_log_tail` + :meth:`extract_errors_from_text`.

        Args:
            log_path: Directory path containing log files.

        Returns:
            A ``(preprocessed, parsed)`` tuple where *preprocessed* is the
            embedding-normalised text and *parsed* is the raw extracted text.
            Both are empty strings on failure.
        """
        if not log_path or (isinstance(log_path, float) and math.isnan(log_path)):
            return ("", "")
        exec_result_path = os.path.join(str(log_path), "execution_results.json")
        parsed = preprocessed = None

        if os.path.exists(exec_result_path):
            try:
                with open(exec_result_path) as fh:
                    exec_result = json.load(fh)
                raw = exec_result.get("error_msg")
                if raw:
                    parsed = self.extract_error_lines(str(raw))
                    preprocessed = self._normalizer.normalize_for_embedding(parsed)
            except Exception as exc:
                logger.warning(f"Failed reading execution_results.json at {log_path}: {exc}")

        if not parsed:
            log_data = self.get_log_tail(log_path)
            parsed = self.extract_errors_from_text(log_data)
            preprocessed = self._normalizer.normalize_for_embedding(parsed)

        return (preprocessed or "", parsed or "")

    # ------------------------------------------------------------------
    # DataFrame-level extraction
    # ------------------------------------------------------------------

    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Concurrently extract error text for every row in *df*.

        Reads log files from the ``log_path`` (or ``log``) column, extracts
        errors, and fills:

        * ``DataFrameKeys.preprocessed_text_key`` — embedding-ready text
        * ``DataFrameKeys.error_reason`` — raw extracted text

        Rows that fail are left as empty strings rather than raising.

        Args:
            df: Input DataFrame.  Must have a ``log_path`` or ``log`` column.

        Returns:
            Updated DataFrame (same index order as input).
        """
        if df is None or df.empty:
            return df

        log_col = "log_path" if "log_path" in df.columns else "log"
        if log_col not in df.columns:
            logger.warning("No log_path / log column found; skipping extraction.")
            return df

        paths = df[log_col].tolist()
        results: list[tuple[str, str]] = [("", "")] * len(paths)

        def _process(i: int, path: str) -> tuple[int, tuple[str, str]]:
            try:
                return i, self.extract_from_path(path)
            except Exception as exc:
                logger.warning(f"extract_from_path failed for row {i}: {exc}")
                return i, ("", "")

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for i, result in executor.map(lambda args: _process(*args), enumerate(paths)):
                results[i] = result

        preprocessed_vals = [r[0] for r in results]
        parsed_vals = [r[1] for r in results]

        df = df.copy()
        df[DataFrameKeys.preprocessed_text_key] = preprocessed_vals
        df[DataFrameKeys.error_reason] = parsed_vals
        return df


# ---------------------------------------------------------------------------
# Module-level standalone functions (ported from src/helpers.py)
# ---------------------------------------------------------------------------

_ERROR_PATTERN = re.compile(r"<[eE]>|\[ ?[Ee][Rr][Rr][Oo][Rr] ?\]", re.IGNORECASE)
_TIMESTAMP_PREFIX_STANDALONE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*")


def extract_errors(raw_log, chunk_size=250, overlap=50):
    text = raw_log.replace("\\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    error_patterns = [
        r"error.*",
        r"encountered error.*",
        r"command exited with non-zero status.*",
        r"traceback.*",
        r"filenotfounderror.*",
    ]
    error_segments = []
    for chunk in chunks:
        for pattern in error_patterns:
            matches = re.findall(pattern, chunk, flags=re.IGNORECASE)
            if matches:
                error_segments.extend(matches)
    errors = list(dict.fromkeys(error_segments))
    return " ".join(errors)


def get_log_tail(log_path):
    try:
        cmd = f"tail -n 10000 {log_path}/console.txt"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else ""
    except Exception as e:
        logger.debug(f"get_log_tail({log_path}): {e}")
        return f"Error reading log: {e}"


def extract_error_lines(error_msg):
    lines = re.findall(r"'([^']*?)'", error_msg)
    if not lines:
        lines = error_msg.splitlines()
    error_lines = [l for l in lines if _ERROR_PATTERN.search(l)]
    if not error_lines:
        return error_msg
    seen = set()
    unique = []
    for line in error_lines:
        key = _TIMESTAMP_PREFIX_STANDALONE.sub("", line).strip()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return "\n".join(unique)


@execution_timer
def replace_limiting_reason_with_actual_reason_concurrently(df, error_reason_column, max_workers=None):
    if df is None or df.empty:
        return df
    log_path_key = "log_path" if "log_path" in df.columns else "log"
    if log_path_key not in df.columns:
        return df
    paths = df[log_path_key].tolist()
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 2) * 5)

    def _process_one(i, path):
        try:
            if path is None or (isinstance(path, float) and math.isnan(path)) or str(path).strip() == "":
                return i, ("", "")
            exec_result_path = os.path.join(str(path), "execution_results.json")
            parsed = preprocessed = None
            if os.path.exists(exec_result_path):
                try:
                    with open(exec_result_path) as f:
                        exec_result = json.load(f)
                    raw = exec_result.get("error_msg")
                    if raw:
                        parsed = extract_error_lines(str(raw))
                        preprocessed = preprocess_error_log(parsed)
                except Exception as e:
                    logger.warning(f"Exception while fetching failure logs from execution_results.json: {e}")
            if not parsed:
                log_data = get_log_tail(path)
                parsed = extract_errors(log_data)
                preprocessed = preprocess_error_log(parsed)
            return i, (preprocessed, parsed)
        except Exception as e:
            logger.warning(f"Failed for row {i} ({path}): {e}")
            return i, ("", "")

    results = [("", "")] * len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_one, i, p) for i, p in enumerate(paths)]
        for fut in as_completed(futures):
            idx, pair = fut.result()
            results[idx] = pair

    df[error_reason_column] = [pre for pre, _ in results]
    df[DataFrameKeys.error_reason] = [raw for _, raw in results]
    return df


@execution_timer
def remove_empty_and_misc_rows(df: pd.DataFrame, errors: list, error_column_name: str):
    df[DataFrameKeys.extracted_error_log] = None
    df[error_column_name] = errors
    df_with_error_reason = df[~df[error_column_name].str.contains("limiting reason", na=False)]
    partial_error_reasons = replace_limiting_reason_with_actual_reason_concurrently(
        df[df[error_column_name].str.contains("limiting reason", na=False)].copy(),
        error_column_name,
    )
    partial_error_reasons.loc[:, DataFrameKeys.extracted_error_log] = partial_error_reasons[
        DataFrameKeys.preprocessed_text_key
    ]
    df = pd.concat([df_with_error_reason, partial_error_reasons], axis=0).reset_index(drop=True)

    # Initialize cluster_name and cluster_class columns
    df[DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key
    df[DataFrameKeys.cluster_class] = None

    # Highest priority: mark empty/no error log rows
    df[DataFrameKeys.cluster_name] = df[error_column_name].apply(is_empty_error_log)

    # For rows still unassigned (non_grouped_key), check for t2t garbage output
    ungrouped_mask = df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
    df.loc[ungrouped_mask, DataFrameKeys.cluster_name] = df.loc[ungrouped_mask].apply(is_t2t_garbage_output, axis=1)

    # Assign cluster_class for t2t rows
    t2t_mask = df[DataFrameKeys.cluster_name] == DataFrameKeys.t2t_garbage_cluster
    df.loc[t2t_mask, DataFrameKeys.cluster_class] = "sdk_issue"

    df[DataFrameKeys.preprocessed_text_key] = df[error_column_name]
    df[error_column_name] = df[error_column_name].apply(mask_numbers)
    df = df.reset_index(drop=True)
    return df
