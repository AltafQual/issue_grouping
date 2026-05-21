"""DAG API loader: first-layer data fetcher for run-id-based test data.

Queries the DAG job-service API to look up the canonical
``excel_report_path`` for a run_id, then loads the corresponding
``.xlsx`` file from disk.  On any failure (missing token, network error,
non-2xx response, no matching job, missing path field, file not found,
parse error, empty DataFrame) the loader returns ``None`` and the caller
is expected to fall back to MySQL.

Layering
--------
This module sits in the **data** layer.  It imports only from:
- ``src.constants`` (NIGHTLY_EXECUTION)
- ``src.logger``
- Standard library / third-party packages (``httpx``, ``pandas``)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx
import pandas as pd

from src.constants import NIGHTLY_EXECUTION
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["load_run_id_via_dag_api"]

_DAG_API_TIMEOUT_S = 30.0
_DAG_API_MAX_RETRIES = 3


def _fetch_excel_report_path(run_id: str) -> Optional[str]:
    """Query the DAG API for *run_id* and return its ``excel_report_path``.

    Args:
        run_id: Testplan/run identifier (e.g. ``"QNN-v2.48.0.260514040326-non_pt_nightly"``).

    Returns:
        The bare ``excel_report_path`` string from the matching job, or
        ``None`` if the token is missing, the request fails, or no path
        is present in the response.
    """
    token = os.environ.get("DAG_API_BEARER_TOKEN", "")
    if not token:
        logger.warning("DAG_API_BEARER_TOKEN not set — DAG API lookup skipped for run_id=%s", run_id)
        return None

    query = f'run_id="{run_id}"'
    last_exc: Exception | None = None
    for attempt in range(_DAG_API_MAX_RETRIES):
        try:
            with httpx.Client(timeout=_DAG_API_TIMEOUT_S, verify=False) as client:
                resp = client.get(
                    NIGHTLY_EXECUTION.DAG_API_BASE,
                    params={"query": query},
                    headers={
                        "accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                )
            resp.raise_for_status()
            body = resp.json()

            jobs = body.get("data") or []
            if not jobs:
                logger.info("DAG API returned no jobs for run_id=%s", run_id)
                return None

            for job in jobs:
                if job.get("run_id") == run_id:
                    path = (job.get("excel_report_path") or "").strip()
                    if path:
                        return path

            path = (jobs[0].get("excel_report_path") or "").strip()
            return path or None

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                logger.warning(
                    "DAG API returned HTTP %s for run_id=%s: %s",
                    exc.response.status_code,
                    run_id,
                    exc.response.text,
                )
                return None  # 4xx — client error, no retry
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        if attempt < _DAG_API_MAX_RETRIES - 1:
            time.sleep(2.0**attempt)

    logger.warning(
        "DAG API request failed after %d attempts for run_id=%s: %s",
        _DAG_API_MAX_RETRIES,
        run_id,
        last_exc,
    )
    return None


def load_run_id_via_dag_api(run_id: str) -> Optional[pd.DataFrame]:
    """Try to load test results for *run_id* via the DAG API + Excel file.

    The flow is: DAG API → ``excel_report_path`` → append ``.xlsx`` →
    ``pandas.read_excel``.  Any failure along the way is logged as a
    warning and the function returns ``None`` so the caller can fall
    back to a different data source (typically MySQL).

    Args:
        run_id: Testplan/run identifier to look up.

    Returns:
        DataFrame on success, ``None`` on any failure or empty data.
    """
    if not run_id:
        return None

    base_path = _fetch_excel_report_path(run_id)
    if not base_path:
        logger.info("No excel_report_path returned by DAG API for run_id=%s", run_id)
        return None

    xlsx_path = base_path + ".xlsx"
    try:
        df = pd.read_excel(xlsx_path)
    except FileNotFoundError:
        logger.warning("DAG API xlsx not found on disk: %s (run_id=%s)", xlsx_path, run_id)
        return None
    except Exception as exc:
        logger.warning("Failed to read xlsx %s for run_id=%s: %s", xlsx_path, run_id, exc)
        return None

    if df is None or df.empty:
        logger.warning("DAG API loaded an empty DataFrame for run_id=%s from %s", run_id, xlsx_path)
        return None

    logger.info("Loaded run_id=%s via DAG API (%d rows) from %s", run_id, len(df), xlsx_path)
    return df
