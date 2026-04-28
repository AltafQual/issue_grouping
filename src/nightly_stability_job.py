"""Core logic for the nightly stability monitor.

Extracted here so it can be called from both the FastAPI endpoint
(GET /api/running_jobs/) and the APScheduler background job without
importing the full api.py module.

Public API
----------
    run_stability_check(query) -> dict   — fetch + analyse + notify
"""

from __future__ import annotations

import asyncio
import os
import pickle
from typing import Any

import httpx
import pandas as pd

from src.constants import NIGHTLY_EXECUTION, StabilityReportConfig
from src.email_helpers import send_email_report
from src.logger import AppLogger
from src.stability_report import RunAnalysis, analyze_type_failures, build_combined_stability_html
from src.teams_helpers import send_teams_breakdown_card, send_teams_summary_card

logger = AppLogger().get_logger(__name__)


async def run_stability_check(
    query: str = NIGHTLY_EXECUTION.DAG_API_DEFAULT_QUERY,
) -> dict[str, Any]:
    """Fetch running QNN auto jobs, analyse failure rates, and send notifications.

    Args:
        query: DAG API filter expression (default: ``status="RUNNING"``).

    Returns:
        Summary dict with keys: status, total_running_auto_jobs,
        email_sent_to, teams_notified, processed.
    """
    token = os.environ.get("DAG_API_BEARER_TOKEN", "")
    if not token:
        logger.error("DAG_API_BEARER_TOKEN not set — skipping stability check")
        return {"status": 401, "error": "DAG_API_BEARER_TOKEN environment variable not set"}

    # ── 1. Fetch running jobs from DAG API ────────────────────────────────────
    dag_data: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            resp = await client.get(
                NIGHTLY_EXECUTION.DAG_API_BASE,
                params={"query": query},
                headers={"accept": "application/json", "Authorization": f"Bearer {token}"},
            )
        resp.raise_for_status()
        dag_data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"DAG API returned HTTP {e.response.status_code}: {e.response.text}")
        return {"status": e.response.status_code, "error": e.response.text}
    except Exception as e:
        logger.exception(f"Error calling DAG API: {e}")
        return {"status": 500, "error": str(e)}

    # ── 2. Filter to QNN auto runs ────────────────────────────────────────────
    auto_jobs = [
        job
        for job in (dag_data.get("data") or [])
        if job.get("run_id", "").startswith("QNN") and "auto" in job.get("run_id", "")
    ]
    logger.info(f"Found {len(auto_jobs)} QNN auto run(s) in RUNNING state")

    # ── 3. Load each pkl and analyse ─────────────────────────────────────────
    processed_runs: list[RunAnalysis] = []
    results_summary: list[dict[str, Any]] = []
    loop = asyncio.get_event_loop()

    for job_info in auto_jobs:
        run_id = job_info.get("run_id", "unknown")
        pkl_path = job_info.get("excel_report_path", "")

        df: pd.DataFrame | None = None
        if run_id and pkl_path:
            try:

                def _load_pkl(path: str) -> pd.DataFrame:
                    with open(path + ".pkl", "rb") as fh:
                        raw = pickle.load(fh)
                    return pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw

                df = await loop.run_in_executor(None, _load_pkl, pkl_path)
            except Exception as e:
                logger.warning(f"{run_id}: Could not load pkl from {pkl_path}: {e}")

        if df is None or df.empty:
            results_summary.append({"run_id": run_id, "status": "no_data"})
            continue

        if "type" not in df.columns or "result" not in df.columns:
            logger.warning(f"{run_id}: DataFrame missing 'type' or 'result' columns — skipping")
            results_summary.append({"run_id": run_id, "status": "missing_columns"})
            continue

        type_stats = analyze_type_failures(df)
        highlighted = [t for t, s in type_stats.items() if s.highlighted]
        logger.info(
            f"{run_id}: {len(type_stats)} types analysed, "
            f"{len(highlighted)} flagged (>={int(StabilityReportConfig.FAILURE_THRESHOLD * 100)}%)"
        )
        processed_runs.append(RunAnalysis(run_id=run_id, job_info=job_info, type_stats=type_stats))
        results_summary.append(
            {
                "run_id": run_id,
                "status": "ok",
                "types_analysed": len(type_stats),
                "highlighted_types": highlighted,
            }
        )

    # ── 4. Send notifications ─────────────────────────────────────────────────
    cfg = StabilityReportConfig
    if processed_runs:
        if cfg.TEAMS_WEBHOOK_URL:
            await send_teams_summary_card(cfg.TEAMS_WEBHOOK_URL, processed_runs)
            await send_teams_breakdown_card(cfg.TEAMS_WEBHOOK_URL, processed_runs)
        elif not cfg.SEND_EMAIL:
            logger.warning("TEAMS_WEBHOOK_URL not set and SEND_EMAIL=false — no notification sent")

        if cfg.SEND_EMAIL:
            html_report = build_combined_stability_html(processed_runs)
            send_email_report("Nightly Stability Analysis", cfg.SENDER, cfg.RECIPIENT, html_report)
            logger.info(f"Stability email sent to {cfg.RECIPIENT} for {len(processed_runs)} run(s)")

    return {
        "status": 200,
        "total_running_auto_jobs": len(auto_jobs),
        "email_sent_to": cfg.RECIPIENT if processed_runs and cfg.SEND_EMAIL else None,
        "teams_notified": bool(processed_runs and cfg.TEAMS_WEBHOOK_URL),
        "processed": results_summary,
    }
