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
import collections
import json
import os
import pickle
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.constants import NIGHTLY_EXECUTION, StabilityReportConfig
from src.logger import AppLogger
from src.monitoring.email_notifier import send_email_report
from src.monitoring.hourly_report import RunAnalysis, analyze_type_failures, build_combined_stability_html
from src.monitoring.teams_notifier import send_teams_breakdown_card, send_teams_summary_card

logger = AppLogger().get_logger(__name__)
_STABILITY_REPORT_SCRIPT = (
    "/prj/qct/webtech_scratch29/altaf/ci2.0/QNN-AUTO/reporting_scripts/"
    "Nightly_Reports/auto_summary_stability_reports/stability_report.py"
)
_STABILITY_REPORT_TEMPLATE_DIR = (
    "/prj/qct/webtech_scratch29/altaf/ci2.0/QNN-AUTO/reporting_scripts/"
    "Nightly_Reports/auto_summary_stability_reports/"
)
_STABILITY_OUTPUT_BASE = "/prj/qct/webtech_hyd19/AUTO_SUMMARY_STABILITY"
_SUBPROCESS_TIMEOUT_SEC = 30 * 60

async def _run_dag_query(query) -> dict:
    token = os.environ.get("DAG_API_BEARER_TOKEN", "")
    if not token:
        logger.error("DAG_API_BEARER_TOKEN not set — skipping stability check")
        return {"status": 401, "error": "DAG_API_BEARER_TOKEN environment variable not set"}
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
    return dag_data


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

    def _load_pkl(path: str) -> pd.DataFrame:
        with open(path + ".pkl", "rb") as fh:
            raw = pickle.load(fh)
        return pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw

    dag_data = await _run_dag_query(query)
    auto_jobs = [
        job
        for job in (dag_data.get("data") or [])
        if job.get("run_id", "").startswith("QNN") and "auto" in job.get("run_id", "")
    ]
    logger.info(f"Found {len(auto_jobs)} QNN auto run(s) in RUNNING state")

    processed_runs: list[RunAnalysis] = []
    results_summary: list[dict[str, Any]] = []
    loop = asyncio.get_event_loop()

    for job_info in auto_jobs:
        run_id = job_info.get("run_id", "unknown")
        pkl_path = job_info.get("excel_report_path", "")

        df: pd.DataFrame | None = None
        if run_id and pkl_path:
            try:
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

    cfg = StabilityReportConfig
    if processed_runs:
        if cfg.TEAMS_WEBHOOK_URL:
            await send_teams_summary_card(cfg.TEAMS_WEBHOOK_URL, processed_runs)
            await send_teams_breakdown_card(cfg.TEAMS_WEBHOOK_URL, processed_runs)
        elif not cfg.SEND_EMAIL:
            logger.warning("TEAMS_WEBHOOK_URL not set and SEND_EMAIL=false — no notification sent")

        if cfg.SEND_EMAIL:
            html_report = build_combined_stability_html(processed_runs)
            send_email_report("Hourly Report — AUTO", cfg.SENDER, cfg.RECIPIENT, html_report)
            logger.info(f"Stability email sent to {cfg.RECIPIENT} for {len(processed_runs)} run(s)")

    return {
        "status": 200,
        "total_running_auto_jobs": len(auto_jobs),
        "email_sent_to": cfg.RECIPIENT if processed_runs and cfg.SEND_EMAIL else None,
        "teams_notified": bool(processed_runs and cfg.TEAMS_WEBHOOK_URL),
        "processed": results_summary,
    }

def _load_processed_stability_run_ids() -> list[str]:
    path = Path(StabilityReportConfig.PROCESSED_STABILITY_RUN_IDS_PATH)
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning(f"Failed to load {path}; treating as empty")
        return []


def _mark_stability_run_processed(run_id: str) -> None:
    path = Path(StabilityReportConfig.PROCESSED_STABILITY_RUN_IDS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    processed = _load_processed_stability_run_ids()
    if run_id not in processed:
        processed.append(run_id)
        path.write_text(json.dumps(processed, indent=3))


def generate_stability_nightly_report() -> dict[str, Any]:
    """Run stability_report.py for each not-yet-processed completed QNN auto-stability run.

    Idempotent across invocations: a run_id is persisted only after a successful
    subprocess exit, so failures (timeout / non-zero exit / exception) are retried
    on the next call. Designed as a sync entry point for a Jenkins job.
    """
    def _drain(stream, buf, tag):
                for line in stream:
                    line = line.rstrip()
                    buf.append(line)
                    logger.info(f"[stability:{tag}] {line}")
                    
    dag_data = asyncio.run(_run_dag_query('status="COMPLETED"'))
    if dag_data.get("status") not in (None, 200):
        logger.error(f"DAG query failed: {dag_data}")
        return {
            "status": dag_data.get("status", 500),
            "error": dag_data.get("error"),
            "candidates": 0,
            "processed_now": [],
            "failed": [],
        }

    candidates = [
        job
        for job in (dag_data.get("data") or [])
        if (rid := job.get("run_id", "")).startswith("QNN") and "auto_stability" in rid
    ]

    already_processed = set(_load_processed_stability_run_ids())
    new_jobs = [j for j in candidates if j["run_id"] not in already_processed]
    logger.info(
        f"DAG returned {len(candidates)} QNN auto-stability run(s); "
        f"{len(new_jobs)} new, {len(candidates) - len(new_jobs)} already processed."
    )

    processed_now: list[str] = []
    failed: list[dict[str, str]] = []

    for job in new_jobs:
        run_id = job["run_id"]
        output_dir = Path(f"{_STABILITY_OUTPUT_BASE}/{run_id}_qgenie")

        # stability_report.py does `shutil.copy(template, output_dir)` then
        # `os.chdir(output_dir)` — if the dir is missing, shutil.copy creates
        # a regular file at the path, then chdir fails with NotADirectoryError
        # and leaves a stale file behind. Heal both cases here.
        if output_dir.exists() and not output_dir.is_dir():
            try:
                output_dir.unlink()
                logger.warning(f"[stability] {run_id}: removed stale file at {output_dir}")
            except Exception as e:
                logger.error(f"[stability] {run_id}: cannot remove stale file {output_dir}: {e}")
                failed.append({"run_id": run_id, "reason": f"stale_file_unremovable: {e}"})
                continue

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"[stability] {run_id}: cannot create output dir {output_dir}: {e}")
            failed.append({"run_id": run_id, "reason": f"mkdir_failed: {e}"})
            continue

        cmd = [
            "python3",
            _STABILITY_REPORT_SCRIPT,
            "-r",
            run_id,
            "-t",
            _STABILITY_REPORT_TEMPLATE_DIR,
            "-s",
            "mlg_user_admin@qti.qualcomm.com",
            "-re",
            "aisw.qipl.auto.qa@qti.qualcomm.com",
            "-o",
            str(output_dir),
            "-v",
            "socwise",
            "--qgenie",
        ]
        logger.info(f"[stability] Running for {run_id}: {shlex.join(cmd)}")
        tail: collections.deque[str] = collections.deque(maxlen=50)
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            reader = threading.Thread(target=_drain, args=(proc.stdout, tail, run_id), daemon=True)
            reader.start()
            try:
                returncode = proc.wait(timeout=_SUBPROCESS_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                reader.join(timeout=5)
                logger.error(
                    f"[stability] {run_id}: timed out after {_SUBPROCESS_TIMEOUT_SEC}s — not marking processed"
                )
                failed.append({"run_id": run_id, "reason": "timeout"})
                continue
            reader.join(timeout=5)
        except Exception as e:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
            logger.exception(f"[stability] {run_id}: subprocess raised {e!r}")
            failed.append({"run_id": run_id, "reason": f"exception: {e}"})
            continue

        if returncode != 0:
            logger.error(f"[stability] {run_id}: exit={returncode}; tail:\n" + "\n".join(tail))
            failed.append({"run_id": run_id, "reason": f"exit_{returncode}"})
            continue

        _mark_stability_run_processed(run_id)
        processed_now.append(run_id)
        logger.info(f"[stability] {run_id}: OK — marked processed")

    return {
        "status": 200,
        "candidates": len(candidates),
        "already_processed": len(candidates) - len(new_jobs),
        "processed_now": processed_now,
        "failed": failed,
    }
