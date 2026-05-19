import asyncio
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.data.mysql_client import sql_connection
from src.logger import AppLogger
from src.nightly_stability_job import run_stability_check
from src.pipeline.cluster_pipeline import ClusteringPipeline, ExecutionMode
from src.pipeline.workers import BackgroundWorkerManager
from src.reports.consolidated_report import run_report_generation_for_all_qairt_ids
from src.utils.run_id_utils import filter_run_ids_within_days

IST = timezone(timedelta(hours=5, minutes=30))
logger = AppLogger().get_logger(__name__)
scheduler = BlockingScheduler(timezone=IST)
RUN_ID_LOOKBACK_DAYS = 20


def consolidated_report_processing_job():
    logger.info("consolidated_report_processing_job started")
    try:
        run_report_generation_for_all_qairt_ids()
        logger.info("consolidated_report_processing_job finished")
    except Exception:
        logger.exception("consolidated_report_processing_job failed")


def run_ids_issue_grouping_processing():
    logger.info("run_ids_issue_grouping_processing started")
    try:
        run_ids_df = sql_connection.fetch_runids()
        all_ids = run_ids_df["testplan_id"].tolist() if not run_ids_df.empty else []
        recent_ids = filter_run_ids_within_days(all_ids, RUN_ID_LOOKBACK_DAYS)
        logger.info("%d/%d run_ids within last %d days", len(recent_ids), len(all_ids), RUN_ID_LOOKBACK_DAYS)

        async def _process():
            pipeline = ClusteringPipeline(update_vector_store=True)
            for run_id in recent_ids:
                if run_id and any(run_id.startswith(tag) for tag in ["QNN", "SNPE"]):
                    await pipeline.process_run_id(run_id, mode=ExecutionMode.SEQUENTIAL)

        asyncio.run(_process())
        logger.info("run_ids_issue_grouping_processing finished")
    except Exception:
        logger.exception("run_ids_issue_grouping_processing failed")


def nightly_stability_monitor_job():
    logger.info("nightly_stability_monitor_job started")
    try:
        result = asyncio.run(run_stability_check())
        logger.info(
            "nightly_stability_monitor_job finished | jobs=%s | teams=%s | email=%s",
            result.get("total_running_auto_jobs", 0),
            result.get("teams_notified"),
            result.get("email_sent_to"),
        )
    except Exception:
        logger.exception("nightly_stability_monitor_job failed")


if __name__ == "__main__":
    worker_manager = BackgroundWorkerManager()
    worker_manager.start()
    try:
        scheduler.add_job(
            consolidated_report_processing_job,
            trigger=IntervalTrigger(hours=7),
            id="qairt_reports",
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(IST),
        )
        scheduler.add_job(
            run_ids_issue_grouping_processing,
            "interval",
            hours=12,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
            next_run_time=datetime.now(IST),
        )
        scheduler.add_job(
            nightly_stability_monitor_job,
            "interval",
            hours=4,
            id="nightly_stability_monitor",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
            next_run_time=datetime.now(IST),
        )
        scheduler.start()
    finally:
        worker_manager.stop(timeout=30.0)
