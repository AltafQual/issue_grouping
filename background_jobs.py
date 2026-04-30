import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.data.mysql_client import sql_connection
from src.nightly_stability_job import run_stability_check
from src.pipeline.cluster_pipeline import ClusteringPipeline, ExecutionMode
from src.pipeline.workers import BackgroundWorkerManager
from src.reports.consolidated_report import run_report_generation_for_all_qairt_ids

IST = timezone(timedelta(hours=5, minutes=30))
logger = logging.getLogger(__name__)
scheduler = BlockingScheduler(timezone=IST)


def consolidated_report_processing_job():
    now = datetime.now(IST)
    print(f"[START]  {now.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30)")
    try:
        run_report_generation_for_all_qairt_ids()
        now2 = datetime.now(IST)
        print(f"[SUCCESS] {now2.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30)")
    except Exception as e:
        now3 = datetime.now(IST)
        print(f"[ERROR]   {now3.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30) -> {e}")


def run_ids_issue_grouping_processing():
    try:
        now = datetime.now(IST)
        print(f"[START]  {now.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30)")
        run_ids = sql_connection.fetch_runids()

        async def _process():
            pipeline = ClusteringPipeline(update_vector_store=True)
            for run_id in run_ids["testplan_id"].tolist():
                if run_id and any(run_id.startswith(tag) for tag in ["QNN", "SNPE"]):
                    await pipeline.process_run_id(run_id, mode=ExecutionMode.SEQUENTIAL)

        asyncio.run(_process())
        now2 = datetime.now(IST)
        print(f"[SUCCESS] {now2.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30)")
    except Exception as e:
        now3 = datetime.now(IST)
        print(f"[ERROR]   {now3.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30) -> {e}")


def nightly_stability_monitor_job():
    now = datetime.now(IST)
    print(f"[STABILITY START]  {now.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30)")
    try:
        result = asyncio.run(run_stability_check())
        now2 = datetime.now(IST)
        print(
            f"[STABILITY SUCCESS] {now2.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30) "
            f"| jobs={result.get('total_running_auto_jobs', 0)} "
            f"| teams={result.get('teams_notified')} "
            f"| email={result.get('email_sent_to')}"
        )
    except Exception as e:
        now3 = datetime.now(IST)
        print(f"[STABILITY ERROR]   {now3.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30) -> {e}")


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
