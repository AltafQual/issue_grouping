import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.consolidated_reports_analysis import run_report_generation_for_all_qairt_ids
from src.helpers import process_tc_ids_async_bg_job, sql_connection

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
        asyncio.run(process_tc_ids_async_bg_job(run_ids))
        now2 = datetime.now(IST)
        print(f"[SUCCESS] {now2.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30)")
    except Exception as e:
        now3 = datetime.now(IST)
        print(f"[ERROR]   {now3.strftime('%Y-%m-%d %H:%M:%S')} IST (UTC+05:30) -> {e}")


if __name__ == "__main__":
    scheduler.add_job(
        consolidated_report_processing_job,
        trigger=IntervalTrigger(hours=10),
        id="qairt_reports",
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(IST),
    )
    scheduler.add_job(
        run_ids_issue_grouping_processing, "interval", hours=12, max_instances=1, coalesce=True, misfire_grace_time=3600
    )
    scheduler.start()
