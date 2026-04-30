"""Background worker manager for async queue processing.

Provides :class:`BackgroundWorkerManager` — a single class that owns the
lifecycle of all daemon worker threads used for asynchronous off-critical-path
updates (FAISS centroid persistence, etc.).

Design goals
------------
* **No import-time side effects** — threads are only started when
  :meth:`BackgroundWorkerManager.start` is called explicitly.
* **Graceful shutdown** — :meth:`BackgroundWorkerManager.stop` signals each
  worker to exit and waits up to *timeout* seconds for them to finish.
* **Single responsibility** — each worker thread processes exactly one queue
  type.

Layering
--------
This module imports from ``src.constants``, ``src.logger``, and standard
library packages.  It uses deferred imports inside the worker functions to
break potential circular-import chains with ``src.failure_analyzer``.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import queue
import re
import shutil
import sys
import traceback
from queue import Queue
from threading import Lock, Thread
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING

from src.constants import FaissConfigurations, FaissDBPath
from src.custom_clustering import CustomEmbeddingCluster
from src.data.mysql_client import sql_connection
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "BackgroundWorkerManager",
    "swap_issue_grouping_db_to_prod",
    "requeue_failed_run_ids",
    "tc_id_scheduler",
    "process_tc_ids_async_bg_job",
]

# Global update queues — imported by helpers.py (for backward compat) and by
# the new ClusteringPipeline.  They are created here so there is a single
# source of truth.
faiss_update_queue: Queue = Queue()

# Module-level scheduler and parquet file path used by tc_id_scheduler /
# process_tc_ids_async_bg_job.
scheduler = BackgroundScheduler()
parquet_file = "run_ids.parquet"


def _faissdb_update_worker() -> None:
    """Worker function for the FAISS centroid update queue.

    Runs in a daemon thread.  Dequeues ``(clustered_df, run_id, op)`` tuples
    from :data:`faiss_update_queue` and persists them to the on-disk vector
    store via :class:`~src.custom_clustering.CustomEmbeddingCluster`.

    This function is **blocking** and designed to run in a background thread —
    never call it directly from the main event loop.
    """
    logger.info("Starting FAISS db update background worker")
    save_lock = Lock()

    while True:
        try:
            task = faiss_update_queue.get(timeout=5)
        except queue.Empty:
            continue

        clustered_df = run_id = None
        try:
            clustered_df, run_id, _ = task
            logger.info(f"Running FAISS DB update for run_id={run_id} " f"with types: {clustered_df.type.unique()}")
            with save_lock:
                CustomEmbeddingCluster().save_threaded(clustered_df, run_id=run_id)

            logger.info(f"Successfully persisted run_id={run_id} to FAISS DB")
        except Exception as e:
            logger.error(f"Error in FAISS update worker: {e}")
            logger.error(traceback.format_exc())

            error_log_path = os.path.join(FaissConfigurations.base_path, "failed_processing_runids_log.txt")
            try:
                with open(error_log_path, "a") as log_file:
                    log_file.write(f"\nRun ID: {run_id} — failed during FAISS save\n")
                    log_file.write(f"Error: {e}\n")
                    log_file.write(traceback.format_exc())
                    log_file.write("\n" + "-" * 80 + "\n")
            except OSError:
                pass
        finally:
            faiss_update_queue.task_done()


class BackgroundWorkerManager:
    """Lifecycle manager for daemon worker threads.

    Starts and stops the daemon threads that process asynchronous update
    queues.  Must be initialised and started explicitly — no threads are
    created at module import time.

    Example (in ``api.py`` lifespan)::

        worker_manager = BackgroundWorkerManager()
        worker_manager.start()
        # … application runs …
        worker_manager.stop(timeout=10.0)

    Attributes:
        _threads: List of running daemon :class:`~threading.Thread` instances.
        _started: Whether :meth:`start` has been called.
    """

    def __init__(self) -> None:
        self._threads: list[Thread] = []
        self._started: bool = False

    def start(self) -> None:
        """Start all background worker daemon threads.

        This method is idempotent — calling it more than once has no effect
        after the first call.
        """
        if self._started:
            logger.warning("BackgroundWorkerManager.start() called more than once — ignoring")
            return

        faiss_thread = Thread(target=_faissdb_update_worker, daemon=True, name="faissdb-update-worker")
        faiss_thread.start()
        self._threads.append(faiss_thread)

        self._started = True
        logger.info(f"BackgroundWorkerManager started {len(self._threads)} worker thread(s)")

    def stop(self, timeout: float = 10.0) -> None:
        """Signal all worker queues to drain and wait for threads to exit.

        Joins each thread with the given *timeout*.  Threads are daemons so
        they will be killed when the process exits regardless, but calling
        ``stop()`` gives in-flight tasks a chance to complete cleanly.

        Args:
            timeout: Maximum seconds to wait per thread.
        """
        logger.info("BackgroundWorkerManager stopping — draining queues …")
        try:
            faiss_update_queue.join()
        except Exception as e:
            logger.warning(f"Error while joining faiss_update_queue: {e}")

        for thread in self._threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(f"Worker thread '{thread.name}' did not finish within {timeout}s")

        logger.info("BackgroundWorkerManager stopped")


# ---------------------------------------------------------------------------
# Module-level functions (migrated from helpers.py)
# ---------------------------------------------------------------------------


def swap_issue_grouping_db_to_prod(src=FaissDBPath.local, dst=FaissDBPath.prod):
    logger.info("pushing the updated embedding data to production")
    # Validate source path
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source path does not exist: {src}")
    if not os.path.isdir(src):
        raise NotADirectoryError(f"Source path is not a directory: {src}")

    # Copy contents from src to dst, overwriting existing files
    shutil.copytree(src, dst, dirs_exist_ok=True)
    logger.info(f"Copied {src} -> {dst} (overwrite enabled)")


def requeue_failed_run_ids():
    """
    Reads failed_processing_runids_log.txt, removes those run IDs from
    processed_runids.json so they get reprocessed, then clears the log file.
    """
    failed_log_path = os.path.join(FaissConfigurations.base_path, "failed_processing_runids_log.txt")
    processed_run_ids_path = os.path.join(FaissConfigurations.base_path, "processed_runids.json")

    if not os.path.isfile(failed_log_path):
        logger.info("No failed_processing_runids_log.txt found. Nothing to requeue.")
        return

    with open(failed_log_path, "r") as f:
        content = f.read()

    failed_run_ids = re.findall(r"^Run ID:\s*(.+)$", content, re.MULTILINE)
    failed_run_ids = [r.strip() for r in failed_run_ids if r.strip()]

    if not failed_run_ids:
        logger.info("No run IDs found in failed log. Clearing empty log file.")
        return

    logger.info(f"Found {len(failed_run_ids)} failed run IDs to requeue: {failed_run_ids}")
    processed_run_ids = []
    if os.path.isfile(processed_run_ids_path):
        with open(processed_run_ids_path, "r") as f:
            processed_run_ids = json.load(f)

    original_count = len(processed_run_ids)
    processed_run_ids = [r for r in processed_run_ids if r not in failed_run_ids]
    removed_count = original_count - len(processed_run_ids)

    if len(processed_run_ids) > 500:
        processed_run_ids = processed_run_ids[100:]

    with open(processed_run_ids_path, "w") as f:
        json.dump(processed_run_ids, f, indent=2)

    logger.info(f"Removed {removed_count} run IDs from processed_runids.json.")
    logger.info("Cleared failed_processing_runids_log.txt.")


def tc_id_scheduler() -> None:
    """Start a background APScheduler job that refreshes ``run_ids.parquet``.

    The job runs every 12 hours and overwrites the local parquet cache with
    fresh run IDs from MySQL.  Safe to call multiple times — idempotent.
    """

    def _update_tc_ids() -> None:
        logger.info("Running TC IDs update background job")
        run_ids = sql_connection.fetch_runids()
        run_ids.to_parquet(parquet_file)
        logger.info("Background task updated Parquet file")

    job_id = "update_tc_ids_job"

    if scheduler.state != STATE_RUNNING:
        scheduler.start()

    if scheduler.get_job(job_id) is None:
        scheduler.add_job(_update_tc_ids, "interval", hours=12, id=job_id)
        logger.info("Scheduled TC ID update job.")
    else:
        logger.info("TC ID update job is already scheduled. Skipping re-scheduling.")


@execution_timer
async def process_tc_ids_async_bg_job(run_ids):
    # Inline import to avoid circular imports: workers.py ↔ cluster_pipeline.py
    from src.pipeline.cluster_pipeline import ClusteringPipeline

    logger.info("processing the parquet and updating faiss as background job")
    run_ids_list = run_ids["testplan_id"].tolist()
    for run_id in run_ids_list:
        if run_id and any(run_id.startswith(tag) for tag in ["QNN", "SNPE"]):
            try:
                await ClusteringPipeline(update_vector_store=True).process_run_id(run_id)
                gc.collect()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error occured while processing: {run_id}: \n{e}")
                error_log_path = os.path.join(FaissConfigurations.base_path, "failed_processing_runids_log.txt")
                with open(error_log_path, "a") as log_file:
                    log_file.write(f"\nRun ID: {run_id}\n")
                    log_file.write(f"\nFailed with error: {e}\n")
                    log_file.write(traceback.format_exc())
                    log_file.write("\n" + "-" * 80 + "\n")
                continue
        else:
            logger.info(f"Skipping processing: {run_id} doesn't start with QNN/SNPE")
    requeue_failed_run_ids()
    gc.collect()
    logger.info("[MemCleanup] Post-job memory cleanup complete")
    swap_issue_grouping_db_to_prod()
    logger.info("Finished background job processing of TC IDs")
