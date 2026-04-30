"""Single authoritative clustering pipeline orchestrator.

Provides :class:`ClusteringPipeline`, which replaces the three overlapping
orchestration variants in ``helpers.py``:

* ``async_process_by_type()`` — concurrent asyncio.gather
* ``async_sequential_process_by_type()`` — sequential per-type
* ``concurrent_process_by_type()`` — ThreadPoolExecutor

All three are replaced by a single ``run(df, mode)`` entry point.

Execution modes
---------------
:attr:`ExecutionMode.SEQUENTIAL`
    Process each test type in series.  Safe for low-memory environments and
    for background jobs where throughput > latency.
:attr:`ExecutionMode.CONCURRENT`
    Process all test types concurrently with ``asyncio.gather``.  Maximises
    throughput on wide DataFrames with many types.
:attr:`ExecutionMode.PROCESS_POOL`
    Runs each type in a separate thread via ``ThreadPoolExecutor`` (each
    thread runs its own event loop).  CPU-bound fallback.

Layering
--------
This module imports from ``src.constants``, ``src.logger``, ``src.utils``,
and uses deferred imports for ``src.failure_analyzer`` and ``src.helpers``
to avoid circular imports.
"""

from __future__ import annotations

import asyncio
import enum
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.constants import FaissConfigurations
from src.failure_analyzer import FailureAnalyzer
from src.logger import AppLogger
from src.pipeline.workers import faiss_update_queue
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["ClusteringPipeline", "ExecutionMode"]


class ExecutionMode(enum.Enum):
    """Execution strategy for :class:`ClusteringPipeline.run`.

    Attributes:
        SEQUENTIAL: Process each test type one after another (default).
        CONCURRENT: Process all test types concurrently with asyncio.gather.
        PROCESS_POOL: Run each type in a ThreadPoolExecutor worker.
    """

    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    PROCESS_POOL = "process_pool"


class ClusteringPipeline:
    """Full clustering pipeline — loads, preprocesses, clusters, and persists.

    This class is the single entry point for running the issue-grouping
    pipeline on a DataFrame of test failures.  It groups the DataFrame by
    ``type``, runs :class:`~src.failure_analyzer.FailureAnalyzer` on each
    group, and optionally enqueues the results for FAISS DB persistence.

    Args:
        update_vector_store: When ``True``, enqueue clustered results to the
            FAISS update queue after processing.  Defaults to ``False``.

    Example::

        pipeline = ClusteringPipeline(update_vector_store=True)
        results = await pipeline.run(df, mode=ExecutionMode.SEQUENTIAL, run_id="QNN-001")
    """

    def __init__(self, update_vector_store: bool = False) -> None:
        self.update_vector_store = update_vector_store

    @execution_timer
    async def run(
        self,
        df: pd.DataFrame,
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        run_id: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run the full clustering pipeline on *df*.

        Args:
            df: DataFrame of failure records.  Must include a ``type`` column.
            mode: Execution strategy.  See :class:`ExecutionMode`.
            run_id: Optional run identifier logged and used for FAISS DB
                persistence.  Required when ``update_vector_store=True``.

        Returns:
            Dictionary mapping ``type`` → clustered :class:`~pandas.DataFrame`.

        Raises:
            src.core.exceptions.PipelineError: On orchestration-level failures.
        """
        logger.info(f"[ClusteringPipeline] mode={mode.value} run_id={run_id} types={df.type.unique().tolist()}")

        if mode == ExecutionMode.SEQUENTIAL:
            results = await self._run_sequential(df)
        elif mode == ExecutionMode.CONCURRENT:
            results = await self._run_concurrent(df)
        elif mode == ExecutionMode.PROCESS_POOL:
            results = await self._run_process_pool(df)
        else:
            raise ValueError(f"Unknown ExecutionMode: {mode}")

        if results and self.update_vector_store and run_id:
            self._enqueue_faiss_update(results, run_id)

        return results

    async def _run_sequential(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Process each type sequentially.

        Args:
            df: Full DataFrame (all types).

        Returns:
            Results dict.
        """
        results: dict[str, pd.DataFrame] = {}
        analyzer = FailureAnalyzer()

        for type_, group_df in df.groupby("type"):
            logger.info(f"[ClusteringPipeline] sequential: processing type={type_} ({len(group_df)} rows)")
            results[type_] = await analyzer.analyze(dataframe=group_df.reset_index(drop=True))

        return results

    async def _run_concurrent(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Process all types concurrently with asyncio.gather.

        Args:
            df: Full DataFrame (all types).

        Returns:
            Results dict.
        """
        results: dict[str, pd.DataFrame] = {}
        analyzer = FailureAnalyzer()

        async def _process_group(type_: str, group_df: pd.DataFrame) -> None:
            results[type_] = await analyzer.analyze(dataframe=group_df.reset_index(drop=True))

        tasks = [_process_group(t, g) for t, g in df.groupby("type")]
        await asyncio.gather(*tasks)
        return results

    async def _run_process_pool(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Process each type in a ThreadPoolExecutor worker.

        Each worker runs its own asyncio event loop via ``asyncio.run()``.

        Args:
            df: Full DataFrame (all types).

        Returns:
            Results dict.
        """

        def _run_in_thread(group_df: pd.DataFrame) -> pd.DataFrame:
            async def _run() -> pd.DataFrame:
                return await FailureAnalyzer().analyze(dataframe=group_df.reset_index(drop=True))

            return asyncio.run(_run())

        results: dict[str, pd.DataFrame] = {}
        grouped = list(df.groupby("type"))

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [loop.run_in_executor(executor, _run_in_thread, g) for _, g in grouped]
            analysis_results = await asyncio.gather(*futures)

        for (type_, _), result in zip(grouped, analysis_results):
            results[type_] = result

        return results

    def _enqueue_faiss_update(self, results: dict[str, pd.DataFrame], run_id: str) -> None:
        """Enqueue clustering results for async FAISS DB update.

        Args:
            results: Per-type clustered DataFrames.
            run_id: Run identifier for logging.
        """
        clustered_df = pd.concat(
            [df.assign(cluster_type=type_) for type_, df in results.items()],
            ignore_index=True,
        )
        faiss_update_queue.put((clustered_df, run_id, "update"))
        logger.info(f"[ClusteringPipeline] enqueued FAISS update for run_id={run_id}")

    @execution_timer
    async def process_run_id(
        self,
        run_id: str,
        mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> dict[str, pd.DataFrame]:
        """Load data for *run_id* from MySQL and run the full pipeline.

        This is a convenience wrapper used by background jobs.

        Args:
            run_id: Test plan ID to process (must start with ``"QNN"`` or
                ``"SNPE"``).
            mode: Execution strategy.

        Returns:
            Per-type clustered DataFrames, or empty dict if run_id is already
            processed.
        """
        processed_run_ids_path = os.path.join(FaissConfigurations.base_path, "processed_runids.json")
        processed_run_ids: list[str] = []
        if os.path.isfile(processed_run_ids_path):
            with open(processed_run_ids_path) as f:
                processed_run_ids = json.load(f)

        if run_id in processed_run_ids:
            logger.info(f"[ClusteringPipeline] Skipping {run_id} — already processed")
            return {}

        df = FailureAnalyzer().load_data(tc_id=run_id)
        pipeline = ClusteringPipeline(update_vector_store=True)
        results = await pipeline.run(df, mode=mode, run_id=run_id)
        gc.collect()
        return results
