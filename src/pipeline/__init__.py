"""Pipeline package — orchestration layer tying all components together.

Modules
-------
workers            — :class:`BackgroundWorkerManager` for lifecycle-managed
                     daemon threads and async update queues.
pregroup_pipeline  — :class:`PreGroupingPipeline` for fuzzy + SPLADE
                     pre-grouping before HDBSCAN.
cluster_pipeline   — :class:`ClusteringPipeline`, the single authoritative
                     orchestrator replacing the three legacy
                     ``helpers.py`` variants.

Layering
--------
``pipeline`` imports from ``src.clustering``, ``src.llm``, ``src.preprocessing``,
``src.embeddings``, ``src.data``, ``src.core``, and ``src.utils``.
It must **not** be imported by any module in those lower layers.
"""

from src.pipeline.workers import BackgroundWorkerManager

__all__ = ["BackgroundWorkerManager"]
