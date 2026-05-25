# LLD Refactor Plan — Issue Grouping `src/`

## Problem Statement

The current `src/` codebase has the following structural deficiencies:

1. **Circular import workarounds** — imports inside functions (6 locations)
2. **God classes** — `CustomEmbeddingCluster` (665 LOC), `ConsolidatedReportAnalysis` (500 LOC), `CombinedRegressionAnalysis` (800 LOC)
3. **No OOP in helpers.py** — 1361 lines of procedural functions, module-level singletons, 3 overlapping async orchestration variants
4. **Side effects at import time** — daemon threads started when modules are imported
5. **4 overlapping normalization strategies** — inconsistent preprocessing
6. **3 overlapping clustering orchestration functions** — unclear when to use which
7. **Dead code** — `FaissIVFFlatIndex`, `SearchInExistingFaiss` in `faiss_db.py` (replaced by `CustomEmbeddingCluster`)
8. **No dependency injection** — singletons hard-wired at module scope, untestable
9. **Bare `except Exception`** — silent swallowing of failures

---

## Target Architecture

```
src/
├── core/                       # Shared foundation (no business logic)
│   ├── __init__.py
│   ├── interfaces.py           # Abstract base classes for all major components
│   ├── exceptions.py           # Domain exceptions (ClusteringError, EmbeddingError, etc.)
│   └── config.py               # Typed config dataclasses (replaces scattered constants.py)
│
├── data/                       # Data access layer (I/O only)
│   ├── __init__.py
│   ├── mysql_client.py         # ConnectToMySql (was db_connections.py)
│   ├── excel_loader.py         # ExcelLoader (was data_loader.py)
│   └── gerrit_client.py        # Merged async_gerrit_client + gerrit_data_fetching_helpers
│
├── preprocessing/              # Input normalization (single source of truth)
│   ├── __init__.py
│   ├── normalizer.py           # ErrorNormalizer — merged 4 preprocessing variants
│   └── log_extractor.py        # extract_error_lines, trim_error_logs (from helpers.py)
│
├── embeddings/                 # Vector representations
│   ├── __init__.py
│   ├── base.py                 # EmbeddingProvider ABC
│   ├── qgenie_provider.py      # QGenieBGEM3Embedding (was embeddings.py)
│   └── bge_provider.py         # BGEM3Embeddings local fallback (was embeddings.py)
│
├── clustering/                 # Cluster store + search
│   ├── __init__.py
│   ├── vector_store.py         # VectorStore — centroid I/O, centroids.npy management
│   ├── metadata_store.py       # MetadataStore — metadata.json R/W, run_id tracking
│   ├── searcher.py             # ClusterSearcher — cosine + hybrid search (was inside CustomEmbeddingCluster)
│   ├── splade_encoder.py       # SPLADEEncoder singleton (was splade_clustering.py)
│   ├── hybrid_matcher.py       # HybridSPLADEMatcher (was splade_clustering.py)
│   ├── ranker.py               # ClusterRanker + ClusterCohesionAnalyzer
│   └── hdbscan_clusterer.py    # HDBSCAN wrapper + parameter config
│
├── llm/                        # Language model integration
│   ├── __init__.py
│   ├── client.py               # CustomQGenieChat + retry logic (was qgenie_llm_calls.py)
│   ├── cluster_namer.py        # generate_cluster_name() logic
│   ├── cluster_classifier.py   # classify_cluster_based_of_type() logic
│   ├── deduplicator.py         # detect_and_merge_near_duplicate_clusters() + qgenie_post_processing()
│   └── prompts.py              # All prompt templates (moved from src/prompts.py)
│
├── pipeline/                   # Orchestration (ties layers together)
│   ├── __init__.py
│   ├── cluster_pipeline.py     # ClusteringPipeline — single orchestration class (replaces 3 helpers.py variants)
│   ├── pregroup_pipeline.py    # PreGroupingPipeline — fuzzy + SPLADE pre-grouping
│   └── workers.py              # BackgroundWorkerManager — lifecycle-managed worker threads/queues
│
├── reports/                    # Report generation
│   ├── __init__.py
│   ├── kpi_calculator.py       # KPICalculator — extracted from ConsolidatedReportAnalysis
│   ├── html_renderer.py        # HTMLRenderer — template rendering utilities
│   ├── regression_report.py    # RegressionReport (was regression_api_call.py + parts of consolidated_reports_analysis.py)
│   ├── consolidated_report.py  # ConsolidatedReport (slim orchestrator, delegates to above)
│   └── enhanced_report.py      # EnhancedReport (was enhanced_consolidated_report.py)
│
├── monitoring/                 # Operational monitoring
│   ├── __init__.py
│   ├── stability_report.py     # StabilityMonitor (was stability_report.py)
│   ├── teams_notifier.py       # TeamsNotifier (was teams_helpers.py)
│   └── email_notifier.py       # EmailNotifier (was email_helpers.py)
│
├── utils/                      # Cross-cutting utilities
│   ├── __init__.py
│   ├── timer.py                # @execution_timer decorator (was execution_timer_log.py)
│   ├── logger.py               # AppLogger (was logger.py — keep in place)
│   └── run_id_utils.py         # get_prev_testplan_id logic
│
└── constants.py                # Keep as-is (already clean dataclasses); add __all__
```

---

## Dependency Rules (enforced by layer)

```
core          ← (no dependencies)
constants     ← (no dependencies)
utils         ← core
data          ← core, utils
preprocessing ← core, utils
embeddings    ← core, utils
clustering    ← core, utils, preprocessing, embeddings
llm           ← core, utils
pipeline      ← clustering, llm, preprocessing, embeddings, data
reports       ← data, pipeline, llm, utils
monitoring    ← data, utils, reports
```

**Rule:** No layer may import from a layer above it. This eliminates circular imports.

---

## Key Design Decisions

### 1. Dependency Injection via Constructor Parameters
All singletons (`sql_connection`, `faiss_runner`, SPLADE encoder) are passed as constructor arguments rather than created at module scope. A top-level `app_context.py` in the entry points (`api.py`, `background_jobs.py`) creates and wires them.

```python
# Before (helpers.py module scope):
sql_connection = ConnectToMySql()
faiss_runner = CustomEmbeddingCluster()

# After (api.py entry point):
mysql = ConnectToMySql(config=db_config)
vector_store = VectorStore(base_path=config.faiss_base_path)
pipeline = ClusteringPipeline(vector_store=vector_store, mysql=mysql, ...)
```

### 2. Single ErrorNormalizer
Replace 4 preprocessing implementations with one configurable `ErrorNormalizer`:

```python
class ErrorNormalizer:
    """Unified error log normalizer. Handles path stripping, PID removal,
    version string normalization, CamelCase splitting, and timestamp removal.

    All normalization logic lives here. Nothing else should preprocess errors.
    """
    def normalize(self, text: str) -> str: ...
    def normalize_batch(self, texts: list[str]) -> list[str]: ...
```

### 3. Split CustomEmbeddingCluster into 3 Focused Classes
```
CustomEmbeddingCluster (665 LOC, does everything)
  →  VectorStore      — manages centroids.npy on disk; add/load/save centroids
  →  MetadataStore    — manages metadata.json; run_id tracking; cluster name mapping
  →  ClusterSearcher  — cosine similarity + hybrid SPLADE search; no I/O
```

### 4. Single ClusteringPipeline (replace 3 helpers.py variants)
```python
class ClusteringPipeline:
    """Orchestrates the full clustering pipeline for a DataFrame of error logs.

    Supports sequential, concurrent, and async execution modes via a single
    entry point — removing the ambiguity of async_process_by_type(),
    async_sequential_process_by_type(), and concurrent_process_by_type().
    """
    async def run(self, df: pd.DataFrame, mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> pd.DataFrame: ...
```

### 5. Explicit Worker Lifecycle (no side effects at import)
```python
class BackgroundWorkerManager:
    """Manages daemon worker threads for async queue processing.

    Threads are started explicitly via start() — never at import time.
    Graceful shutdown via stop() with configurable timeout.
    """
    def start(self) -> None: ...
    def stop(self, timeout: float = 10.0) -> None: ...
```

### 6. Typed Domain Exceptions
```python
# core/exceptions.py
class IssueGroupingError(Exception): ...
class EmbeddingError(IssueGroupingError): ...
class ClusteringError(IssueGroupingError): ...
class LLMError(IssueGroupingError): ...
class DatabaseError(IssueGroupingError): ...
class VectorStoreError(IssueGroupingError): ...
```

---

## Implementation Phases

### Phase 1 — Core Foundation (no business logic)
Files to create: `src/core/interfaces.py`, `src/core/exceptions.py`
- Define ABCs: `EmbeddingProvider`, `ClusterStore`, `Normalizer`, `LLMClient`
- Define domain exceptions hierarchy
- No functional changes yet

### Phase 2 — Data Layer Cleanup
Files to refactor: `db_connections.py` → `src/data/mysql_client.py`
Files to merge: `async_gerrit_client.py` + `gerrit_data_fetching_helpers.py` → `src/data/gerrit_client.py`
Files to move: `data_loader.py` → `src/data/excel_loader.py`
- Implement `DatabaseConnection` ABC cleanly
- Remove duplicate Gerrit logic between the two files
- Add docstrings to all public methods

### Phase 3 — Preprocessing Consolidation
Files to create: `src/preprocessing/normalizer.py`, `src/preprocessing/log_extractor.py`
- Merge `preprocess_error_log()`, `ErrorNormalizer.normalize()`, `trim_error_logs()`, `extract_error_lines()` into one `ErrorNormalizer` class
- Keep old functions in helpers.py as thin shims calling `ErrorNormalizer` (for backward compatibility during transition)

### Phase 4 — Clustering Layer Split
Files to create: `src/clustering/vector_store.py`, `src/clustering/metadata_store.py`, `src/clustering/searcher.py`
- Extract `VectorStore` (centroids.npy management) from `CustomEmbeddingCluster`
- Extract `MetadataStore` (metadata.json management) from `CustomEmbeddingCluster`
- Extract `ClusterSearcher` (similarity search) — no more imports inside `_get_hybrid_matcher()`
- Move `SPLADEEncoder`, `HybridSPLADEMatcher` to `src/clustering/splade_encoder.py`, `src/clustering/hybrid_matcher.py`
- Move `ClusterRanker`, `ClusterCohesionAnalyzer` to `src/clustering/ranker.py`
- Delete `faiss_db.py` (dead code)

### Phase 5 — LLM Layer Separation
Files to create: `src/llm/cluster_namer.py`, `src/llm/cluster_classifier.py`, `src/llm/deduplicator.py`
- Move `generate_cluster_name()` to `ClusterNamer`
- Move `classify_cluster_based_of_type()` to `ClusterClassifier`
- Move `qgenie_post_processing()` + `detect_and_merge_near_duplicate_clusters()` to `Deduplicator`
- Fix: circuit breaker pattern for QGenie API (not just retry)

### Phase 6 — Pipeline Consolidation
Files to create: `src/pipeline/cluster_pipeline.py`, `src/pipeline/pregroup_pipeline.py`, `src/pipeline/workers.py`
- `ClusteringPipeline.run(df, mode)` replaces `async_process_by_type`, `async_sequential_process_by_type`, `concurrent_process_by_type`
- `PreGroupingPipeline.run(df)` replaces `fuzzy_cluster_grouping` + `splade_pregroup`
- `BackgroundWorkerManager` replaces daemon threads started at import in `failure_analyzer.py`
- Clean up `helpers.py` — keep only helper functions that don't belong elsewhere; target < 400 LOC

### Phase 7 — Reports Decomposition
Files to create: `src/reports/kpi_calculator.py`, `src/reports/html_renderer.py`
- Extract `KPICalculator` from `ConsolidatedReportAnalysis` (40+ methods → focused KPI class)
- Extract `HTMLRenderer` for shared rendering utilities
- `ConsolidatedReport` becomes a thin orchestrator (< 200 LOC)
- `CombinedRegressionAnalysis` loses inherited bloat; uses composition over inheritance

### Phase 8 — Monitoring Layer
Files to refactor: `stability_report.py`, `teams_helpers.py`, `email_helpers.py` → `src/monitoring/`
- Wrap into `StabilityMonitor`, `TeamsNotifier`, `EmailNotifier` classes
- All `send_*` functions become methods with proper `__init__` for config injection

### Phase 9 — Documentation Pass
- Add Google-style docstrings to all public classes and methods
- Add `__all__` to every module
- Update type annotations throughout

### Phase 10 — Cleanup & Validation
- Remove `faiss_db.py` entirely
- Remove inline imports (fix circular deps properly)
- Run `black -l 120` + `isort -l 120 -m 3` across all modified files
- Verify all entry points (`api.py`, `app.py`, `background_jobs.py`) still work

---

## Files to Create (New)

| File | Purpose |
|------|---------|
| `src/core/__init__.py` | |
| `src/core/interfaces.py` | ABCs for embedding, clustering, LLM |
| `src/core/exceptions.py` | Domain exception hierarchy |
| `src/data/__init__.py` | |
| `src/preprocessing/__init__.py` | |
| `src/preprocessing/normalizer.py` | Unified ErrorNormalizer |
| `src/preprocessing/log_extractor.py` | Log extraction utilities |
| `src/embeddings/__init__.py` | |
| `src/embeddings/base.py` | EmbeddingProvider ABC |
| `src/clustering/__init__.py` | |
| `src/clustering/vector_store.py` | VectorStore |
| `src/clustering/metadata_store.py` | MetadataStore |
| `src/clustering/searcher.py` | ClusterSearcher |
| `src/clustering/ranker.py` | ClusterRanker + ClusterCohesionAnalyzer |
| `src/clustering/hdbscan_clusterer.py` | HDBSCAN wrapper |
| `src/llm/__init__.py` | |
| `src/llm/client.py` | CustomQGenieChat |
| `src/llm/cluster_namer.py` | ClusterNamer |
| `src/llm/cluster_classifier.py` | ClusterClassifier |
| `src/llm/deduplicator.py` | Deduplicator |
| `src/pipeline/__init__.py` | |
| `src/pipeline/cluster_pipeline.py` | ClusteringPipeline |
| `src/pipeline/pregroup_pipeline.py` | PreGroupingPipeline |
| `src/pipeline/workers.py` | BackgroundWorkerManager |
| `src/reports/__init__.py` | |
| `src/reports/kpi_calculator.py` | KPICalculator |
| `src/reports/html_renderer.py` | HTMLRenderer |
| `src/monitoring/__init__.py` | |
| `src/utils/__init__.py` | |
| `src/utils/timer.py` | @execution_timer |
| `src/utils/run_id_utils.py` | get_prev_testplan_id logic |

## Files to Refactor (Move/Split)

| Old File | New Location(s) |
|----------|----------------|
| `src/db_connections.py` | `src/data/mysql_client.py` |
| `src/data_loader.py` | `src/data/excel_loader.py` |
| `src/async_gerrit_client.py` + `src/gerrit_data_fetching_helpers.py` | `src/data/gerrit_client.py` |
| `src/embeddings.py` | `src/embeddings/qgenie_provider.py` + `src/embeddings/bge_provider.py` |
| `src/custom_clustering.py` | `src/clustering/vector_store.py` + `src/clustering/metadata_store.py` + `src/clustering/searcher.py` |
| `src/splade_clustering.py` | `src/clustering/splade_encoder.py` + `src/clustering/hybrid_matcher.py` + `src/clustering/ranker.py` |
| `src/helpers.py` (1361 LOC) | Distributed to `pipeline/`, `preprocessing/`, `data/`; residual < 200 LOC |
| `src/qgenie_llm_calls.py` | `src/llm/client.py` + `src/llm/cluster_namer.py` + `src/llm/cluster_classifier.py` + `src/llm/deduplicator.py` |
| `src/failure_analyzer.py` | Slimmed down; delegates to `pipeline/cluster_pipeline.py` |
| `src/prompts.py` | `src/llm/prompts.py` |
| `src/execution_timer_log.py` | `src/utils/timer.py` |
| `src/get_prev_testplan_id.py` | `src/utils/run_id_utils.py` |
| `src/stability_report.py` | `src/monitoring/stability_report.py` |
| `src/teams_helpers.py` | `src/monitoring/teams_notifier.py` |
| `src/email_helpers.py` | `src/monitoring/email_notifier.py` |
| `src/consolidated_reports_analysis.py` | `src/reports/consolidated_report.py` + `src/reports/regression_report.py` + `src/reports/kpi_calculator.py` |
| `src/enhanced_consolidated_report.py` | `src/reports/enhanced_report.py` |

## Files to Delete (Dead Code)

| File | Reason |
|------|--------|
| `src/faiss_db.py` | Fully replaced by `CustomEmbeddingCluster`; `FaissIVFFlatIndex` and `SearchInExistingFaiss` are unused |

---

## Docstring Standard (Google Style)

```python
class ClusterSearcher:
    """Performs cosine and hybrid SPLADE similarity search over cluster centroids.

    Decoupled from storage — receives a VectorStore and HybridSPLADEMatcher
    via constructor injection. Does not perform any disk I/O.

    Args:
        vector_store: Loaded VectorStore containing centroids.
        hybrid_matcher: Optional SPLADE hybrid scorer; falls back to pure cosine.
        similarity_threshold: Minimum cosine similarity to return a match.

    Example:
        searcher = ClusterSearcher(vector_store=vs, hybrid_matcher=hm)
        result = searcher.search(embedding=emb, cluster_type="quantizer")
    """

    def search(self, embedding: np.ndarray, cluster_type: str) -> SearchResult | None:
        """Search for the best matching cluster for a given embedding.

        Args:
            embedding: Normalized embedding vector (shape: [dim]).
            cluster_type: Test type key (e.g., "quantizer", "verifier").

        Returns:
            SearchResult with cluster name and score, or None if no match
            above threshold.

        Raises:
            VectorStoreError: If the cluster_type has no loaded index.
        """
```

---

## Success Criteria

- [ ] No imports inside functions (0 occurrences of `import` inside function bodies)
- [ ] No module-level side effects (no singletons or threads created at import time)
- [ ] `helpers.py` reduced from 1361 LOC to < 300 LOC (thin orchestration shims only)
- [ ] `CustomEmbeddingCluster` (665 LOC) replaced by 3 focused classes each < 200 LOC
- [ ] `ConsolidatedReportAnalysis` (500 LOC) replaced by orchestrator < 150 LOC + extracted KPI/HTML classes
- [ ] All public classes and methods have Google-style docstrings
- [ ] `black -l 120` and `isort -l 120 -m 3` pass with no changes
- [ ] All entry points (`api.py`, `app.py`, `background_jobs.py`) still import and run
- [ ] `faiss_db.py` deleted
- [ ] 0 bare `except Exception` without re-raise or logging

## Implementation Status (as of 2026-04-30)

### Completed migrations
- `src/data/` — mysql_client.py, excel_loader.py, gerrit_client.py ✓
- `src/core/` — interfaces.py, exceptions.py ✓
- `src/embeddings/` — qgenie_provider.py, bge_provider.py ✓
- `src/clustering/` — vector_store, metadata_store, searcher, ranker, splade_encoder, hybrid_matcher, hdbscan_clusterer ✓
- `src/llm/` — client.py, cluster_namer.py, cluster_classifier.py, deduplicator.py, prompts.py ✓
- `src/monitoring/` — stability_report.py, teams_notifier.py, email_notifier.py ✓
- `src/utils/` — timer.py, run_id_utils.py ✓
- `src/preprocessing/` — normalizer.py, log_extractor.py ✓
- `src/pipeline/` — workers.py ✓
- `src/reports/` — html_renderer.py, kpi_calculator.py, regression_report.py, consolidated_report.py, enhanced_report.py ✓

### Remaining shims (backward-compat, not deleted)
- `src/consolidated_reports_analysis.py` → shim → `src/reports/consolidated_report.py`
- `src/regression_api_call.py` → shim → `src/reports/regression_report.py`
- `src/enhanced_consolidated_report.py` — kept as standalone script (CLAUDE.md entry point)

### Deleted old files
- src/execution_timer_log.py, src/stability_report.py, src/teams_helpers.py, src/email_helpers.py, src/get_prev_testplan_id.py
- src/helpers.py, src/db_connections.py, src/data_loader.py, src/embeddings.py, src/faiss_db.py
- src/async_gerrit_client.py, src/gerrit_data_fetching_helpers.py, src/splade_clustering.py

### Still TODO (Phase 6+)
- ClusteringPipeline / PreGroupingPipeline / BackgroundWorkerManager (pipeline/)
- Slim helpers.py to < 400 LOC (currently no helpers.py, logic in pipeline/)
- Slim CustomEmbeddingCluster (in custom_clustering.py, ~665 LOC)

---



| Risk | Mitigation |
|------|-----------|
| Circular import resolution breaks runtime | Introduce thin shim imports in old locations during transition; remove only when all call sites updated |
| `centroids.npy` / `metadata.json` alignment invariant violated during VectorStore/MetadataStore split | Add explicit assert in `VectorStore.load()` + unit test for row count equality |
| SPLADE singleton shared state across tests | `SPLADEEncoder` gains `reset()` classmethod for test isolation |
| `api.py` / `background_jobs.py` breaking during incremental refactor | Keep old `src/helpers.py` and `src/custom_clustering.py` as shim files until Phase 6 complete |
| Workers started in wrong order | `BackgroundWorkerManager` enforces `start_order: list[str]` in config |
