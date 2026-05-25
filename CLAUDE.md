# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
export UV_CACHE_DIR='/path/to/your/cache'   # optional: redirect uv cache
uv sync                                      # Install dependencies (uses devpi.qualcomm.com index)
source .venv/bin/activate                    # Activate virtual environment
```

### Running

**Docker (preferred — see `Makefile` for the full target list):**
```bash
make up                  # prod stack: nginx (8080) → fastapi (8010) + streamlit (8501)
make dev                 # dev overlay: bind-mounts ./ into container + uvicorn --reload
make redeploy-fastapi    # rebuild & restart only fastapi (streamlit keeps serving)
make logs                # tail all services; `make ps` for status; `make health` for probes
make splade-up           # SPLADE GPU stack — run on the GPU host (separate compose file)
make help                # list every target
```

**Bare-metal (without Docker):**
```bash
# FastAPI server (development)
uvicorn api:app --reload --port 8010

# FastAPI server (production — multi-worker)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker --max-requests 1 --max-requests-jitter 0 \
  -b 0.0.0.0:8010 "api:app" --graceful-timeout 30 --keep-alive 5

# SPLADE encoder microservice (deploy on GPU machine; clients set SPLADE_API_URL)
uvicorn splade_api:app --reload --port 8000

# Background job scheduler (runs independently of the API)
python background_jobs.py

# Standalone enhanced HTML report generator
python src/enhanced_consolidated_report.py --qairt_id qaisw-v2.46.0.260319041023_nightly
python src/enhanced_consolidated_report.py --qairt_id <id> --no_llm   # skip LLM, metrics only
python src/enhanced_consolidated_report.py --qairt_id <id> --cache_llm # cache LLM results

# Streamlit UI
streamlit run app.py
```

### Testing module __main__ blocks
These are the only "tests" — each module has a self-contained `__main__` block:
```bash
# Hourly stability report — generates synthetic HTML to /tmp/stability_report_test.html
python -m src.monitoring.hourly_report

# Enhanced consolidated report (requires joblib artifact on disk)
python src/enhanced_consolidated_report.py --qairt_id <id> --no_llm
```

### Required environment variables
```bash
# Required (code raises if unset):
QGENIE_API_KEY=...          # LLM service (cluster naming/classification)
MYSQL_PASSWORD=...          # MySQL auth

# Optional — code-level defaults exist in src/data/mysql_client.py and
# src/reports/regression_report.py; leave unset to use those defaults:
MYSQL_USER=...              # MySQL user
MYSQL_HOST=...              # MySQL host (e.g. hydcrpmysqlprd10)
MYSQL_DB=...                # MySQL schema
ISSUE_GROUPING_API_URL=...  # FastAPI base URL used by Streamlit pages
SPLADE_API_URL=...          # Remote SPLADE encoder; if unset, SPLADE loads locally
QA2_CONFIG_FILE_PATH=...    # Override the QA2 web config YAML path
GERRIT_USER_NAME=...        # Gerrit code-review data
GERRIT_HTTP_PASSWORD=...    # Gerrit HTTP password

# Hourly stability monitor (GET /api/running_jobs/):
DAG_API_BEARER_TOKEN=...    # Bearer token for the DAG API (qa2 dashboard)
TEAMS_WEBHOOK_URL=...       # Power Automate webhook — default notification channel
SEND_EMAIL=true             # opt-in to also send the HTML email report (default: false)
```

**Compose env-var trap.** `docker-compose.yml` uses **list-form** pass-through (`- MYSQL_HOST` with no `=`) for optional vars: if the host shell has the var set it forwards, otherwise the var stays *unset* in the container. **Do not** rewrite these as `MYSQL_HOST: ${MYSQL_HOST:-}` — that map-form sets the var to the empty string in the container, which defeats Python's `os.environ.get("MYSQL_HOST", default)` because empty-but-set ≠ unset.

### Code Quality
After any code change, format with line length 120:
```bash
black -l 120 <files>
isort -l 120 -m 3 <files>
```

## Architecture

ML-powered error log clustering and regression analysis system for Qualcomm's QNN/QAIRT test infrastructure.

### Entry Points
- **`api.py`** — FastAPI REST API (port 8010 in docker, 8001 historically). Swagger UI at exactly `/api` (no trailing slash) — set by `docs_url="/api"`. Starts two background threads at startup: `tc_id_scheduler` (issue grouping for new run IDs) and `consolidated_report_worker` (queue-based HTML report generation).
- **`splade_api.py`** — Standalone SPLADE encoder microservice (port 8000). Deploy on GPU machine; client machines set `SPLADE_API_URL` env var and `SPLADEEncoder` calls it remotely instead of loading the model locally. GPU work serialised behind an `asyncio.Semaphore(1)` to prevent vGPU OOM.
- **`app.py`** — Streamlit multi-page UI (port 8501): Issue Grouping, Regression, Error Classification pages (in `streamlit_pages/`). Pages import `src/data/mysql_client` and `src/failure_analyzer` directly, so they need the same env vars and NFS access as the API.
- **`background_jobs.py`** — Standalone APScheduler process running two interval jobs: `run_ids_issue_grouping_processing` and `consolidated_report_processing_job`.

### Docker Stack

Three compose files, driven through `Makefile`:

- **`docker-compose.yml`** — prod stack: `nginx` (8080) → `fastapi` (8010) + `streamlit` (8501). Both app services share the `x-app-env` and `x-host-user` YAML anchors.
- **`docker-compose.dev.yml`** — overlay applied via `make dev`. Bind-mounts `./` into `/app` and runs uvicorn `--reload` so source edits hot-reload.
- **`docker-compose.splade.yml`** — separate stack for the GPU host. Reserves an NVIDIA device, mounts a named `splade_hf_cache` volume, exposes port 8000.

**NFS / root-squash.** `/prj/qct` filers are exported with root-squash, so the container must run as the host user. `x-host-user` pins `user: 4741839:200` plus supplementary `group_add` GIDs (`mlg`, `aisw-team`, `access.aisw.qipl.filer.*`). On a different host, override via `export UID=$(id -u) GID=$(id -g)` before `make up`. `PYTHONDONTWRITEBYTECODE=1` is set so the non-root user doesn't try to write `__pycache__/` next to root-owned source.

**autofs propagation.** `/prj/qct/webtech_scratch*` are autofs trigger points. The bind mount uses `propagation: rslave` (long-form `type: bind`) so new automounts on the host propagate into the container. Without `rslave`, a fresh path under autofs (e.g. a brand-new run-id directory) appears missing inside the container even though it resolves on the host.

**Re-deployments.** Use `make redeploy-fastapi` / `make redeploy-streamlit` to rebuild and restart a single service after `git pull` — the other service keeps serving. The vector DB lives in a host bind-mount (`./issue_grouping_db`), so it survives container recreation; `make clean-volumes` is the only target that destroys it.

### Module Layout (`src/`)

The codebase uses a layered modular structure. Dependencies flow strictly downward — upper layers may import from lower layers, never the reverse.

```
src/
├── failure_analyzer.py      ← pipeline entry point (FailureAnalyzer class)
├── nightly_stability_job.py ← hourly stability check orchestration
├── custom_clustering.py     ← backward-compat shim (deprecated, wraps src.clustering.*)
├── enhanced_consolidated_report.py ← standalone CLI report script
├── constants.py             ← all configuration constants and dataclasses
├── logger.py                ← AppLogger singleton
│
├── core/         interfaces.py, exceptions.py
├── data/         mysql_client.py, excel_loader.py, gerrit_client.py
├── embeddings/   base.py (FallbackEmbeddings), qgenie_provider.py, bge_provider.py
├── preprocessing/ normalizer.py (preprocess_error_log, trim_error_logs), log_extractor.py
├── clustering/   vector_store.py, metadata_store.py, searcher.py, hybrid_matcher.py,
│                 splade_encoder.py, hdbscan_clusterer.py, ranker.py
├── llm/          client.py, cluster_namer.py, cluster_classifier.py, deduplicator.py, prompts.py
├── pipeline/     cluster_pipeline.py (ClusteringPipeline), pregroup_pipeline.py, workers.py
├── monitoring/   hourly_report.py, teams_notifier.py, email_notifier.py
├── reports/      consolidated_report.py, regression_report.py, enhanced_report.py,
│                 html_renderer.py, kpi_calculator.py
└── utils/        timer.py (@execution_timer), run_id_utils.py, excel_exporter.py
```

### Core Pipeline

**`src/failure_analyzer.py`** — `FailureAnalyzer` class. Orchestrates per-test-type clustering:
1. Load data via `src.data.excel_loader.ExcelLoader` or `src.data.mysql_client.get_tc_id_df`
2. Preprocess/normalize logs (`src.preprocessing.normalizer`, `src.preprocessing.log_extractor`)
3. Check `CustomEmbeddingCluster` (shim over `src.clustering.*`) for existing cluster matches
4. SPLADE pre-grouping pass (`src.pipeline.pregroup_pipeline.splade_pregroup`) — catches semantically equivalent errors before HDBSCAN
5. Fuzzy pre-grouping pass (`fuzzy_cluster_grouping`) for near-identical short texts
6. For remaining ungrouped errors: generate embeddings → HDBSCAN → LLM naming/classification
7. Post-processing: `ClusterRanker` + `ClusterCohesionAnalyzer` add cosine-similarity metadata columns
8. Enqueue to `faiss_update_queue` for async vector DB persistence

**`src/pipeline/cluster_pipeline.py`** — `ClusteringPipeline` class. Single orchestration entry point replacing the old `helpers.py` variants. `run(df, mode)` accepts `ExecutionMode.SEQUENTIAL`, `CONCURRENT`, or `PROCESS_POOL`.

**`src/pipeline/workers.py`** — `BackgroundWorkerManager` class. Owns all daemon worker threads. No import-time side effects — threads start only when `.start()` is called. One worker: `_faissdb_update_worker` processes the `faiss_update_queue` for async centroid persistence.

**`src/pipeline/pregroup_pipeline.py`** — `check_if_issue_alread_grouped`, `fuzzy_cluster_grouping`, `splade_pregroup`. Two pre-grouping passes before HDBSCAN.

### Clustering Layer (`src/clustering/`)

- **`vector_store.py`** — `VectorStore`: manages `centroids.npy` (normalized embeddings)
- **`metadata_store.py`** — `MetadataStore`: manages `metadata.json` (`{cluster_name: {class, run_ids: {…}}}`)
- **`searcher.py`** — `ClusterSearcher`: cosine similarity search over vector store
- **`hybrid_matcher.py`** — `HybridSPLADEMatcher`: α=0.55 cosine + β=0.45 SPLADE dot-product; class-level `_cluster_vec_cache` for in-memory SPLADE encodings; falls back to pure cosine if encoder unavailable
- **`splade_encoder.py`** — `SPLADEEncoder`: singleton transformer (`naver/splade-cocondenser-ensembledistil`); encodes text to sparse scipy CSR via `log(1+relu(logits)).max(dim=1)`; `release()` called on FastAPI shutdown
- **`ranker.py`** — `ClusterRanker` (adds `rank`, `representativeness_score`, `is_core_member`), `ClusterCohesionAnalyzer` (adds `cluster_cohesion_score`, `is_loose_cluster`), plus `merge_similar_clusters`, `reassign_unclustered_logs`, `update_labels_with_merged_clusters`

**`src/custom_clustering.py`** — Deprecated backward-compatibility shim. Thin `CustomEmbeddingCluster` wrapper that composes `VectorStore` + `MetadataStore` + `ClusterSearcher`. New code should use `src.clustering.*` directly.

### Embeddings (`src/embeddings/`)

`FallbackEmbeddings` (`base.py`): tries `QGenieEmbeddingsProvider` first, falls back to local BGE-M3 model (`models/models--BAAI--bge-m3/`).

### LLM Layer (`src/llm/`)

- **`client.py`** — `CustomQGenieChat` wrapping `QGenieChat` (QGenie SDK → Vertex AI Gemini 2.5 Pro/Flash) with 10-attempt exponential backoff retry
- **`cluster_namer.py`** — `generate_cluster_name_for_single_rows()`
- **`cluster_classifier.py`** — `assign_cluster_class()`
- **`deduplicator.py`** — `detect_and_merge_near_duplicate_clusters()`, `qgenie_post_processing()`, `subcluster_verifier_failed()`
- **`prompts.py`** — All LLM system messages and prompt templates

### Hourly Stability Monitor (`src/monitoring/`)

Triggered by `GET /api/running_jobs/` → `src/nightly_stability_job.py`.

**Flow:**
1. DAG API call (`NIGHTLY_EXECUTION.DAG_API_BASE`) — filters `run_id` starting with `"QNN"` and containing `"auto"`
2. Each run's `excel_report_path` → `pickle.load()` → DataFrame
3. `analyze_type_failures(df)` → `dict[str, TypeStats]`
   - Failure % = `FAIL / (Total − PARENT_FAIL − NOT_RUN) × 100`
   - Flags type when failure rate ≥ `StabilityReportConfig.FAILURE_THRESHOLD` (50%)
   - Non-primary types also build `soc_name → host → FAIL count` mapping (top-5 hosts per SoC)
4. `RunAnalysis(run_id, job_info, type_stats)` — aggregate container
5. **Notifications** (if `processed_runs` non-empty): `TEAMS_WEBHOOK_URL` → two Adaptive Cards; `SEND_EMAIL=true` → HTML via `smtphost.qualcomm.com`
6. Runs with `has_flags=False` appear only in the Overview table, not the detailed sections

**Teams one-time setup:** channel → `...` → Workflows → "Send webhook alerts to a channel" → copy URL → set `TEAMS_WEBHOOK_URL`.

### Reports (`src/reports/`)

- **`consolidated_report.py`** — `ConsolidatedReportAnalysis` / `CombinedRegressionAnalysis`: HTML regression reports with executive summaries, failure tables, Gerrit change tracking, QGenie insights
- **`regression_report.py`** — Two-run-ID regression comparison
- **`enhanced_report.py`** / **`src/enhanced_consolidated_report.py`** — Reads `joblib` artifact, generates KPI dashboard, BU cards, heatmaps, cluster analysis HTML
- **`html_renderer.py`** / **`kpi_calculator.py`** — Rendering and metric helpers

### Data Flow
```
User Input (run_id or Excel file)
  → src.data.mysql_client / src.data.excel_loader
  → src.preprocessing.normalizer (preprocess_error_log, trim_error_logs)
  → CustomEmbeddingCluster.batch_search (hybrid cosine+SPLADE via src.clustering.*)
  → Cache hit → return existing cluster name
  → Cache miss → pregroup_pipeline (fuzzy + SPLADE) → FallbackEmbeddings → HDBSCAN
               → src.llm.cluster_namer / cluster_classifier
  → src.clustering.ranker (ClusterRanker + ClusterCohesionAnalyzer)
  → faiss_update_queue → BackgroundWorkerManager._faissdb_update_worker (async)
  → MySQL update via update_error_map_qgenie_table
```

### Vector Database Layout
`issue_grouping_db/` — one folder per test type (e.g. `quantizer_custom`). Each folder:
- `centroids.npy` — normalized embedding centroids; **row order must match `metadata.json` key order**
- `metadata.json` — `{cluster_name: {class, run_ids: {run_id: {tc_uuid: {...}}}}}`
- `splade_vectors.npz` — scipy CSR sparse matrix; rows aligned with `splade_cluster_names.json`
- `splade_cluster_names.json` — ordered cluster names for SPLADE index alignment
- `processed_runids.json` — guards against re-processing
- `failed_processing_runids_log.txt` — error log

### Key Constants (`src/constants.py`)
- `ClusterSpecificKeys.non_grouped_key = -1` — Unclassified errors
- `FaissConfigurations.base_path = "issue_grouping_db"` — Vector DB root
- `SPLADEConfigurations` — model name, `use_quantized` flag (default True → `rasyosef/splade-small` ~17 MB), hybrid α/β weights, EMA decay=0.85, cohesion threshold, core member percentile
- `CONSOLIDATED_REPORTS.prev_release_info` — Hard-coded previous release string; **must be updated monthly**
- `StabilityReportConfig` — `FAILURE_THRESHOLD` (0.50), `SENDER`/`RECIPIENT`, `SEND_EMAIL`, `TEAMS_WEBHOOK_URL`

### External Dependencies
- **MySQL** (`hydcrpmysqlprd10`) — schema `mlg_qa.{test_type}`
- **QGenie API** — LLM service (`QGENIE_API_KEY`)
- **Gerrit** (`review.qualcomm.com`) — `GERRIT_USER_NAME`, `GERRIT_HTTP_PASSWORD`
- **Vertex AI** — Gemini 2.5 Pro/Flash via QGenie SDK
- **PyPI** — internal Qualcomm index: `https://devpi.qualcomm.com/qcom/dev/+simple`

### Important Invariants
- `centroids.npy` row count must equal `metadata.json` key count — `VectorStore.update()` asserts this.
- Run IDs must start with `"QNN"` or `"SNPE"` to be processed by `process_tc_ids_async_bg_job`.
- `BackgroundWorkerManager` has no import-time side effects; call `.start()` explicitly (done in `api.py` lifespan and `background_jobs.py`).
- `SPLADEEncoder` is a singleton; `release()` must be called on shutdown to free GPU/CPU memory and clear `HybridSPLADEMatcher._cluster_vec_cache`.
- `src/custom_clustering.py` is a deprecated shim — new code should import from `src.clustering.*` directly.
- FastAPI mounts Swagger at exactly `/api` (no trailing slash) via `docs_url="/api"`. Nginx needs both `location = /api` (exact) and `location /api/` (prefix) blocks; `/openapi.json` is served at the root and needs its own `location = /openapi.json` proxy.
- `iterate_db_get_testplan` (in `src/utils/run_id_utils.py`) returns an empty `(df, df, None, None)` tuple when the QA2 config YAML or its DB lookups fail — it must NEVER call `sys.exit`. Callers gather it via `asyncio.gather(return_exceptions=True)` and treat any `BaseException` as a soft failure (regression suppressed, clustering still returned).

### Output DataFrame Columns (after full pipeline)
- `clusters` — assigned cluster name (`-1` = ungrouped)
- `cluster_class` — LLM-assigned class label
- `issue_already_occured` — `True` if matched from vector DB
- `preprocessed_reason` — normalized error text used for embeddings/SPLADE
- `extracted_error_log` — content from truncated "Limiting Reason" logs
- `rank` — 1 = most representative cluster member (cosine to centroid)
- `representativeness_score` — [0,1] cosine representativeness
- `is_core_member` — top 50% by cosine score within cluster
- `cluster_cohesion_score` — mean pairwise embedding cosine [0,1]
- `is_loose_cluster` — cohesion < 0.35 and cluster size > 5
