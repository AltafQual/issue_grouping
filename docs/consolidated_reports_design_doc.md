# Design Document: `consolidated_reports_analysis.py` — Refactoring & Architecture

**File**: `src/consolidated_reports_analysis.py`
**Current size**: 1921 lines
**Target size**: ~1150 lines across 5 modules (~40% reduction)
**Author**: Generated via Claude Code architectural analysis
**Date**: 2026-04-03

---

## Table of Contents

1. [Current State Diagnosis](#1-current-state-diagnosis)
2. [Design Patterns — Existing (As-Is)](#2-design-patterns--existing-as-is)
3. [Design Patterns — Proposed (To-Be)](#3-design-patterns--proposed-to-be)
4. [Anti-Patterns Being Removed](#4-anti-patterns-being-removed)
5. [Target Module Architecture](#5-target-module-architecture)
6. [Module Specifications](#6-module-specifications)
7. [Deduplication Map](#7-deduplication-map)
8. [File Size Estimates](#8-file-size-estimates)
9. [Implementation Order](#9-implementation-order)
10. [Verification Strategy](#10-verification-strategy)

---

## 1. Current State Diagnosis

### Class Inventory

| Class | Lines | Responsibility |
|---|---|---|
| `OrderedDefaultDict` | 293–304 | Ordered dict with default factory |
| `ConsolidatedReportAnalysis` | 307–376 | Fetches run IDs, builds prev run IDs, loads regression JSON |
| `RegressionAnalysisReport` | 379–776 | Per-run-id regression HTML report generation |
| `CombinedRegressionAnalysis` | 779–1859 | Multi-run aggregation, consolidated QAIRT report |

### Standalone Functions

| Function | Lines | Purpose |
|---|---|---|
| `filter_error_logs()` | 204–229 | Regex-based noise filtering on error lists |
| `get_cummilative_sumary()` | 232–242 | LLM summary via QGenie |
| `generate_executive_summary()` | 245–290 | HTML executive summary table |
| `_extract_id_date()` | 1862–1881 | Parse date from QAIRT ID string |
| `_months_back()` | 1884 | Date arithmetic helper |
| `should_process_id()` | 1887–1896 | Date-window filter for QAIRT IDs |
| `run_report_generation_for_all_qairt_ids()` | 1899–1914 | Entry point for batch processing |

### Critical Problem Methods

| Method | Lines | Problem |
|---|---|---|
| `generate_qairt_regression_report()` | 246 lines | Data merge + HTML + file I/O + 4 special cases all mixed |
| `generate_regression_analysis_report()` | 107 lines | Data filter + HTML + file I/O + summary generation mixed |
| `__build_bu_runtime_heatmap_html()` | 105 lines | Data aggregation + HTML rendering in one method |
| `__build_kpi_overview_html()` | 73 lines | Data aggregation + HTML rendering in one method |
| `merge_two_jsons()` | 64 lines | Complex recursive merge with 4 distinct strategies |

---

## 2. Design Patterns — Existing (As-Is)

These patterns are **already present** in the codebase, intentionally or organically.

---

### 2.1 Composite Pattern *(Structural)*

**Where**: `CombinedRegressionAnalysis` owns a collection of `RegressionAnalysisReport` objects in `self._regression_analysis_object`.

**How it works**: `CombinedRegressionAnalysis` treats each `RegressionAnalysisReport` uniformly — iterating over them to merge data, aggregate KPIs, build heatmaps, and generate the consolidated report. The composite (combined) object delegates per-run work to the leaf objects (individual reports).

```
CombinedRegressionAnalysis          ← Composite
  └── _regression_analysis_object
        ├── RegressionAnalysisReport (run_id_1)   ← Leaf
        ├── RegressionAnalysisReport (run_id_2)   ← Leaf
        └── RegressionAnalysisReport (run_id_n)   ← Leaf
```

---

### 2.2 Facade Pattern *(Structural)*

**Where**: `generate_final_summary_report()` (line 1856).

**How it works**: A single public method hides the two-step pipeline (per-run-id reports → consolidated QAIRT report) behind one call. Callers don't need to know the internal sequence.

```python
def generate_final_summary_report(self, qairt_id=None):
    self.generate_each_run_id_regression_report(qairt_id)   # step 1
    return self.generate_qairt_regression_report(qairt_id)  # step 2
```

---

### 2.3 Template Method Pattern *(Behavioural)*

**Where**: The four `__create_detailed_*_regression_page()` methods (lines 545, 588, 611, 632).

**How it works**: Each method follows the same skeleton — create output dir → build HTML table header → iterate data → write file → return server path. Only the column names, data shape, and filename differ. The skeleton is repeated (copy-pasted) rather than abstracted into a base template, which is the anti-pattern being fixed.

**Current (broken template)**:
```
__create_detailed_type_regression_page()
__create_detailed_soc_regression_page()
__create_detailed_runtime_regression_page()
__create_detailed_model_failure_regression_page()
```
All four share the same structure but duplicate it.

---

### 2.4 Strategy Pattern *(Behavioural)*

**Where**: `classify_run_id()` (line 796) and `fetch_filtered_regression_data_from_all_ids()` (line 1207).

**How it works**:
- `classify_run_id()` accepts an optional `rules` list of `(label, predicate)` pairs. Each predicate is a callable (lambda from `_regex()`). The classification strategy is swappable at call time.
- `fetch_filtered_regression_data_from_all_ids()` accepts a `key` parameter (`"soc_name"`, `"name"`, `"dsp_type"`) that changes the aggregation strategy without changing the loop structure.

---

### 2.5 Chain of Responsibility / Pipeline Pattern *(Behavioural)*

**Where**: The full report generation pipeline.

**How it works**: Data flows through a chain of transformations:
```
Raw regression JSON
  → filter_regression_data()          (filter noise types)
  → __type_runtime_based_error_data() (restructure to runtime→type)
  → merge_two_jsons()                 (merge across run IDs)
  → pivot_type_to_runtime()           (transpose axes)
  → extract_converter_quantizer_logs()(split out tools)
  → get_cummilative_sumary()          (LLM summarize)
  → HTML builder methods              (render)
  → file write                        (persist)
```
Each stage transforms the data and passes it to the next. No stage knows about the full pipeline.

---

### 2.6 Decorator Pattern *(Structural — Python decorator)*

**Where**: `@execution_timer` on `generate_executive_summary()` and `get_cummilative_sumary()`.

**How it works**: The `execution_timer` decorator wraps these functions to log execution time without modifying their logic. Classic decorator pattern — adds behaviour (timing) transparently.

---

### 2.7 Null Object Pattern *(Behavioural)*

**Where**: `OrderedDefaultDict.__missing__()` (line 300) and throughout the code where `regression_data.get("key", {})` is used.

**How it works**: Instead of raising `KeyError` or checking for `None` everywhere, missing keys return a safe default (`{}`, `[]`, `""`, `0`). This prevents null-pointer-style errors propagating through the pipeline.

---

### 2.8 Repository Pattern *(Architectural)*

**Where**: `save_regression_analysis_objects()` / `load_regression_analysis_objects()` (lines 820–861) using `joblib`.

**How it works**: The `CombinedRegressionAnalysis` class persists and retrieves `RegressionAnalysisReport` objects to/from disk (joblib files) independently of the report generation logic. This is a lightweight repository — the storage mechanism (joblib) is abstracted behind `save`/`load` methods.

---

### 2.9 Value Object / Data Transfer Object (DTO) Pattern *(Structural)*

**Where**: `RegressionAnalysisReport` accumulates state during processing:
- `self.error_summary_list`
- `self.model_regressed_errors_list`
- `self.runtime_type_regression_error_data`
- `self.regression_data`
- `self.gerrits_information`

**How it works**: After `generate_regression_analysis_report()` runs, the object acts as a DTO — a container of processed results that `CombinedRegressionAnalysis` reads from. The object is serialized (joblib) and deserialized between runs, reinforcing the DTO role.

---

### 2.10 Rule Engine Pattern *(Behavioural)*

**Where**: `classify_run_id()` (line 796) and `filter_error_logs()` (line 204).

**How it works**:
- `classify_run_id()` evaluates an ordered list of `(label, predicate)` rules, returning the first match. Rules are data (a list), not code branches.
- `filter_error_logs()` compiles a list of regex patterns and applies each as a rule to filter error strings.

Both implement a lightweight rule engine: a list of conditions evaluated in order, with a default fallback.

---

### 2.11 Lazy Initialization Pattern *(Creational)*

**Where**: `generate_each_run_id_regression_report()` (line 929) — loads cached joblib artifacts first; only processes run IDs not already in the cache.

**How it works**: Expensive computation (regression analysis per run ID) is skipped if the result already exists on disk. The object is initialized lazily from cache, and only missing entries are computed.

---

### 2.12 Recursive Descent / Visitor-like Pattern *(Behavioural)*

**Where**: `merge_two_jsons()` (line 863).

**How it works**: The method recursively walks a nested dict structure, applying merge rules at each node based on the type of value encountered (dict → recurse, list → extend, scalar → overwrite). This mirrors the Visitor pattern — the traversal logic is separated from the merge action applied at each node.

---

## 3. Design Patterns — Proposed (To-Be)

These patterns are being **introduced** by the refactoring.

---

### 3.1 Separation of Concerns / Layered Architecture *(Architectural)*

**What**: Split the monolithic file into 4 focused layers:

```
┌─────────────────────────────────────────────┐
│  consolidated_reports_analysis.py           │  ← Orchestration Layer
│  (Classes: orchestrate pipeline steps)      │
├─────────────────────────────────────────────┤
│  report_data_aggregator.py                  │  ← Data Layer
│  (Pure functions: extract, filter, merge)   │
├─────────────────────────────────────────────┤
│  report_html_builder.py                     │  ← Presentation Layer
│  (Pure functions: data → HTML string)       │
├─────────────────────────────────────────────┤
│  report_file_writer.py                      │  ← Persistence Layer
│  (Functions: write HTML files to disk)      │
├─────────────────────────────────────────────┤
│  report_constants.py                        │  ← Configuration Layer
│  (Constants, CSS, magic strings, rules)     │
└─────────────────────────────────────────────┘
```

**Why**: Each layer has one reason to change. Adding a new chart type only touches the presentation layer. Changing the file path structure only touches the persistence layer.

---

### 3.2 Template Method Pattern — Properly Applied *(Behavioural)*

**What**: The four `__create_detailed_*_regression_page()` methods collapse into one `__create_detailed_page()` using `detailed_failure_table_html()` as the template.

**Before** (4 copies of the same skeleton):
```python
def __create_detailed_type_regression_page(self, ...):
    output_dir = os.path.join(...)
    os.makedirs(output_dir, exist_ok=True)
    html = "<html><head>...</head><body><table>..."
    # type-specific row building
    with open(file_path, "w") as f: f.write(html)
    return server_prefix + file_path, count

def __create_detailed_soc_regression_page(self, ...):
    # identical skeleton, different columns/data
```

**After** (one template + data mapping):
```python
def __create_detailed_page(self, filename, title, columns, rows_data):
    html = detailed_failure_table_html(title, columns, rows_data)  # template
    path = write_detailed_page(...)                                  # persistence
    return path, len(rows_data)
```

---

### 3.3 Parameterize from Above / Strategy via Parameters *(Behavioural)*

**What**: The three failure chart methods (`__get_soc_failure_table`, `__get_model_failure_table`, `__get_dsp_type_wise_failure_table`) collapse into one `__get_failure_chart_section(key, title, color, exclude_labels, run_ids)`.

**Before**: Three methods, each hardcoding one combination of `(key, title, color, exclude_labels)`.

**After**: One method parameterized by those values. The variation is data, not code.

```python
# Three calls replace three methods:
self.__get_failure_chart_section("soc_name", "SOC Summary",      "#e74c3c", exclude_labels={"host"})
self.__get_failure_chart_section("name",     "Model Summary",    "#8e44ad")
self.__get_failure_chart_section("dsp_type", "DSP Type Summary", "#1abc9c")
```

---

### 3.4 Pure Function / Functional Core Pattern *(Architectural)*

**What**: All HTML generation functions in `report_html_builder.py` are pure — they take data in, return an HTML string out, with no side effects, no `self`, no file I/O, no LLM calls.

**Why**: Pure functions are trivially testable (assert output == expected_html), composable, and reusable. The existing `_bar_chart_html()` and `_donut_html()` are already pure but trapped as instance methods — this frees them.

```python
# Pure: same input always produces same output, no side effects
def bar_chart_html(data: list[tuple], color: str) -> str: ...
def donut_html(data: list[tuple]) -> str: ...
def rowspan_table_rows_html(label: str, rows: list, gerrit_cell: str) -> str: ...
def failure_chart_section(title, counts, failure_data, bar_color, ...) -> str: ...
```

---

### 3.5 Single Responsibility Principle — Method Decomposition *(SOLID)*

**What**: `generate_qairt_regression_report()` (246 lines) splits into three focused methods:

| Method | Responsibility | Est. Lines |
|---|---|---|
| `_merge_all_runtime_type_data()` | Merge `runtime_type_regression_error_data` across all run IDs | ~25 |
| `_build_runtime_table_html(combined_data, gerrits_data)` | Render Tools/CPU/other runtime rows with rowspan logic | ~80 |
| `generate_qairt_regression_report()` | Orchestrate: KPI + table + charts + heatmap + BU + file write | ~40 |

Similarly `generate_regression_analysis_report()` (107 lines) splits into:

| Method | Responsibility | Est. Lines |
|---|---|---|
| `_collect_regression_data()` | Filter + build type/soc/runtime dicts | ~30 |
| `_build_regression_html_body()` | Render HTML sections | ~30 |
| `generate_regression_analysis_report()` | Orchestrate + write file | ~20 |

---

### 3.6 Dependency Injection — Function Injection *(Creational/Behavioural)*

**What**: Data aggregation functions (`aggregate_kpi_metrics`, `aggregate_heatmap_data`) accept `classify_fn` as a parameter instead of calling `self.classify_run_id()` directly.

**Why**: Decouples the aggregation logic from the classification strategy. In tests, a mock `classify_fn` can be injected. In production, `self.classify_run_id` is passed. The aggregator module has no dependency on `CombinedRegressionAnalysis`.

```python
# In report_data_aggregator.py — no import of CombinedRegressionAnalysis needed
def aggregate_kpi_metrics(regression_objects: dict, classify_fn: callable) -> dict:
    for run_id, run_obj in regression_objects.items():
        bu_set.add(classify_fn(run_id))   # injected strategy
```

---

### 3.7 Constant Object / Configuration Object Pattern *(Creational)*

**What**: `report_constants.py` centralizes all magic strings, numeric thresholds, regex patterns, color codes, and CSS into named constants.

**Why**: Eliminates the "magic string" anti-pattern. When the server URL changes, one line changes. When a new error filter pattern is needed, one list is updated. When a new BU rule is added, one list entry is added — no code logic changes.

```python
# Before: scattered across 8 locations
if (failure.get("cluster_class") or "") == "sdk_issue":  # line 514
if (entry.get("cluster_class") or "") == "sdk_issue":    # line 1462
# ...6 more times

# After: one definition
SDK_ISSUE_CLASS = "sdk_issue"
if (failure.get("cluster_class") or "") == SDK_ISSUE_CLASS:
```

---

### 3.8 Extract Method Refactoring → Helper Function Pattern

**What**: The repeated rowspan-filtering logic (4 copy-pastes in `generate_qairt_regression_report`) is extracted into `build_rowspan_rows()`.

**Before** (repeated 4 times with minor variations):
```python
summary_idx_to_avoid = []
for idx, summary in enumerate(summaries_list):
    if any(s in summary.lower() for s in self.list_of_summay_to_avoid):
        summary_idx_to_avoid.append(idx)
updated_runtimes = []
for idx, runtime in enumerate(runtimes):
    if idx not in summary_idx_to_avoid:
        updated_runtimes.append((runtime, idx))
```

**After** (one function, called 4 times):
```python
def build_rowspan_rows(types_dict, summaries_fn, avoid_list) -> list[tuple]:
    ...
```

---

## 4. Anti-Patterns Being Removed

| Anti-Pattern | Where | Fix |
|---|---|---|
| **God Method** | `generate_qairt_regression_report()` (246 lines) | Split into 3 focused methods |
| **God Method** | `generate_regression_analysis_report()` (107 lines) | Split into 3 focused methods |
| **Copy-Paste Programming** | 4 `__create_detailed_*` methods | Template Method → `__create_detailed_page()` |
| **Copy-Paste Programming** | 3 failure table methods | Parameterize → `__get_failure_chart_section()` |
| **Copy-Paste Programming** | 4 rowspan blocks in `generate_qairt_regression_report` | Extract → `build_rowspan_rows()` + `rowspan_table_rows_html()` |
| **Magic Strings** | `"sdk_issue"`, `"regression_htmls"`, server URL scattered 6–8× | Centralize in `report_constants.py` |
| **Inappropriate Intimacy** | `_bar_chart_html()`, `_donut_html()` as instance methods with no `self` usage | Move to pure standalone functions |
| **Feature Envy** | `classify_run_id()` uses no instance state but lives as instance method | Move to standalone function |
| **Inline CSS / Embedded Assets** | `REPORT_CSS` (178 lines) as a Python string in the main module | Move to `report_constants.py` |
| **Mixed Abstraction Levels** | Data aggregation and HTML rendering interleaved in same methods | Separate into data layer and presentation layer |
| **Primitive Obsession** | `"soc_name"`, `"name"`, `"dsp_type"` as raw strings selecting aggregation strategy | Could become an enum in `report_constants.py` |

---

## 5. Target Module Architecture

```
src/
├── consolidated_reports_analysis.py   (~600 lines)  ← Orchestration
│     Classes: OrderedDefaultDict, ConsolidatedReportAnalysis,
│              RegressionAnalysisReport, CombinedRegressionAnalysis
│     Standalone: should_process_id, _extract_id_date, _months_back,
│                 run_report_generation_for_all_qairt_ids
│
├── report_constants.py                (~80 lines)   ← Configuration
│     REPORT_CSS, NUM_FAILURES_TO_SHOW, SERVER_PREFIX,
│     REGRESSION_HTMLS_DIR, SDK_ISSUE_CLASS, TYPES_TO_EXCLUDE,
│     CHART_COLORS, BU_RULES, ERROR_FILTER_PATTERNS, SUMMARIES_TO_AVOID
│
├── report_data_aggregator.py          (~180 lines)  ← Data Layer
│     filter_error_logs(), aggregate_failures_by_key(),
│     aggregate_kpi_metrics(), aggregate_heatmap_data(),
│     merge_runtime_type_data(), build_rowspan_rows(),
│     classify_run_id()
│
├── report_html_builder.py             (~250 lines)  ← Presentation Layer
│     bar_chart_html(), donut_html(), failure_chart_section(),
│     detailed_failure_table_html(), kpi_card_grid_html(),
│     bu_runtime_heatmap_html(), rowspan_table_rows_html(),
│     executive_summary_html(), list_to_html_ul(), log_link_html()
│
└── report_file_writer.py              (~40 lines)   ← Persistence Layer
      ensure_dir(), write_html_page(), write_detailed_page()
```

### Dependency Graph (no circular dependencies)

```
report_constants.py
       ↑
report_data_aggregator.py  ←── report_constants.py
       ↑
report_html_builder.py     ←── report_constants.py
       ↑
report_file_writer.py      (no internal deps)
       ↑
consolidated_reports_analysis.py  ←── all four modules above
```

---

## 6. Module Specifications

### 6.1 `report_constants.py`

**Pattern**: Constant Object / Configuration Object

```python
SERVER_PREFIX          = "https://aisw-hyd.qualcomm.com/fs"
REGRESSION_HTMLS_DIR   = "regression_htmls"
SDK_ISSUE_CLASS        = "sdk_issue"
TYPES_TO_EXCLUDE       = frozenset({"bm_regression", "bm_verifier"})
HOST_SOC_EXCLUDE       = "host"
BENCHMARK_TYPE_EXCLUDE = "benchmark"
TYPES_TO_PROCESS_CPU   = frozenset({"inference", "verifier"})
NUM_FAILURES_TO_SHOW   = 10

CHART_COLORS = {
    "soc":     "#e74c3c",
    "model":   "#8e44ad",
    "dsp":     "#1abc9c",
    "default": "#00629B",
}

BU_RULES = [
    (r"(?:^|[^a-z0-9])win(?:dows)?(?:[^a-z0-9]|$)", "Compute"),
    (r"(?:^|[^a-z0-9])auto(?:[^a-z0-9]|$)",          "Auto"),
    (r"(?:^|[^a-z0-9])llm(?:[^a-z0-9]|$)",           "GenAI"),
    (r"(?:^|[^a-z0-9])pt(?:[^a-z0-9]|$)",            "Mobile/IOT/XR"),
]
BU_DEFAULT = "Unknown"

ERROR_FILTER_PATTERNS = [
    r"\btimer\s+expired\b",
    r"\bmodel\s+not\s+found\b",
    # ... (13 patterns total)
]

SUMMARIES_TO_AVOID = ["no logs to provide", "no logs"]

REPORT_CSS = """<style>...(178 lines)...</style>"""
```

---

### 6.2 `report_data_aggregator.py`

**Pattern**: Pure Functions, Dependency Injection, Strategy via Parameters

| Function | Signature | Replaces |
|---|---|---|
| `filter_error_logs` | `(logs, custom=None) -> list` | Existing standalone function (move) |
| `classify_run_id` | `(run_id, rules=BU_RULES) -> str` | `CombinedRegressionAnalysis.classify_run_id()` |
| `aggregate_failures_by_key` | `(reg_objects, key, run_ids=None, apply_filter=False) -> dict` | `fetch_filtered_regression_data_from_all_ids()` |
| `aggregate_kpi_metrics` | `(reg_objects, classify_fn) -> dict` | Data loop in `__build_kpi_overview_html()` |
| `aggregate_heatmap_data` | `(reg_objects, classify_fn) -> tuple[dict, list]` | Data loop in `__build_bu_runtime_heatmap_html()` |
| `merge_runtime_type_data` | `(reg_objects, merge_fn) -> dict` | Merge loop in `generate_qairt_regression_report()` |
| `build_rowspan_rows` | `(types_dict, summaries_fn, avoid_list, exclude=None) -> list[tuple]` | 4 copy-pasted rowspan filter blocks |

---

### 6.3 `report_html_builder.py`

**Pattern**: Pure Functions, Template Method, Parameterize from Above

| Function | Signature | Replaces |
|---|---|---|
| `bar_chart_html` | `(data: list[tuple], color: str) -> str` | `_bar_chart_html()` instance method |
| `donut_html` | `(data: list[tuple]) -> str` | `_donut_html()` instance method |
| `failure_chart_section` | `(title, counts, failure_data, bar_color, top_k, avoid_list, exclude_labels=None) -> str` | All 3 of `__get_soc/model/dsp_failure_table()` |
| `detailed_failure_table_html` | `(title, columns, rows: list[dict]) -> str` | HTML part of all 4 `__create_detailed_*` methods |
| `kpi_card_grid_html` | `(metrics: dict) -> str` | HTML part of `__build_kpi_overview_html()` |
| `bu_runtime_heatmap_html` | `(heatmap_data: dict, all_runtimes: list) -> str` | HTML part of `__build_bu_runtime_heatmap_html()` |
| `rowspan_table_rows_html` | `(label, rows: list[tuple], gerrit_cell="") -> str` | 4 copy-pasted rowspan render blocks |
| `executive_summary_html` | `(soc_errors, model_errors=None) -> str` | `generate_executive_summary()` |
| `list_to_html_ul` | `(items: list) -> str` | `CombinedRegressionAnalysis.list_to_html_ul()` |
| `log_link_html` | `(log_path, server_prefix) -> str` | Inline `f"<a href=...>Log</a>"` repeated in detail pages |

---

### 6.4 `report_file_writer.py`

**Pattern**: Single Responsibility, DRY

| Function | Signature | Replaces |
|---|---|---|
| `ensure_dir` | `(path: str) -> None` | `os.makedirs()` calls scattered in 4 methods |
| `write_html_page` | `(output_dir, filename, html) -> str` | Repeated `open().write()` pattern |
| `write_detailed_page` | `(dest_folder, run_id_a, run_id_b, filename, html, server_prefix) -> str` | The `regression_htmls/{a}_{b}/` write pattern in all 4 `__create_detailed_*` methods |

---

### 6.5 `consolidated_reports_analysis.py` (Refactored)

**Pattern**: Facade, Composite, Repository, Orchestrator

#### `RegressionAnalysisReport` — key changes

```python
# BEFORE: 4 near-identical methods
def __create_detailed_type_regression_page(self, test_type, runtime, clustered_data): ...
def __create_detailed_soc_regression_page(self, soc_name, soc_regression_data): ...
def __create_detailed_runtime_regression_page(self, runtime, runtime_regression_data): ...
def __create_detailed_model_failure_regression_page(self, model_regression_data): ...

# AFTER: 1 generic method
def __create_detailed_page(self, filename, title, columns, rows_data) -> tuple[str, int]:
    html = detailed_failure_table_html(title, columns, rows_data)
    path = write_detailed_page(self.__destination_folder,
                               self.__current_run_id, self.__prev_run_id,
                               filename, html, self.__server_prefix)
    return path, len(rows_data)
```

#### `CombinedRegressionAnalysis` — key changes

```python
# BEFORE: 3 near-identical methods
def __get_soc_failure_table(self, top_k=NUM_FAILURES_TO_SHOW): ...
def __get_model_failure_table(self, top_k=NUM_FAILURES_TO_SHOW): ...
def __get_dsp_type_wise_failure_table(self, top_k=NUM_FAILURES_TO_SHOW): ...

# AFTER: 1 parameterized method
def __get_failure_chart_section(self, key, title, color,
                                 exclude_labels=None, run_ids=None) -> str:
    data   = aggregate_failures_by_key(self._regression_analysis_object,
                                       key, run_ids, apply_filter=True)
    counts = sorted([(k, len(v)) for k, v in data.items()
                     if k and k not in (exclude_labels or set())],
                    key=lambda x: -x[1])[:NUM_FAILURES_TO_SHOW]
    return failure_chart_section(title, counts, data, color,
                                 NUM_FAILURES_TO_SHOW, self.list_of_summay_to_avoid,
                                 exclude_labels)

# BEFORE: __get_bu_metrics_charts calls 3 separate methods
# AFTER: calls __get_failure_chart_section 3 times with different params
def __get_bu_metrics_charts(self, run_ids) -> str:
    return (
        self.__get_failure_chart_section("soc_name", "SOC Summary",
                                          CHART_COLORS["soc"],
                                          exclude_labels={"host"}, run_ids=run_ids) +
        self.__get_failure_chart_section("name", "Model Summary",
                                          CHART_COLORS["model"], run_ids=run_ids) +
        self.__get_failure_chart_section("dsp_type", "DSP Type Summary",
                                          CHART_COLORS["dsp"], run_ids=run_ids)
    )

# BEFORE: __build_kpi_overview_html() = 73 lines (data + HTML mixed)
# AFTER: 3 lines
def __build_kpi_overview_html(self) -> str:
    metrics = aggregate_kpi_metrics(self._regression_analysis_object, self.classify_run_id)
    metrics["unique_gerrits_count"] = self.unique_gerrits_count
    return kpi_card_grid_html(metrics)

# BEFORE: __build_bu_runtime_heatmap_html() = 105 lines (data + HTML mixed)
# AFTER: 3 lines
def __build_bu_runtime_heatmap_html(self) -> str:
    heatmap_data, all_runtimes = aggregate_heatmap_data(
        self._regression_analysis_object, self.classify_run_id)
    return bu_runtime_heatmap_html(heatmap_data, all_runtimes)

# BEFORE: generate_qairt_regression_report() = 246 lines
# AFTER: split into 3 methods
def _merge_all_runtime_type_data(self) -> dict:          # ~25 lines
def _build_runtime_table_html(self, combined_data,
                               gerrits_data) -> str:     # ~80 lines
def generate_qairt_regression_report(self, qairt_id):    # ~40 lines (orchestrator)
```

---

## 7. Deduplication Map

| Duplicated Pattern | Occurrences | Current Lines | After | Saved Lines |
|---|---|---|---|---|
| Failure chart table (SOC/Model/DSP) | 3 methods | ~105 | 1 function + 1 method | ~75 |
| Detailed HTML page creator | 4 methods | ~100 | 1 function + 1 method | ~70 |
| Rowspan filter + render block | 4 inline blocks | ~60 | 2 functions | ~45 |
| Summary-to-avoid inline check | 8+ inline `any(...)` | ~24 | 1 helper | ~18 |
| `os.makedirs` + `open().write()` | 4 locations | ~20 | 1 function | ~15 |
| Magic string `"sdk_issue"` | 8 locations | — | 1 constant | maintainability |
| Magic string `"regression_htmls"` | 6 locations | — | 1 constant | maintainability |
| Server URL prefix | 3 locations | — | 1 constant | maintainability |
| `_bar_chart_html` / `_donut_html` as instance methods | 2 methods | ~44 | 2 pure functions | 0 lines saved, testability gained |
| `classify_run_id` as instance method | 1 method | ~22 | 1 standalone function | 0 lines saved, reusability gained |
| **Total** | | **~1921 lines** | | **~770 lines saved → ~1150 total** |

---

## 8. File Size Estimates

| File | Est. Lines | Primary Pattern(s) |
|---|---|---|
| `report_constants.py` | ~80 | Constant Object |
| `report_data_aggregator.py` | ~180 | Pure Functions, Strategy, DI |
| `report_html_builder.py` | ~250 | Pure Functions, Template Method |
| `report_file_writer.py` | ~40 | Single Responsibility |
| `consolidated_reports_analysis.py` | ~600 | Facade, Composite, Repository, Orchestrator |
| **Total** | **~1150** | **vs 1921 today (~40% reduction)** |

---

## 9. Implementation Order

Each step is independently verifiable — run report generation after each step and diff the output HTML.

| Step | Action | Risk |
|---|---|---|
| 1 | Create `report_constants.py`, move all constants. Update imports in main file. | Low — no logic change |
| 2 | Move `_bar_chart_html`, `_donut_html`, `list_to_html_ul`, `generate_executive_summary` to `report_html_builder.py` as pure functions. | Low — already pure |
| 3 | Create `report_file_writer.py`. Extract `write_detailed_page` from the 4 `__create_detailed_*` methods. | Low |
| 4 | Collapse 4 `__create_detailed_*` methods → `__create_detailed_page()` using `detailed_failure_table_html()`. | Medium — verify all 4 detail page types |
| 5 | Create `report_data_aggregator.py`. Move `filter_error_logs`, extract `aggregate_failures_by_key` from `fetch_filtered_regression_data_from_all_ids`. | Low |
| 6 | Extract `aggregate_kpi_metrics` from `__build_kpi_overview_html()`. Extract `aggregate_heatmap_data` from `__build_bu_runtime_heatmap_html()`. | Medium — verify KPI counts and heatmap values |
| 7 | Collapse 3 failure table methods → `__get_failure_chart_section()`. | Medium — verify SOC/Model/DSP charts |
| 8 | Extract `build_rowspan_rows` + `rowspan_table_rows_html`. Collapse 4 rowspan blocks. | Medium — verify runtime table rows |
| 9 | Split `generate_qairt_regression_report` into `_merge_all_runtime_type_data` + `_build_runtime_table_html` + thin orchestrator. | High — largest method, most logic |
| 10 | Split `generate_regression_analysis_report` similarly. | Medium |
| 11 | Move `classify_run_id` to standalone function in `report_data_aggregator.py`. | Low |

---

## 10. Verification Strategy

### Per-Step Verification
After each implementation step, run:
```bash
python -c "
from src.consolidated_reports_analysis import CombinedRegressionAnalysis, ConsolidatedReportAnalysis
cra = ConsolidatedReportAnalysis()
combined = CombinedRegressionAnalysis(cra)
combined.generate_final_summary_report('<test_qairt_id>')
"
```
Diff the output HTML against the pre-refactor baseline.

### Unit Test Targets (new, enabled by pure functions)
```python
# report_html_builder.py — trivially testable
def test_bar_chart_html_empty():
    assert bar_chart_html([], "#red") == "<p><i>No data</i></p>"

def test_bar_chart_html_single():
    html = bar_chart_html([("soc_a", 10)], "#e74c3c")
    assert "soc_a" in html and "100%" in html

def test_rowspan_table_rows_html_single_row():
    html = rowspan_table_rows_html("CPU", [("inference", "<ul>summary</ul>")], "<td>gerrit</td>")
    assert "rowspan='1'" in html and "CPU" in html

# report_data_aggregator.py — testable with mock data
def test_classify_run_id_compute():
    assert classify_run_id("qaisw-win-v2.44.0_nightly") == "Compute"

def test_aggregate_failures_by_key_filters_host():
    # inject mock regression_objects
    result = aggregate_failures_by_key(mock_objects, key="soc_name")
    assert "host" not in result  # host filtered by caller
```

### Regression Test
The final HTML output of `generate_final_summary_report()` must be byte-for-byte identical (or semantically equivalent) to the pre-refactor output for the same input data.

---

*End of Design Document*
