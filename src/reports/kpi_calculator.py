"""KPI calculator — extracted from ConsolidatedReportAnalysis.

Provides :class:`KPICalculator` along with the module-level utility
functions :func:`filter_error_logs`, :func:`get_cummilative_sumary`, and
:func:`generate_executive_summary` that were previously defined in
``src.consolidated_reports_analysis``.

Layering
--------
Imports from ``src.reports.html_renderer``, ``src.utils.timer``,
and ``src.logger``.  Does **not** import from ``src.consolidated_reports_analysis``
to avoid circular dependencies.
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from src.llm.client import cummilative_summary_generation, error_summary_generation
from src.logger import AppLogger
from src.reports.html_renderer import HTMLRenderer
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "KPICalculator",
    "OrderedDefaultDict",
    "filter_error_logs",
    "generate_executive_summary",
    "get_cummilative_sumary",
    "error_summary_generation",
    "cummilative_summary_generation",
]

# ---------------------------------------------------------------------------
# Shared utility types
# ---------------------------------------------------------------------------

NUM_FAILURES_TO_SHOW = 10


class OrderedDefaultDict(OrderedDict):
    """An :class:`~collections.OrderedDict` that behaves like :class:`~collections.defaultdict`.

    Args:
        default_factory: Callable invoked with no arguments to supply a
            default value for missing keys.  Mirrors the ``defaultdict``
            constructor signature.

    Example::

        d = OrderedDefaultDict(list)
        d["a"].append(1)
    """

    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable or None")
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value


# ---------------------------------------------------------------------------
# Module-level pure functions (moved from consolidated_reports_analysis.py)
# ---------------------------------------------------------------------------


def filter_error_logs(error_logs_list: list[str], custom_filter_list: list[str] | None = None) -> list[str]:
    """Filter out infrastructure / environment errors from a list of error strings.

    Removes entries that match known noise patterns (timer expiry, device not
    found, missing shared libraries, etc.) using compiled regular expressions.

    Args:
        error_logs_list: Raw error log strings to filter.
        custom_filter_list: Optional additional regex patterns to suppress.

    Returns:
        Filtered list containing only SDK-issue-like error strings.
    """
    filter_list = [
        r"\btimer\s+expired\b",
        r"\bmodel\s+not\s+found\b",
        r"\bdevice\s+creation\b",
        r"\binference\s+passed\s+but\s+output\s+not\s+generated\b",
        r"\bverifier\s+failed\b",
        r"\bdevice\s+not\s+found\b",
        r"\bmissing\s+shared\s+libraries\b",
        r"\bdevice(?:[_\s-])?unavailable\b",
        r"\bnot\s+found\b",
        r"\bno\s+space\s+left\s+on\s+device\b",
        r"\bno\s+such\s+file\s+or\s+directory",
        r"\bfilenotfounderror\b",
        r"\bqnn-net-run.exe\b",
    ]
    if custom_filter_list:
        filter_list.extend(custom_filter_list)
    _error_filters = tuple(re.compile(p, re.IGNORECASE) for p in filter_list)
    filtered = [log for log in error_logs_list if log and not any(p.search(log) for p in _error_filters)]
    logger.info(f"Error logs in total: {len(error_logs_list)}, filtered error logs: {len(filtered)}")
    return filtered


@execution_timer
def get_cummilative_sumary(
    errors: list[str],
    filter: bool = True,
    custom_filter: list[str] | None = None,
    short_summary: bool = False,
) -> str:
    """Generate a cumulative LLM summary for a list of error strings.

    Args:
        errors: Error log strings to summarise.
        filter: When ``True``, passes *errors* through :func:`filter_error_logs`
            before calling the LLM.
        custom_filter: Additional regex patterns forwarded to :func:`filter_error_logs`.
        short_summary: When ``True``, requests a shorter summary from the LLM.

    Returns:
        HTML summary string produced by the LLM, or an empty string on failure.
    """
    if filter:
        logger.info("Filter enabled, filtering logs")
        return cummilative_summary_generation(
            filter_error_logs(errors, custom_filter), short_final_summary=short_summary
        )
    return cummilative_summary_generation(errors, short_final_summary=short_summary)


@execution_timer
def generate_executive_summary(
    soc_errors_list: list[str] | None = None,
    model_error_list: list[str] | None = None,
    filter: bool = True,
) -> str:
    """Generate an executive summary HTML table from SOC and/or model error lists.

    Args:
        soc_errors_list: SOC/runtime error strings used for the first column.
        model_error_list: Model error strings used for the second column (optional).
        filter: Passed to :func:`get_cummilative_sumary` to enable pre-filtering.

    Returns:
        HTML ``<table class="exec-summary">`` string with one or two columns.
    """

    def ensure_td(summary_html: str) -> str:
        if summary_html and "<td" in summary_html.lower():
            return summary_html
        return f"<td>{summary_html or '-'}</td>"

    inline_css = """
    <style>
    table.exec-summary {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    table.exec-summary th {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #ccc;
        font-weight: normal;
    }
    table.exec-summary td {
        text-align: left;
        vertical-align: middle;
        padding: 12px;
    }
    table.exec-summary td li {
        margin-bottom: 2px;
        list-style: disc;
        margin-left: 20px;
    }
    </style>
    """

    if model_error_list:
        executive_summary = (
            inline_css + "<table class='exec-summary'><tr><th>SOC/RunTime Summary</th><th>Model Summary</th></tr>"
        )
        soc_runtime_summary = get_cummilative_sumary(soc_errors_list, filter)
        model_summary = get_cummilative_sumary(model_error_list, filter)
        executive_summary += "<tr>" f"{ensure_td(soc_runtime_summary)}" f"{ensure_td(model_summary)}" "</tr></table>"
    else:
        executive_summary = inline_css + "<table class='exec-summary'><tr><th>SOC/RunTime Summary</th></tr>"
        soc_runtime_summary = get_cummilative_sumary(soc_errors_list, filter)
        executive_summary += "<tr>" f"{ensure_td(soc_runtime_summary)}" "</tr></table>"

    return executive_summary


# ---------------------------------------------------------------------------
# KPICalculator
# ---------------------------------------------------------------------------


class KPICalculator:
    """Computes KPI metrics and builds HTML sections from clustered failure data.

    Extracted from :class:`~src.consolidated_reports_analysis.CombinedRegressionAnalysis`
    to separate KPI computation from orchestration logic.

    Args:
        regression_analysis_object: Mapping of ``run_id → RegressionAnalysisReport``
            objects, as maintained by ``CombinedRegressionAnalysis``.
        classify_run_id_fn: Callable that maps a run-ID string to a BU label
            (e.g. ``"Auto"``, ``"Compute"``).
        unique_gerrits_count: Total number of unique Gerrit commits merged across
            all run IDs (used in the KPI overview card).
        summaries_to_avoid: Error summary strings to suppress in output tables.
            Defaults to ``["no logs to provide", "no logs"]``.

    Example::

        kpi = KPICalculator(
            regression_analysis_object=combined.regression_analysis_object,
            classify_run_id_fn=combined.classify_run_id,
            unique_gerrits_count=combined.unique_gerrits_count,
        )
        html = kpi.build_soc_failure_table()
    """

    def __init__(
        self,
        regression_analysis_object: dict[str, Any],
        classify_run_id_fn: Callable[[str], str],
        unique_gerrits_count: int = 0,
        summaries_to_avoid: list[str] | None = None,
    ) -> None:
        self._regression_analysis_object = regression_analysis_object
        self._classify_run_id = classify_run_id_fn
        self._unique_gerrits_count = unique_gerrits_count
        self._summaries_to_avoid = summaries_to_avoid or ["no logs to provide", "no logs"]

    # ------------------------------------------------------------------
    # Data aggregation
    # ------------------------------------------------------------------

    def get_filtered_data(
        self,
        key: str = "soc_name",
        apply_filter: bool = False,
        run_ids: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Aggregate and optionally filter failure data from all run IDs.

        Iterates over regression analysis objects, collects per-failure entries
        keyed by *key* (e.g. ``"soc_name"``, ``"name"``, ``"dsp_type"``), and
        applies :func:`filter_error_logs` when *apply_filter* is ``True``.

        Args:
            key: DataFrame column name to group failures by.
            apply_filter: When ``True``, pass each error list through
                :func:`filter_error_logs` to strip infrastructure noise.
            run_ids: Optional subset of run IDs to process.  Defaults to all IDs
                in ``regression_analysis_object``.

        Returns:
            Mapping of ``{key_value: [error_reason, ...]}`` sorted by descending
            error count.
        """
        results: OrderedDefaultDict = OrderedDefaultDict(list)
        ids_to_process = run_ids if run_ids is not None else list(self._regression_analysis_object.keys())
        for run_id in ids_to_process:
            run_obj = self._regression_analysis_object[run_id]
            if not run_obj or not run_obj.regression_data:
                continue
            model_failure_data = run_obj.regression_data["model"]
            logger.info(f"Processing: {run_id}: total model failure: {len(model_failure_data)}")
            for _, failures_list in model_failure_data.items():
                for failure in failures_list:
                    if (
                        key in failure
                        and failure[key]
                        and "reason" in failure
                        and failure["reason"]
                        and (failure.get("cluster_class") or "") == "sdk_issue"
                        and (failure.get("type") or "").lower() not in run_obj.types_to_filter_for_regression_analysis
                    ):
                        results[failure[key].lower()].append(failure["reason"])

        logger.info(f"Total filtered errors: {len(results)}")
        if apply_filter:
            updated: OrderedDefaultDict = OrderedDefaultDict(list)
            for k, v in results.items():
                updated[k] = filter_error_logs(v)
        else:
            updated = results

        return dict(sorted(updated.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    # ------------------------------------------------------------------
    # HTML section builders
    # ------------------------------------------------------------------

    def build_soc_failure_table(self, top_k: int = NUM_FAILURES_TO_SHOW) -> str:
        """Build an HTML SOC failure summary with bar chart, donut, and table.

        Args:
            top_k: Maximum number of SOCs to include in charts and table.

        Returns:
            HTML string containing the SOC summary section.
        """
        logger.info("Building Soc Failure Table")
        failure_data = self.get_filtered_data(apply_filter=True)
        soc_counts = sorted([(k, len(v)) for k, v in failure_data.items() if k and k != "host"], key=lambda x: -x[1])[
            :top_k
        ]
        html = '<div class="chart-wrap"><h3>SOC Summary</h3>'
        html += '<div class="chart-grid-2col">'
        html += (
            f'<div class="chart-box"><h4>Failure Count by SOC (Top {top_k})</h4>'
            + HTMLRenderer.bar_chart_html(soc_counts, color="#e74c3c")
            + "</div>"
        )
        html += (
            f'<div class="chart-box"><h4>SOC Distribution (Top {top_k})</h4>'
            + HTMLRenderer.donut_html(soc_counts)
            + "</div>"
        )
        html += "</div>"
        html += "<table border='1'><tr><th>Soc Name</th><th>Summary</th></tr>"

        idx_count = 0
        for soc_name, errors_list in failure_data.items():
            if soc_name == "host":
                continue
            summary = get_cummilative_sumary(errors_list, filter=False, short_summary=True)
            if any(s in summary.lower() for s in self._summaries_to_avoid):
                continue
            idx_count += 1
            html += f"<tr><td>{soc_name}</td><td><ul>{summary}</ul></td></tr>"
            if idx_count >= top_k:
                break

        if idx_count == 0:
            html += "<tr><td colspan='2'><i>No failures found</i></td></tr>"
        html += "</table></div>"
        return html

    def build_model_failure_table(self, top_k: int = NUM_FAILURES_TO_SHOW) -> str:
        """Build an HTML model failure summary with bar chart, donut, and table.

        Args:
            top_k: Maximum number of models to include in charts and table.

        Returns:
            HTML string containing the model summary section.
        """
        logger.info("Building model failure table")
        failure_data = self.get_filtered_data(key="name", apply_filter=True)
        model_counts = sorted([(k, len(v)) for k, v in failure_data.items() if k], key=lambda x: -x[1])[:top_k]
        html = '<div class="chart-wrap"><h3>Model Summary</h3>'
        html += '<div class="chart-grid-2col">'
        html += (
            f'<div class="chart-box"><h4>Failure Count by Model (Top {top_k})</h4>'
            + HTMLRenderer.bar_chart_html(model_counts, color="#8e44ad")
            + "</div>"
        )
        html += (
            f'<div class="chart-box"><h4>Model Distribution (Top {top_k})</h4>'
            + HTMLRenderer.donut_html(model_counts)
            + "</div>"
        )
        html += "</div>"
        html += "<table border='1'><tr><th>Model Name</th><th>Summary</th></tr>"

        idx_count = 0
        for model_name, errors_list in failure_data.items():
            summary = get_cummilative_sumary(errors_list, filter=False, short_summary=True)
            if any(s in summary.lower() for s in self._summaries_to_avoid):
                continue
            idx_count += 1
            html += f"<tr><td>{model_name}</td><td><ul>{summary}</ul></td></tr>"
            if idx_count >= top_k:
                break

        if idx_count == 0:
            html += "<tr><td colspan='2'><i>No failures found</i></td></tr>"
        html += "</table></div>"
        return html

    def build_dsp_failure_table(self, top_k: int = NUM_FAILURES_TO_SHOW) -> str:
        """Build an HTML DSP type failure summary with bar chart, donut, and table.

        Args:
            top_k: Maximum number of DSP types to include in charts and table.

        Returns:
            HTML string containing the DSP type summary section.
        """
        logger.info("Building DSP type failure table")
        failure_data = self.get_filtered_data(key="dsp_type", apply_filter=True)
        dsp_counts = sorted([(k, len(v)) for k, v in failure_data.items() if k], key=lambda x: -x[1])
        html = '<div class="chart-wrap"><h3>DSP Type Summary</h3>'
        html += '<div class="chart-grid-2col">'
        html += (
            '<div class="chart-box"><h4>Failure Count by DSP Type</h4>'
            + HTMLRenderer.bar_chart_html(dsp_counts, color="#1abc9c")
            + "</div>"
        )
        html += '<div class="chart-box"><h4>DSP Type Distribution</h4>' + HTMLRenderer.donut_html(dsp_counts) + "</div>"
        html += "</div>"
        html += "<table border='1'><tr><th>DSP Name</th><th>Summary</th></tr>"

        idx_count = 0
        for dsp_name, errors_list in failure_data.items():
            summary = get_cummilative_sumary(errors_list, filter=False, short_summary=True)
            if any(s in summary.lower() for s in self._summaries_to_avoid):
                continue
            idx_count += 1
            html += f"<tr><td>{dsp_name}</td><td><ul>{summary}</ul></td></tr>"
            if idx_count >= top_k:
                break

        if idx_count == 0:
            html += "<tr><td colspan='2'><i>No failures found</i></td></tr>"
        html += "</table></div>"
        return html

    def build_bu_metrics_charts(self, run_ids: list[str], top_k: int = NUM_FAILURES_TO_SHOW) -> str:
        """Build SOC, model, and DSP failure charts for a subset of run IDs.

        Used in the BU-wise analysis report to show per-BU failure breakdowns.

        Args:
            run_ids: List of run IDs belonging to a single BU.
            top_k: Maximum number of items per chart.

        Returns:
            HTML string containing the three chart sections (SOC, model, DSP).
        """
        html = ""

        soc_data = self.get_filtered_data(key="soc_name", apply_filter=True, run_ids=run_ids)
        soc_counts = sorted([(k, len(v)) for k, v in soc_data.items() if k and k != "host"], key=lambda x: -x[1])[
            :top_k
        ]
        if soc_counts:
            html += '<div class="chart-wrap"><h4>SOC Summary</h4><div class="chart-grid-2col">'
            html += (
                f'<div class="chart-box"><h4>Failure Count by SOC (Top {top_k})</h4>'
                + HTMLRenderer.bar_chart_html(soc_counts, color="#e74c3c")
                + "</div>"
            )
            html += (
                f'<div class="chart-box"><h4>SOC Distribution (Top {top_k})</h4>'
                + HTMLRenderer.donut_html(soc_counts)
                + "</div>"
            )
            html += "</div></div>"

        model_data = self.get_filtered_data(key="name", apply_filter=True, run_ids=run_ids)
        model_counts = sorted([(k, len(v)) for k, v in model_data.items() if k], key=lambda x: -x[1])[:top_k]
        if model_counts:
            html += '<div class="chart-wrap"><h4>Model Summary</h4><div class="chart-grid-2col">'
            html += (
                f'<div class="chart-box"><h4>Failure Count by Model (Top {top_k})</h4>'
                + HTMLRenderer.bar_chart_html(model_counts, color="#8e44ad")
                + "</div>"
            )
            html += (
                f'<div class="chart-box"><h4>Model Distribution (Top {top_k})</h4>'
                + HTMLRenderer.donut_html(model_counts)
                + "</div>"
            )
            html += "</div></div>"

        dsp_data = self.get_filtered_data(key="dsp_type", apply_filter=True, run_ids=run_ids)
        dsp_counts = sorted([(k, len(v)) for k, v in dsp_data.items() if k], key=lambda x: -x[1])
        if dsp_counts:
            html += '<div class="chart-wrap"><h4>DSP Type Summary</h4><div class="chart-grid-2col">'
            html += (
                '<div class="chart-box"><h4>Failure Count by DSP Type</h4>'
                + HTMLRenderer.bar_chart_html(dsp_counts, color="#1abc9c")
                + "</div>"
            )
            html += (
                '<div class="chart-box"><h4>DSP Type Distribution</h4>' + HTMLRenderer.donut_html(dsp_counts) + "</div>"
            )
            html += "</div></div>"

        return html

    def build_kpi_overview(self) -> str:
        """Build the Executive Overview KPI card grid HTML.

        Aggregates total failures, models failed, run IDs processed, BUs, and
        Gerrit count from all processed run IDs.

        Returns:
            HTML string containing the ``.overview-section`` KPI grid.
        """
        logger.info("Building KPI dashboard Card")

        total_failures = 0
        total_models_failed = 0
        total_model_failure_entries = 0
        soc_set: set[str] = set()
        runtime_set: set[str] = set()
        bu_set: set[str] = set()
        cluster_name_set: set[str] = set()

        for run_id, run_obj in self._regression_analysis_object.items():
            if not run_obj:
                continue
            rd = run_obj.regression_data
            if not rd or rd.get("status") != 200:
                continue

            bu_set.add(self._classify_run_id(run_id))

            for _type, runtimes in rd.get("type", {}).items():
                if _type.lower() in run_obj.types_to_filter_for_regression_analysis:
                    continue
                if not isinstance(runtimes, dict):
                    continue
                for rt, clusters in runtimes.items():
                    if not isinstance(clusters, dict):
                        continue
                    runtime_set.add(rt)
                    for cluster_name, entries in clusters.items():
                        if isinstance(entries, list):
                            sdk_entries = [e for e in entries if (e.get("cluster_class") or "") == "sdk_issue"]
                            total_failures += len(sdk_entries)
                            cluster_name_set.add(cluster_name.lower())
                            for entry in sdk_entries:
                                if isinstance(entry, dict) and entry.get("soc_name"):
                                    soc_set.add(entry["soc_name"])

            for _, entries in rd.get("model", {}).items():
                sdk_entries = [
                    e
                    for e in entries
                    if (e.get("cluster_class") or "") == "sdk_issue"
                    and (e.get("type") or "").lower() not in run_obj.types_to_filter_for_regression_analysis
                ]
                if sdk_entries:
                    total_models_failed += 1
                    total_model_failure_entries += len(sdk_entries)
                    for entry in sdk_entries:
                        if isinstance(entry, dict):
                            if entry.get("soc_name"):
                                soc_set.add(entry["soc_name"])
                            if entry.get("runtime"):
                                runtime_set.add(entry["runtime"])

        total_run_ids = len(self._regression_analysis_object)

        html = '<div class="overview-section">'
        html += '<div class="section-title">&#128202; Executive Overview</div>'
        html += '<div class="kpi-grid">'
        html += f'<div class="kpi d"><div class="lbl">Total Failures</div><div class="val">{total_failures:,}</div><div class="sub">Across all run IDs</div></div>'
        html += f'<div class="kpi w"><div class="lbl">Models Failed</div><div class="val">{total_models_failed:,}</div><div class="sub">{total_model_failure_entries:,} failure entries</div></div>'
        html += f'<div class="kpi"><div class="lbl">Run IDs Processed</div><div class="val">{total_run_ids}</div><div class="sub">{len(bu_set)} Business Units</div></div>'
        html += f'<div class="kpi s"><div class="lbl">Gerrits Merged</div><div class="val">{self._unique_gerrits_count}</div><div class="sub">Unique commits</div></div>'
        html += "</div></div>"
        return html

    def build_bu_runtime_heatmap(self) -> str:
        """Build a BU × Runtime failure heatmap HTML table.

        Rows are Business Units; columns are runtime targets (cpu, gpu, htp,
        etc.).  Cell values are failure counts; colour intensity encodes severity.

        Returns:
            HTML string containing the ``.heatmap-wrap`` section, or ``""``
            when no active BU data exists.
        """
        logger.info("Building BU runtime heatmap")
        _BU_ORDER = ["Auto", "Compute", "Mobile/IOT/XR", "GenAI", "Unknown"]
        _BU_COLORS = {
            "Auto": "#e74c3c",
            "Compute": "#2980b9",
            "Mobile/IOT/XR": "#27ae60",
            "GenAI": "#8e44ad",
            "Unknown": "#95a5a6",
        }
        _ALL_RUNTIMES = ["cpu", "gpu", "gpu_fp16", "htp", "htp_fp16", "mcp", "mcp_x86", "lpai"]

        def _heat_class(val: int, max_val: int) -> str:
            if val == 0:
                return "heat-0"
            r = val / max_val
            if r < 0.10:
                return "heat-1"
            if r < 0.25:
                return "heat-2"
            if r < 0.50:
                return "heat-3"
            if r < 0.75:
                return "heat-4"
            return "heat-5"

        heatmap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for run_id, run_obj in self._regression_analysis_object.items():
            if not run_obj:
                continue
            rd = run_obj.regression_data
            if not rd or rd.get("status") != 200:
                continue
            bu = self._classify_run_id(run_id)
            for _type, runtimes in rd.get("type", {}).items():
                if _type.lower() in run_obj.types_to_filter_for_regression_analysis:
                    continue
                if not isinstance(runtimes, dict):
                    continue
                for rt, clusters in runtimes.items():
                    if not isinstance(clusters, dict):
                        continue
                    cnt = sum(
                        len([e for e in errors_list if (e.get("cluster_class") or "") == "sdk_issue"])
                        for errors_list in clusters.values()
                    )
                    heatmap[bu][rt] += cnt

        active_bus = [bu for bu in _BU_ORDER if bu in heatmap]
        if not active_bus:
            return ""

        all_vals = [heatmap[bu][rt] for bu in active_bus for rt in _ALL_RUNTIMES]
        max_heat = max(all_vals) if all_vals else 1

        html = '<div class="heatmap-wrap">'
        html += "<h3>Failure Heatmap: BU &times; Runtime</h3>"
        html += '<p style="color:var(--text-muted);font-size:.86em;margin-top:-8px;margin-bottom:12px">Cell values = total failure count. Color intensity = severity.</p>'
        html += "<table><tr><th>BU / Runtime</th>"
        for rt in _ALL_RUNTIMES:
            html += f"<th>{rt.upper()}</th>"
        html += "<th>TOTAL</th></tr>"

        for bu in active_bus:
            row_total = sum(heatmap[bu][rt] for rt in _ALL_RUNTIMES)
            color = _BU_COLORS.get(bu, "#95a5a6")
            html += f'<tr><td style="font-weight:700;background:{color};color:#fff;text-align:left;padding:8px 12px">{bu}</td>'
            for rt in _ALL_RUNTIMES:
                val = heatmap[bu][rt]
                html += f'<td class="{_heat_class(val, max_heat)}">{val:,}</td>'
            html += f'<td style="font-weight:800;background:#f8f9fa">{row_total:,}</td></tr>'

        html += '<tr style="background:#f0f2f5"><td style="font-weight:700;text-align:left;padding:8px 12px">TOTAL</td>'
        grand_total = 0
        for rt in _ALL_RUNTIMES:
            ct = sum(heatmap[bu][rt] for bu in active_bus)
            grand_total += ct
            html += f'<td style="font-weight:700">{ct:,}</td>'
        html += f'<td style="font-weight:800">{grand_total:,}</td></tr>'
        html += "</table>"

        html += '<div class="heat-legend">Intensity:'
        for cls, lbl in [
            ("heat-0", "0"),
            ("heat-1", "Low"),
            ("heat-2", "Medium"),
            ("heat-3", "High"),
            ("heat-4", "Very High"),
            ("heat-5", "Critical"),
        ]:
            html += f'<span class="{cls}">{lbl}</span>'
        html += "</div>"
        html += "</div>"
        return html


# ---------------------------------------------------------------------------
# LLM-based summary generation
# ---------------------------------------------------------------------------
# `error_summary_generation` and `cummilative_summary_generation` are imported
# from `src.llm.client` (the canonical implementations).  They are re-exported
# below for backward compatibility with existing callers of this module.
