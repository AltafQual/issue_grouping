"""Stability monitor — per-type failure analysis and HTML report generation.

Provides :class:`StabilityMonitor` as the high-level class interface, plus
the dataclasses and functions that make up the core stability analysis logic.

Public API
----------
    StabilityMonitor                     — class entry point
    RunAnalysis                          — dataclass for per-run analysis
    TypeStats                            — dataclass for per-type statistics
    analyze_type_failures(df)            — functional API
    build_combined_stability_html(runs)  — HTML report builder

Layering
--------
This module imports from ``src.constants``, ``src.core.interfaces``,
and ``src.logger``.  No imports from ``src.hourly_report``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.constants import StabilityReportConfig
from src.core.interfaces import INotifier
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = [
    "StabilityMonitor",
    "RunAnalysis",
    "TypeStats",
    "analyze_type_failures",
    "build_combined_stability_html",
]

_PRIMARY_TYPES: frozenset[str] = frozenset(
    {"converter", "quantizer", "savecontext","unit_test_host"}
)
_HOST_COLUMN_CANDIDATES: tuple[str, ...] = ("host", "host_name", "node_name", "node", "machine")


_CSS = """
/* ── Reset / Layout ─────────────────────────────────────────────────────── */
* { box-sizing: border-box; }
body {
    font-family: Arial, Helvetica, sans-serif;
    max-width: 1300px;
    margin: 0 auto;
    padding: 24px;
    color: #212529;
    background: #ffffff;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {
    font-size: 1.5rem;
    color: #0d6efd;
    border-bottom: 3px solid #0d6efd;
    padding-bottom: 8px;
    margin-bottom: 4px;
}
h2 { font-size: 1.15rem; color: #343a40; margin: 24px 0 6px; }
p.subtitle { color: #6c757d; font-size: 13px; margin: 2px 0 12px; }
ul.run-id-list { margin: 4px 0 14px 20px; padding: 0; color: #343a40; font-size: 13px; list-style: disc; }
ul.run-id-list li { margin: 2px 0; }

/* ── Meta / key-value table ─────────────────────────────────────────────── */
table.meta          { border-collapse: collapse; font-size: 13px; margin-bottom: 16px; }
table.meta td       { padding: 4px 16px 4px 0; vertical-align: top; }
table.meta td.label { color: #6c757d; font-weight: bold; white-space: nowrap; min-width: 130px; }

/* ── Data tables (overview + per-type summary) ───────────────────────────── */
table.data           { border-collapse: collapse; width: 100%; margin-bottom: 6px; }
table.data th        {
    padding: 8px 10px;
    background: #343a40;
    color: #ffffff;
    font-size: 12px;
    text-align: center;
    white-space: nowrap;
}
table.data th.left   { text-align: left; }
table.data td        { padding: 6px 10px; font-size: 12px; border: 1px solid #dee2e6; vertical-align: top; }
table.data td.center { text-align: center; vertical-align: middle; }
table.data tr.flagged { background: #fff3cd; }

/* ── Host breakdown grid (3 SoC cards per row) ───────────────────────────── */
.host-grid  { display: flex; flex-wrap: wrap; gap: 10px; }
.host-card  { flex: 0 0 calc(33.33% - 7px); min-width: 130px; }
.soc-label  {
    font-weight: bold; font-size: 11px; color: #343a40;
    border-bottom: 1px solid #dee2e6; margin-bottom: 3px; padding-bottom: 2px;
}
.host-row   { font-size: 11px; color: #495057; padding-left: 4px; line-height: 1.65; }

/* ── Utility classes ─────────────────────────────────────────────────────── */
.badge-high {
    display: inline-block;
    background: #dc3545;
    color: #ffffff;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: bold;
    margin-left: 5px;
    vertical-align: middle;
}
.text-danger  { color: #dc3545; font-weight: bold; }
.text-success { color: #228B22; }
.text-muted   { color: #6c757d; }
.text-primary { color: #0d6efd; text-decoration: none; }
.text-primary:hover { text-decoration: underline; }

/* ── Section divider ─────────────────────────────────────────────────────── */
hr.section { border: none; border-top: 2px solid #dee2e6; margin: 28px 0 18px; }
"""


def _is_primary_type(test_type: str) -> bool:
    return test_type.lower().replace("_", "").replace(" ", "") in _PRIMARY_TYPES


def _detect_host_column(df: pd.DataFrame) -> str | None:
    """Return the first host-like column found in *df*, or None."""
    for col in _HOST_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


@dataclass
class TypeStats:
    """Per-test-type failure statistics for a single run."""

    test_type: str
    total: int
    passed: int
    failed: int
    parent_fail: int
    not_run: int
    failure_rate: float
    highlighted: bool
    host_failures: dict[str, dict[str, tuple[int, int, str]]] = field(default_factory=dict)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def effective_total(self) -> int:
        return self.total - self.parent_fail - self.not_run

    @property
    def failure_pct(self) -> float:
        return self.failure_rate * 100


@dataclass
class RunAnalysis:
    """Full analysis for a single run ID."""

    run_id: str
    job_info: dict[str, Any]
    type_stats: dict[str, TypeStats]

    @property
    def total_tests(self) -> int:
        return sum(s.total for s in self.type_stats.values())

    @property
    def total_failed(self) -> int:
        return sum(s.failed for s in self.type_stats.values())

    @property
    def total_parent_fail(self) -> int:
        return sum(s.parent_fail for s in self.type_stats.values())

    @property
    def total_not_run(self) -> int:
        return sum(s.not_run for s in self.type_stats.values())

    @property
    def total_passed(self) -> int:
        return sum(s.passed for s in self.type_stats.values())

    @property
    def effective_total(self) -> int:
        return self.total_tests - self.total_parent_fail - self.total_not_run

    @property
    def overall_failure_pct(self) -> float:
        return (self.total_failed / self.effective_total * 100) if self.effective_total else 0.0

    @property
    def flagged_types(self) -> list[str]:
        return [t for t, s in self.type_stats.items() if s.highlighted]

    @property
    def has_flags(self) -> bool:
        return bool(self.flagged_types)


def analyze_type_failures(
    df: pd.DataFrame,
    threshold: float = StabilityReportConfig.FAILURE_THRESHOLD,
    detail_row_cap: int = StabilityReportConfig.DETAIL_ROW_CAP,
) -> dict[str, TypeStats]:
    """Group *df* by the ``type`` column and compute failure statistics per type.

    Args:
        df: Must contain ``type`` and ``result`` columns.
        threshold: Failure rate at or above which a type is flagged.
        detail_row_cap: Max FAIL rows stored per flagged type.

    Returns:
        Mapping of test-type name → :class:`TypeStats`.
    """
    result_upper: pd.Series = df["result"].astype(str).str.strip().str.upper()
    host_col: str | None = _detect_host_column(df)
    stats: dict[str, TypeStats] = {}

    for test_type, type_df in df.groupby("type"):
        type_str = str(test_type)
        type_res: pd.Series = result_upper.loc[type_df.index]

        total = len(type_df)
        passed = int((type_res == "PASS").sum())
        failed = int((type_res == "FAIL").sum())
        parent_fail = int((type_res == "PARENT_FAIL").sum())
        not_run = int((type_res == "NOT_RUN").sum())

        effective_total = total - parent_fail - not_run
        rate = failed / effective_total if effective_total > 0 else 0.0
        highlighted = rate >= threshold

        host_failures: dict[str, dict[str, tuple[int, int, str]]] = {}
        if not _is_primary_type(type_str):
            if "soc_name" in type_df.columns:
                for soc_name, soc_df in type_df.groupby("soc_name"):
                    soc_str = str(soc_name)
                    soc_res = result_upper.loc[soc_df.index]
                    is_nord = "nord" in soc_str.lower()
                    host_failures[soc_str] = {}

                    if is_nord and host_col and "device_id" in soc_df.columns:
                        for dev_val, dev_df in soc_df.groupby("device_id"):
                            d_res = result_upper.loc[dev_df.index]
                            fail_c = int((d_res == "FAIL").sum())
                            pass_c = int((d_res == "PASS").sum())
                            if fail_c > 0 or pass_c > 0:
                                host_name = str(dev_df[host_col].iloc[0]) if not dev_df.empty else ""
                                host_failures[soc_str][str(dev_val)] = (fail_c, pass_c, host_name)
                    elif host_col:
                        for host_val, host_df in soc_df.groupby(host_col):
                            h_res = result_upper.loc[host_df.index]
                            fail_c = int((h_res == "FAIL").sum())
                            pass_c = int((h_res == "PASS").sum())
                            if fail_c > 0 or pass_c > 0:
                                host_failures[soc_str][str(host_val)] = (fail_c, pass_c, "")
                    else:
                        fail_c = int((soc_res == "FAIL").sum())
                        pass_c = int((soc_res == "PASS").sum())
                        if fail_c > 0 or pass_c > 0:
                            host_failures[soc_str][""] = (fail_c, pass_c, "")

        fail_rows = (
            type_df.loc[type_res == "FAIL"].head(detail_row_cap).to_dict(orient="records") if highlighted else []
        )

        stats[type_str] = TypeStats(
            test_type=type_str,
            total=total,
            passed=passed,
            failed=failed,
            parent_fail=parent_fail,
            not_run=not_run,
            failure_rate=rate,
            highlighted=highlighted,
            host_failures=host_failures,
            failure_rows=fail_rows,
        )

    return stats


def _sorted_type_stats(type_stats: dict[str, TypeStats]) -> list[TypeStats]:
    return sorted(type_stats.values(), key=lambda s: (-int(s.highlighted), -s.failure_rate))


def _render_host_failures_cell(host_failures: dict[str, dict[str, tuple[int, int, str]]], top_n: int = 5) -> str:
    if not host_failures:
        return "&mdash;"

    cards: list[str] = []
    for soc, entries in sorted(host_failures.items()):
        if not entries:
            continue
        is_nord = "nord" in soc.lower()
        top_entries = sorted(entries.items(), key=lambda x: -x[1][0])[:top_n]
        host_rows = []
        for entry_key, (fail_c, pass_c, extra) in top_entries:
            if is_nord:
                name = extra if extra else entry_key
                device_id = entry_key
                dev_part = (
                    f' <span class="text-muted">({device_id})</span>'
                    if device_id and not device_id.startswith(name)
                    else ""
                )
            else:
                name = entry_key if entry_key else "—"
                dev_part = ""
            pass_part = f'/<span class="text-success">{pass_c}</span>' if pass_c > 0 else ""
            host_rows.append(
                f'<div class="host-row">'
                f'{name} - <span class="text-danger">{fail_c}</span>{pass_part}{dev_part}'
                f"</div>"
            )
        cards.append(f'<div class="host-card"><div class="soc-label">{soc}</div>{"".join(host_rows)}</div>')

    return f'<div class="host-grid">{"".join(cards)}</div>' if cards else "&mdash;"


def _render_type_summary_table(type_stats: dict[str, TypeStats]) -> str:
    rows: list[str] = []
    for s in _sorted_type_stats(type_stats):
        badge = '<span class="badge-high">HIGH</span>' if s.highlighted else ""
        flag_cls = "text-danger" if s.highlighted else ""
        row_cls = ' class="flagged"' if s.highlighted else ""
        pct_str = f"{s.failure_pct:.1f}%" if s.effective_total > 0 else "N/A"
        rows.append(
            f"<tr{row_cls}>"
            f"<td>{s.test_type}{badge}</td>"
            f'<td class="center">{s.total}</td>'
            f'<td class="center text-success">{s.passed}</td>'
            f'<td class="center {flag_cls}">{s.failed}</td>'
            f'<td class="center text-muted">{s.parent_fail}</td>'
            f'<td class="center text-muted">{s.not_run}</td>'
            f'<td class="center {flag_cls}">{pct_str}</td>'
            f"<td>{_render_host_failures_cell(s.host_failures)}</td>"
            f"</tr>"
        )
    header = (
        "<tr>"
        '<th class="left">Test Type</th>'
        "<th>Total</th><th>Pass</th>"
        "<th>FAIL</th><th>PARENT_FAIL</th><th>NOT_RUN</th>"
        "<th>Failure&nbsp;%</th>"
        "<th>Failed Hosts (SoC / top-5 hosts&nbsp;&mdash;&nbsp;fail/pass)</th>"
        "</tr>"
    )
    return f'<table class="data"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'


def _render_overview_table(runs: list[RunAnalysis]) -> str:
    rows: list[str] = []
    for r in runs:
        fail_cls = "text-danger" if r.total_failed else "text-success"
        flag_cls = "text-danger" if r.has_flags else "text-success"
        rows.append(
            "<tr>"
            f'<td><a class="text-primary" href="#{r.run_id}">{r.run_id}</a></td>'
            f'<td class="center">{r.total_tests}</td>'
            f'<td class="center text-success">{r.total_passed}</td>'
            f'<td class="center {fail_cls}">{r.total_failed}</td>'
            f'<td class="center text-muted">{r.total_parent_fail}</td>'
            f'<td class="center text-muted">{r.total_not_run}</td>'
            f'<td class="center {fail_cls}">{r.overall_failure_pct:.1f}%</td>'
            f'<td class="center {flag_cls}">{len(r.flagged_types)} flagged</td>'
            "</tr>"
        )
    header = (
        "<tr>"
        '<th class="left">Run ID</th>'
        "<th>Total</th><th>Pass</th>"
        "<th>FAIL</th><th>PARENT_FAIL</th><th>NOT_RUN</th>"
        "<th>Failure&nbsp;%</th><th>Flagged Types</th>"
        "</tr>"
    )
    return f'<table class="data"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'


def _elapsed_label(start_time_str: str) -> str:
    """Return human-readable elapsed time since *start_time_str*.

    Examples: ``'2h 15m'``, ``'1d 3h'``.  Returns ``''`` if unparseable.
    """
    try:
        s = start_time_str.rstrip("Z").split("+")[0]
        delta = datetime.now() - datetime.fromisoformat(s)
        total_minutes = max(0, int(delta.total_seconds())) // 60
        hours, minutes = divmod(total_minutes, 60)
        if hours >= 24:
            days, remaining_hours = divmod(hours, 24)
            return f"{days}d {remaining_hours}h"
        return f"{hours}h {minutes}m"
    except Exception:
        return ""


def _render_run_section(run: RunAnalysis, threshold_pct: int) -> str:
    ji = run.job_info
    status = ji.get("status", "RUNNING")
    start_time = ji.get("start_time") or ji.get("created_at") or ji.get("submitted_at") or "N/A"
    job_id = ji.get("job_id") or ji.get("run_id") or "N/A"
    flagged_count = len(run.flagged_types)
    flag_cls = "text-danger" if flagged_count else "text-success"
    elapsed = _elapsed_label(start_time) if start_time != "N/A" else ""
    elapsed_part = f' <span class="text-muted">(started {elapsed} ago)</span>' if elapsed else ""
    meta_rows = "\n".join(
        [
            f'<tr><td class="label">Job ID</td><td>{job_id}</td></tr>',
            f'<tr><td class="label">Status</td><td>{status}</td></tr>',
            f'<tr><td class="label">Start Time</td><td>{start_time}{elapsed_part}</td></tr>',
            f'<tr><td class="label">Types Flagged</td>'
            f'<td class="{flag_cls}">{flagged_count} / {len(run.type_stats)}'
            f" types &ge; {threshold_pct}% failure rate</td></tr>",
        ]
    )
    return (
        f'<hr class="section">'
        f'<h2 id="{run.run_id}">{run.run_id}</h2>'
        f'<table class="meta"><tbody>{meta_rows}</tbody></table>' + _render_type_summary_table(run.type_stats)
    )


def build_combined_stability_html(
    runs: list[RunAnalysis],
    threshold: float = StabilityReportConfig.FAILURE_THRESHOLD,
) -> str:
    """Render a single combined HTML stability report for all *runs*.

    Args:
        runs: List of :class:`RunAnalysis` objects.
        threshold: Failure rate used to flag types.

    Returns:
        Complete HTML string.
    """
    threshold_pct = int(threshold * 100)
    run_ids_list = "".join(f"<li>{r.run_id}</li>" for r in runs)
    run_sections = "\n".join(_render_run_section(r, threshold_pct) for r in runs if r.has_flags)
    return f"""<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Auto Nightly Hourly Failure Analysis</title>
            <style>{_CSS}</style>
            </head>
            <body>
            <h1>Auto Nightly Hourly Failure Analysis</h1>
            <p class="subtitle">Run IDs:</p>
            <ul class="run-id-list">{run_ids_list}</ul>
            <p class="subtitle">
                <strong>Flagged types</strong> have Failure&nbsp;% &ge; {threshold_pct}%.
                &nbsp;|&nbsp;
                <strong>Failure&nbsp;%</strong> = FAIL &divide; (Total &minus; PARENT_FAIL &minus; NOT_RUN) &times; 100
            </p>
            <h2>Overview &mdash; All Runs</h2>
            {_render_overview_table(runs)}
            {run_sections}
            </body>
            </html>"""


class StabilityMonitor:
    """High-level stability monitoring orchestrator.

    Wraps the functional ``analyze_type_failures`` / ``build_combined_stability_html``
    API in a class that can be dependency-injected with notifiers.

    Args:
        notifiers: List of :class:`~src.core.interfaces.INotifier` instances to
            call when a flagged run is detected.
        failure_threshold: Per-type failure rate (0 - 1) above which a type is flagged.

    Example::

        monitor = StabilityMonitor(notifiers=[TeamsNotifier(webhook_url=...)])
        run = monitor.analyze(run_id="QNN-001", job_info={}, df=df)
        monitor.notify([run])
    """

    def __init__(
        self,
        notifiers: list[INotifier] | None = None,
        failure_threshold: float = StabilityReportConfig.FAILURE_THRESHOLD,
    ) -> None:
        self.notifiers = notifiers or []
        self.failure_threshold = failure_threshold

    def analyze(self, run_id: str, job_info: dict[str, Any], df: pd.DataFrame) -> RunAnalysis:
        """Analyze a single run's failure DataFrame.

        Args:
            run_id: Test-plan run identifier.
            job_info: Metadata dict from the DAG API.
            df: Full result DataFrame loaded from the run's pickle artifact.

        Returns:
            :class:`RunAnalysis` with aggregated type statistics.
        """
        type_stats = analyze_type_failures(df)
        return RunAnalysis(run_id=run_id, job_info=job_info, type_stats=type_stats)

    def build_html_report(self, runs: list[RunAnalysis]) -> str:
        """Build a combined HTML stability report for all *runs*."""
        return build_combined_stability_html(runs)

    def notify(self, runs: list[RunAnalysis]) -> None:
        """Send notifications via all registered notifiers.

        Args:
            runs: List of analysed runs.
        """
        html_report = self.build_html_report(runs)
        for notifier in self.notifiers:
            try:
                notifier.send(
                    subject="Hourly Report — AUTO",
                    body=html_report,
                    runs=runs,
                )
            except Exception as e:
                logger.error(f"Notifier {type(notifier).__name__} failed: {e}")
