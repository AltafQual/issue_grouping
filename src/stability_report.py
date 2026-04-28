"""Stability report: per-type failure analysis and combined HTML report generation.

Public API
----------
    analyze_type_failures(df)           -> dict[str, TypeStats]
    build_combined_stability_html(runs) -> str

Typical usage::

    from src.stability_report import RunAnalysis, analyze_type_failures, build_combined_stability_html

    type_stats = analyze_type_failures(df)
    run        = RunAnalysis(run_id=run_id, job_info=job_info, type_stats=type_stats)
    html       = build_combined_stability_html([run, ...])

Failure % formula
-----------------
    Failure % = FAIL / (Total - PARENT_FAIL - NOT_RUN) * 100

Only tests that actually executed are in the denominator.  PARENT_FAIL and
NOT_RUN rows are shown for visibility but excluded from the rate calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.constants import StabilityReportConfig

_PRIMARY_TYPES: frozenset[str] = frozenset({"converter", "quantizer", "savecontext"})
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
    failed: int  # result == 'FAIL' only
    parent_fail: int  # result == 'PARENT_FAIL'
    not_run: int  # result == 'NOT_RUN'
    failure_rate: float  # FAIL / effective_total; 0.0 when effective_total == 0
    highlighted: bool  # True when failure_rate >= threshold

    # soc_name → { host_name → FAIL count }
    # Populated only for non-primary types.
    # When no host column exists, the inner key is an empty string ("").
    host_failures: dict[str, dict[str, int]] = field(default_factory=dict)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def effective_total(self) -> int:
        """Tests that actually executed (excludes PARENT_FAIL and NOT_RUN)."""
        return self.total - self.parent_fail - self.not_run

    @property
    def failure_pct(self) -> float:
        return self.failure_rate * 100


@dataclass
class RunAnalysis:
    """Full analysis for a single run ID."""

    run_id: str
    job_info: dict[str, Any]
    type_stats: dict[str, TypeStats]  # keyed by test_type name

    @property
    def total_tests(self) -> int:
        return sum(s.total for s in self.type_stats.values())

    @property
    def total_failed(self) -> int:
        """Count of FAIL results only."""
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

    Processing is done in explicit filter steps to prevent data mixing:

    1. Slice the full DataFrame to one ``type`` at a time.
    2. Within that slice, partition rows by ``result`` value.
    3. For non-primary types, further slice the FAIL partition by ``soc_name``
       and then by host column to build the soc→host→count mapping.

    Result values are matched case-insensitively (PASS / FAIL / PARENT_FAIL / NOT_RUN).
    Anything else counts toward *total* only.

    Args:
        df:             Must contain ``type`` and ``result`` columns.
        threshold:      Failure rate at or above which a type is flagged (default 0.20).
        detail_row_cap: Max FAIL rows stored per flagged type.

    Returns:
        Mapping of test-type name → :class:`TypeStats`.
    """
    # Normalise result column once up-front to avoid repeated str ops per row.
    result_upper: pd.Series = df["result"].astype(str).str.strip().str.upper()

    host_col: str | None = _detect_host_column(df)

    stats: dict[str, TypeStats] = {}

    # ── Step 1: isolate one type at a time ───────────────────────────────────
    for test_type, type_df in df.groupby("type"):
        type_str = str(test_type)

        # Aligned result values for this type's rows only.
        type_res: pd.Series = result_upper.loc[type_df.index]

        # ── Step 2: partition by result ──────────────────────────────────────
        total = len(type_df)
        passed = int((type_res == "PASS").sum())
        failed = int((type_res == "FAIL").sum())
        parent_fail = int((type_res == "PARENT_FAIL").sum())
        not_run = int((type_res == "NOT_RUN").sum())

        effective_total = total - parent_fail - not_run
        rate = failed / effective_total if effective_total > 0 else 0.0
        highlighted = rate >= threshold

        # ── Step 3: soc → host → count (non-primary types only) ─────────────
        host_failures: dict[str, dict[str, int]] = {}
        if not _is_primary_type(type_str):
            # Isolate FAIL rows for this type.
            fail_mask = type_res == "FAIL"
            fail_df = type_df.loc[fail_mask]

            if "soc_name" in fail_df.columns and not fail_df.empty:
                for soc_name, soc_df in fail_df.groupby("soc_name"):
                    soc_str = str(soc_name)
                    host_failures[soc_str] = {}

                    if host_col:
                        # Further split by host within this soc.
                        for host_val, host_df in soc_df.groupby(host_col):
                            host_failures[soc_str][str(host_val)] = len(host_df)
                    else:
                        # No host column — store total FAIL count under empty key.
                        host_failures[soc_str][""] = len(soc_df)

        # Collect raw FAIL rows for highlighted types (capped).
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
    """Flagged types first, then descending failure rate."""
    return sorted(type_stats.values(), key=lambda s: (-int(s.highlighted), -s.failure_rate))


def _render_host_failures_cell(host_failures: dict[str, dict[str, int]], top_n: int = 5) -> str:
    """Render SoC → host → count as a 3-column flex grid.

    Every SoC that has at least one failure gets a card.  Within each card only
    the top *top_n* hosts (by FAIL count, descending) are shown.
    """
    if not host_failures:
        return "&mdash;"

    cards: list[str] = []
    for soc, hosts in sorted(host_failures.items()):
        top_hosts = sorted(hosts.items(), key=lambda x: -x[1])[:top_n]
        host_rows = "".join(
            f'<div class="host-row">{host if host else "—"} &mdash; {count}</div>' for host, count in top_hosts
        )
        cards.append(f'<div class="host-card">' f'<div class="soc-label">{soc}</div>' f"{host_rows}" f"</div>")

    return f'<div class="host-grid">{"".join(cards)}</div>'


def _render_type_summary_table(type_stats: dict[str, TypeStats]) -> str:
    """Render the per-type pass/fail summary table for one run.

    Columns: Test Type | Total | Pass | FAIL | PARENT_FAIL | NOT_RUN | Failure % | Failed Hosts
    """
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
        "<th>Failed Hosts (SoC / top-5 hosts&nbsp;&mdash;&nbsp;count)</th>"
        "</tr>"
    )
    return f'<table class="data"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'


def _render_overview_table(runs: list[RunAnalysis]) -> str:
    """Top-level summary: one row per run ID with aggregate counts."""
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


def _render_run_section(run: RunAnalysis, threshold_pct: int) -> str:
    """Render metadata + type summary table for one run."""
    ji = run.job_info
    status = ji.get("status", "RUNNING")
    start_time = ji.get("start_time") or ji.get("created_at") or ji.get("submitted_at") or "N/A"
    job_id = ji.get("job_id") or ji.get("run_id") or "N/A"

    flagged_count = len(run.flagged_types)
    flag_cls = "text-danger" if flagged_count else "text-success"

    meta_rows = "\n".join(
        [
            f'<tr><td class="label">Job ID</td><td>{job_id}</td></tr>',
            f'<tr><td class="label">Status</td><td>{status}</td></tr>',
            f'<tr><td class="label">Start Time</td><td>{start_time}</td></tr>',
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
        runs:      List of :class:`RunAnalysis` objects (one per processed run ID).
        threshold: Failure rate used to flag types (default from :class:`StabilityReportConfig`).

    Returns:
        Complete HTML string ready to be embedded in an email or written to a file.
    """
    threshold_pct = int(threshold * 100)
    run_ids_list = "".join(f"<li>{r.run_id}</li>" for r in runs)
    run_sections = "\n".join(_render_run_section(r, threshold_pct) for r in runs)

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

if __name__ == "__main__":
    import random

    random.seed(42)

    TYPES = ["converter", "quantizer", "savecontext", "htp", "gpu", "cpu", "lpai"]
    SOCS = ["NordLE", "Kailua", "Lanai"]
    HOSTS = {
        "NordLE": ["hydciqlab01", "hydciqlab02", "hydciqlab03"],
        "Kailua": ["hydciqlab04", "hydciqlab05"],
        "Lanai": ["hydciqlab06"],
    }
    RESULTS = ["PASS", "FAIL", "PARENT_FAIL", "NOT_RUN"]
    RESULT_WEIGHTS = [0.60, 0.25, 0.10, 0.05]
    MODELS = ["mobilenet_v2", "resnet50", "bert_base", "yolov5s", "efficientnet_b0"]

    def _make_df(run_id: str) -> pd.DataFrame:
        rows = []
        for test_type in TYPES:
            n = random.randint(40, 120)
            for _ in range(n):
                soc = random.choice(SOCS)
                rows.append(
                    {
                        "tc_uuid": f"{run_id}-{random.randint(10000, 99999)}",
                        "type": test_type,
                        "result": random.choices(RESULTS, RESULT_WEIGHTS)[0],
                        "soc_name": soc,
                        "host": random.choice(HOSTS[soc]),
                        "model_name": random.choice(MODELS),
                        "runtime": random.choice(["htp_fp16", "cpu", "gpu_fp16"]),
                        "reason": "Error: segfault in layer conv2d" if random.random() < 0.3 else "",
                    }
                )
        return pd.DataFrame(rows)

    run_ids = ["QNN-auto-v2.47-260428001", "QNN-auto-v2.47-260428002"]
    processed: list[RunAnalysis] = []

    for rid in run_ids:
        df = _make_df(rid)
        ts = analyze_type_failures(df)
        processed.append(
            RunAnalysis(run_id=rid, job_info={"status": "RUNNING", "start_time": "2026-04-28 00:15:00"}, type_stats=ts)
        )

        print(f"\n── {rid} ──")
        for tname, s in sorted(ts.items()):
            flag = " [HIGH]" if s.highlighted else ""
            print(
                f"  {tname:<15}  total={s.total:>3}  pass={s.passed:>3}  "
                f"fail={s.failed:>3}  pf={s.parent_fail:>2}  nr={s.not_run:>2}  "
                f"rate={s.failure_pct:.1f}%{flag}"
            )
            for soc, hosts in s.host_failures.items():
                print(f"    {soc}")
                for h, c in sorted(hosts.items(), key=lambda x: -x[1]):
                    print(f"      {h} — {c}")

    html = build_combined_stability_html(processed)
    out_path = "/tmp/stability_report_test.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport written to {out_path}")
