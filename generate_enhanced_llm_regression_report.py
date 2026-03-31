"""
Enhanced Regression Report Generator
Reads the saved regression object from joblib artifacts and the existing HTML
(to extract already-generated LLM summaries), then produces a new enhanced HTML
with KPI cards, bar charts, donut charts, heatmap, and LLM summaries.

No LLM re-generation is performed — all summaries are extracted from the existing HTML.
"""

import os
import re
import sys
from html.parser import HTMLParser

from joblib import load

# ─── Paths ────────────────────────────────────────────────────────────────────
QAIRT_ID = "qaisw-v2.46.0.260312041218_nightly"
CONSOLIDATED_REPORTS_BASE = "/prj/qct/webtech_hyd19/CONSOLIDATED_REPORTS"
QAIRT_FOLDER = os.path.join(CONSOLIDATED_REPORTS_BASE, QAIRT_ID)

RUN_ID_A = "QNN-v2.46.0.260312041218_llm-llm_nightly"
RUN_ID_B = "QNN-v2.45.0.260311041706_llm-llm_nightly"
PAIR_KEY = f"{RUN_ID_A}_{RUN_ID_B}"

REGRESSION_ARTIFACTS_PATH = os.path.join(
    QAIRT_FOLDER, "regression_artifacts", f"{QAIRT_ID}_regression_analysis_object.joblib"
)
EXISTING_HTML_PATH = os.path.join(QAIRT_FOLDER, "regression_htmls", PAIR_KEY, f"{PAIR_KEY}.html")
OUTPUT_HTML_PATH = os.path.join(QAIRT_FOLDER, "regression_htmls", PAIR_KEY, f"{PAIR_KEY}_enhanced.html")

SERVER_PREFIX = "https://aisw-hyd.qualcomm.com/fs"

# ─── CSS ──────────────────────────────────────────────────────────────────────
REPORT_CSS = """
<style>
    :root {
        --primary: #00629B;
        --secondary: #3253DC;
        --bg-body: #f0f4f8;
        --bg-container: #ffffff;
        --text-main: #333333;
        --text-muted: #6c757d;
        --border-color: #e2e8f0;
        --table-head-bg: #00629B;
        --table-head-text: #ffffff;
        --row-hover: #ebf5ff;
        --accent-danger: #e74c3c;
        --accent-success: #27ae60;
        --accent-warn: #e67e22;
        --accent-purple: #8e44ad;
        --shadow: 0 2px 8px rgba(0,0,0,0.08);
        --shadow-lg: 0 4px 16px rgba(0,0,0,0.12);
    }

    * { box-sizing: border-box; }

    body {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background-color: var(--bg-body);
        color: var(--text-main);
        margin: 0;
        padding: 24px;
        line-height: 1.6;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        background-color: var(--bg-container);
        padding: 40px 48px;
        border-radius: 12px;
        box-shadow: var(--shadow-lg);
    }

    h1, h2, h3, h4 { color: var(--primary); font-weight: 700; margin-top: 1.5em; margin-bottom: 0.6em; }
    h1 { font-size: 2em; border-bottom: 3px solid var(--primary); padding-bottom: 12px; margin-top: 0; }
    h2 { font-size: 1.55em; border-bottom: 2px solid var(--border-color); padding-bottom: 8px; }
    h3 { font-size: 1.25em; color: #2c3e50; }
    h4 { font-size: 1em; color: var(--primary); }

    /* ── KPI Cards ── */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 32px; }
    .kpi {
        background: #fff;
        border-radius: 12px;
        padding: 20px 18px;
        box-shadow: var(--shadow);
        border-left: 5px solid var(--primary);
        transition: transform .15s;
    }
    .kpi:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }
    .kpi.danger  { border-left-color: var(--accent-danger); }
    .kpi.warn    { border-left-color: var(--accent-warn); }
    .kpi.success { border-left-color: var(--accent-success); }
    .kpi.purple  { border-left-color: var(--accent-purple); }
    .kpi .lbl { font-size: .72em; color: var(--text-muted); text-transform: uppercase; letter-spacing: .9px; font-weight: 600; margin-bottom: 6px; }
    .kpi .val { font-size: 2.2em; font-weight: 800; color: var(--primary); line-height: 1; }
    .kpi.danger  .val { color: var(--accent-danger); }
    .kpi.warn    .val { color: var(--accent-warn); }
    .kpi.success .val { color: var(--accent-success); }
    .kpi.purple  .val { color: var(--accent-purple); }
    .kpi .sub { font-size: .72em; color: var(--text-muted); margin-top: 4px; }

    /* ── Section wrapper ── */
    .section { background: #fff; border: 1px solid var(--border-color); border-radius: 10px; padding: 24px 28px; margin-bottom: 28px; }
    .section-title { font-size: 1.05em; font-weight: 700; color: var(--primary); margin: 0 0 18px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }

    /* ── Charts grid ── */
    .chart-grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
    .chart-grid-3col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
    .chart-box { background: #f8fafc; border: 1px solid var(--border-color); border-radius: 8px; padding: 16px 18px; }
    .chart-box h4 { margin: 0 0 14px; font-size: .88em; color: var(--primary); text-transform: uppercase; letter-spacing: .5px; }

    /* ── Horizontal bar chart ── */
    .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: .82em; }
    .bar-row .bar-label { width: 160px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; color: var(--text-main); font-weight: 500; }
    .bar-row .bar-track { flex: 1; background: #e2e8f0; border-radius: 4px; height: 14px; }
    .bar-row .bar-fill  { height: 14px; border-radius: 4px; transition: width .3s; }
    .bar-row .bar-val   { width: 40px; text-align: right; font-weight: 700; flex-shrink: 0; color: var(--text-main); }

    /* ── Donut chart (CSS-only) ── */
    .donut-wrap { display: flex; align-items: center; gap: 20px; }
    .donut-legend { font-size: .8em; line-height: 2; }
    .donut-legend .dot { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }

    /* ── Heatmap ── */
    .heatmap-wrap { overflow-x: auto; margin-bottom: 24px; }
    .heatmap-wrap table { border-collapse: collapse; font-size: .86em; min-width: 100%; }
    .heatmap-wrap th { background: var(--table-head-bg); color: var(--table-head-text); padding: 9px 14px; font-size: .78em; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }
    .heatmap-wrap td { text-align: center; padding: 9px 14px; font-weight: 700; font-size: .84em; border: 1px solid var(--border-color); white-space: nowrap; }
    .heat-0 { background: #f8fafc; color: #bbb; }
    .heat-1 { background: #fff3cd; color: #856404; }
    .heat-2 { background: #ffd6a5; color: #7d4e00; }
    .heat-3 { background: #ffb3b3; color: #8b0000; }
    .heat-4 { background: #ff6b6b; color: #fff; }
    .heat-5 { background: #c0392b; color: #fff; }
    .heat-legend { display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; align-items: center; font-size: .78em; color: var(--text-muted); }
    .heat-legend span { padding: 2px 10px; border-radius: 4px; font-weight: 700; }

    /* ── Tables ── */
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; background: white; border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden; }
    th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border-color); font-size: .92em; vertical-align: top; word-wrap: break-word; }
    th { background-color: var(--table-head-bg); color: var(--table-head-text); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; font-size: .82em; }
    tr:nth-child(even) { background-color: #f8fafc; }
    tr:hover { background-color: var(--row-hover); }

    /* ── Executive summary table ── */
    table.exec-summary { table-layout: fixed; }
    table.exec-summary th { background-color: #2c3e50; text-align: left; padding: 10px 14px; }
    table.exec-summary td { vertical-align: top; padding: 14px; }
    table.exec-summary td li { margin-bottom: 4px; list-style: disc; margin-left: 20px; }

    /* ── QGenie summary cell ── */
    td.qgenie-summary { vertical-align: top; padding: 10px; }
    td.qgenie-summary ul { margin: 0; padding-left: 20px; list-style: disc; }
    td.qgenie-summary li { margin-bottom: 4px; font-size: .9em; }

    /* ── Links ── */
    a { color: var(--secondary); text-decoration: none; font-weight: 500; }
    a:hover { text-decoration: underline; color: #003d73; }

    /* ── Cluster badge ── */
    .cluster-badge {
        display: inline-block;
        background: #e8f4fd;
        color: var(--primary);
        border: 1px solid #bee3f8;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: .78em;
        font-weight: 600;
        margin: 2px;
    }

    /* ── Footer ── */
    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border-color); color: var(--text-muted); font-size: .88em; text-align: center; }

    ul { margin: 0; padding-left: 20px; }
    li { margin-bottom: 5px; }

    @media (max-width: 900px) {
        .chart-grid-2col, .chart-grid-3col { grid-template-columns: 1fr; }
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
</style>
"""

# ─── Colour palettes ───────────────────────────────────────────────────────────
PALETTE_DANGER = ["#e74c3c", "#c0392b", "#ff6b6b", "#ff9999", "#ffb3b3"]
PALETTE_BLUE = ["#00629B", "#3253DC", "#2980b9", "#5dade2", "#85c1e9"]
PALETTE_MIXED = [
    "#e74c3c",
    "#e67e22",
    "#f1c40f",
    "#27ae60",
    "#3498db",
    "#8e44ad",
    "#1abc9c",
    "#d35400",
    "#2c3e50",
    "#7f8c8d",
    "#c0392b",
    "#16a085",
    "#2980b9",
    "#8e44ad",
    "#27ae60",
]


# ─── HTML summary extractor ────────────────────────────────────────────────────
class SummaryExtractor(HTMLParser):
    """
    Extracts the inner HTML of every <td class="qgenie-summary"> block
    and the two <td> blocks inside the exec-summary table.
    """

    def __init__(self):
        super().__init__()
        self._in_qgenie = False
        self._depth = 0
        self._buf = []
        self.qgenie_summaries = []  # list of raw HTML strings

        self._in_exec = False
        self._exec_depth = 0
        self._exec_buf = []
        self.exec_summaries = []  # [soc_runtime_html, model_html]

        self._in_exec_table = False
        self._exec_table_depth = 0

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")

        # Detect exec-summary table
        if tag == "table" and "exec-summary" in cls:
            self._in_exec_table = True
            self._exec_table_depth = 1
            return

        if self._in_exec_table:
            if tag == "table":
                self._exec_table_depth += 1
            if tag == "td" and not self._in_exec:
                self._in_exec = True
                self._exec_depth = 1
                self._exec_buf = []
                return
            if self._in_exec:
                self._exec_depth += 1
                self._exec_buf.append(self._reconstruct_tag(tag, attrs))
                return

        # Detect qgenie-summary td
        if tag == "td" and "qgenie-summary" in cls:
            self._in_qgenie = True
            self._depth = 1
            self._buf = []
            return

        if self._in_qgenie:
            self._depth += 1
            self._buf.append(self._reconstruct_tag(tag, attrs))

    def handle_endtag(self, tag):
        if self._in_exec_table:
            if tag == "table":
                self._exec_table_depth -= 1
                if self._exec_table_depth == 0:
                    self._in_exec_table = False
                    return
            if self._in_exec:
                self._exec_depth -= 1
                if self._exec_depth == 0:
                    self._in_exec = False
                    self.exec_summaries.append("".join(self._exec_buf))
                    self._exec_buf = []
                    return
                self._exec_buf.append(f"</{tag}>")
                return

        if self._in_qgenie:
            self._depth -= 1
            if self._depth == 0:
                self._in_qgenie = False
                self.qgenie_summaries.append("".join(self._buf))
                self._buf = []
                return
            self._buf.append(f"</{tag}>")

    def handle_data(self, data):
        if self._in_exec and self._in_exec_table:
            self._exec_buf.append(data)
        elif self._in_qgenie:
            self._buf.append(data)

    def handle_entityref(self, name):
        ref = f"&{name};"
        if self._in_exec and self._in_exec_table:
            self._exec_buf.append(ref)
        elif self._in_qgenie:
            self._buf.append(ref)

    def handle_charref(self, name):
        ref = f"&#{name};"
        if self._in_exec and self._in_exec_table:
            self._exec_buf.append(ref)
        elif self._in_qgenie:
            self._buf.append(ref)

    @staticmethod
    def _reconstruct_tag(tag, attrs):
        attr_str = ""
        for k, v in attrs:
            if v is None:
                attr_str += f" {k}"
            else:
                attr_str += f' {k}="{v}"'
        return f"<{tag}{attr_str}>"


def extract_summaries_from_html(html_path):
    """Return (exec_summaries_list, qgenie_summaries_list) from existing HTML."""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    parser = SummaryExtractor()
    parser.feed(content)
    return parser.exec_summaries, parser.qgenie_summaries


# ─── Chart helpers ────────────────────────────────────────────────────────────
def horizontal_bar_chart(title, data_dict, color_list=None, max_label_width=160):
    """
    data_dict: {label: count}
    Returns HTML string for a bar chart inside a .chart-box.
    """
    if not data_dict:
        return ""
    max_val = max(data_dict.values()) or 1
    if color_list is None:
        color_list = PALETTE_MIXED
    rows = ""
    for i, (label, val) in enumerate(sorted(data_dict.items(), key=lambda x: -x[1])):
        pct = int(val / max_val * 100)
        color = color_list[i % len(color_list)]
        rows += (
            f'<div class="bar-row">'
            f'<span class="bar-label" title="{label}">{label}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="bar-val">{val}</span>'
            f"</div>"
        )
    return f'<div class="chart-box"><h4>{title}</h4>{rows}</div>'


def css_donut_chart(title, data_dict, color_list=None, size=120):
    """
    Renders a CSS conic-gradient donut chart with a legend.
    data_dict: {label: count}
    """
    if not data_dict:
        return ""
    if color_list is None:
        color_list = PALETTE_MIXED
    total = sum(data_dict.values()) or 1
    items = sorted(data_dict.items(), key=lambda x: -x[1])

    # Build conic-gradient stops
    stops = []
    cumulative = 0
    colors_used = []
    for i, (label, val) in enumerate(items):
        color = color_list[i % len(color_list)]
        colors_used.append(color)
        pct_start = cumulative / total * 100
        cumulative += val
        pct_end = cumulative / total * 100
        stops.append(f"{color} {pct_start:.1f}% {pct_end:.1f}%")

    gradient = ", ".join(stops)
    donut_style = (
        f"width:{size}px;height:{size}px;border-radius:50%;" f"background:conic-gradient({gradient});" f"flex-shrink:0;"
    )
    hole_style = (
        f"width:{int(size*0.55)}px;height:{int(size*0.55)}px;"
        f"background:white;border-radius:50%;margin:auto;"
        f"margin-top:{int(size*0.225)}px;"
    )

    legend_items = ""
    for i, (label, val) in enumerate(items):
        pct = val / total * 100
        color = colors_used[i]
        legend_items += (
            f'<div><span class="dot" style="background:{color}"></span>' f"{label}: <b>{val}</b> ({pct:.1f}%)</div>"
        )

    return (
        f'<div class="chart-box"><h4>{title}</h4>'
        f'<div class="donut-wrap">'
        f'<div style="{donut_style}"><div style="{hole_style}"></div></div>'
        f'<div class="donut-legend">{legend_items}</div>'
        f"</div></div>"
    )


def heat_class(val, thresholds=(0, 1, 5, 20, 50, 100)):
    """Map a count to a heat-N CSS class."""
    if val == 0:
        return "heat-0"
    for i, t in enumerate(thresholds[1:], 1):
        if val <= t:
            return f"heat-{i}"
    return "heat-5"


def soc_runtime_heatmap(soc_runtime_matrix, socs, runtimes):
    """
    soc_runtime_matrix: {soc: {runtime: count}}
    Returns HTML heatmap table.
    """
    header = "<tr><th>SOC \\ Runtime</th>" + "".join(f"<th>{r.upper()}</th>" for r in runtimes) + "<th>Total</th></tr>"
    rows = ""
    for soc in socs:
        row_total = sum(soc_runtime_matrix.get(soc, {}).get(rt, 0) for rt in runtimes)
        cells = ""
        for rt in runtimes:
            val = soc_runtime_matrix.get(soc, {}).get(rt, 0)
            cls = heat_class(val)
            cells += f'<td class="{cls}">{val if val else "-"}</td>'
        total_cls = heat_class(row_total)
        cells += f'<td class="{total_cls}"><b>{row_total}</b></td>'
        rows += f"<tr><th style='background:#f0f4f8;color:#2c3e50;text-align:left;'>{soc}</th>{cells}</tr>"

    legend = (
        '<div class="heat-legend">'
        '<span class="heat-0">0</span>'
        '<span class="heat-1">1–5</span>'
        '<span class="heat-2">6–20</span>'
        '<span class="heat-3">21–50</span>'
        '<span class="heat-4">51–100</span>'
        '<span class="heat-5">&gt;100</span>'
        "</div>"
    )
    return (
        f'<div class="heatmap-wrap">' f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>" f"{legend}</div>"
    )


# ─── Main report builder ───────────────────────────────────────────────────────
def build_enhanced_report():
    print(f"Loading regression object from: {REGRESSION_ARTIFACTS_PATH}")
    all_objects = load(REGRESSION_ARTIFACTS_PATH, mmap_mode="r")
    llm_obj = all_objects.get(RUN_ID_A)
    if llm_obj is None:
        print(f"ERROR: Key '{RUN_ID_A}' not found in joblib object.")
        sys.exit(1)

    rd = llm_obj.regression_data
    print(f"Loaded regression data: {rd.get('run_id_a')} vs {rd.get('run_id_b')}")

    # ── Extract summaries from existing HTML ──────────────────────────────────
    print(f"Extracting LLM summaries from existing HTML: {EXISTING_HTML_PATH}")
    exec_summaries, qgenie_summaries = extract_summaries_from_html(EXISTING_HTML_PATH)
    print(f"  Found {len(exec_summaries)} exec summary cells, {len(qgenie_summaries)} qgenie summary cells")

    # exec_summaries[0] = SOC/Runtime summary, exec_summaries[1] = Model summary
    exec_soc_html = exec_summaries[0] if len(exec_summaries) > 0 else "<p>-</p>"
    exec_model_html = exec_summaries[1] if len(exec_summaries) > 1 else "<p>-</p>"

    # qgenie_summaries order: [cpu, htp, Hawi, Kaanapali, LemansIVI, ...]
    # We'll map them by order of appearance in the existing HTML

    # ── Compute metrics from regression_data ─────────────────────────────────
    type_data = rd.get("type", {})
    model_data = rd.get("model", {})

    # Type/runtime failure counts
    type_runtime_counts = {}  # {type: {runtime: count}}
    soc_counts = {}  # {soc: count}
    runtime_counts = {}  # {runtime: count}
    soc_runtime_matrix = {}  # {soc: {runtime: count}}
    cluster_counts = {}  # {cluster_name: count}

    for t, runtimes in type_data.items():
        type_runtime_counts[t] = {}
        for rt, clusters in runtimes.items():
            cnt = sum(len(v) for v in clusters.values())
            type_runtime_counts[t][rt] = cnt
            runtime_counts[rt] = runtime_counts.get(rt, 0) + cnt
            for cluster_name, entries in clusters.items():
                cluster_counts[cluster_name] = cluster_counts.get(cluster_name, 0) + len(entries)
                for e in entries:
                    soc = e.get("soc_name", "Unknown")
                    soc_counts[soc] = soc_counts.get(soc, 0) + 1
                    soc_runtime_matrix.setdefault(soc, {})
                    soc_runtime_matrix[soc][rt] = soc_runtime_matrix[soc].get(rt, 0) + 1

    total_failures = sum(soc_counts.values())
    total_models_failed = len(model_data)
    top_soc = max(soc_counts, key=soc_counts.get) if soc_counts else "N/A"
    top_soc_count = soc_counts.get(top_soc, 0)
    htp_failures = runtime_counts.get("htp", 0)
    cpu_failures = runtime_counts.get("cpu", 0)

    all_socs = sorted(soc_counts.keys(), key=lambda s: -soc_counts[s])
    all_runtimes = sorted(runtime_counts.keys())

    # ── Build qgenie summary map ──────────────────────────────────────────────
    # Order in existing HTML: runtime rows first (cpu, htp), then SOC rows
    runtime_order = sorted(runtime_counts.keys())
    soc_order = all_socs

    qgenie_idx = 0
    runtime_qgenie = {}
    for rt in runtime_order:
        if qgenie_idx < len(qgenie_summaries):
            runtime_qgenie[rt] = qgenie_summaries[qgenie_idx]
            qgenie_idx += 1

    soc_qgenie = {}
    for soc in soc_order:
        if qgenie_idx < len(qgenie_summaries):
            soc_qgenie[soc] = qgenie_summaries[qgenie_idx]
            qgenie_idx += 1

    # ── Start building HTML ───────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enhanced Regression Report — {RUN_ID_A} vs {RUN_ID_B}</title>
{REPORT_CSS}
</head>
<body>
<div class="container">

<h1>Regression Analysis Report</h1>
<p style="color:var(--text-muted);margin-top:-10px;font-size:.95em;">
  <b>{RUN_ID_A}</b> &nbsp;vs&nbsp; <b>{RUN_ID_B}</b>
</p>
"""

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    html += """<div class="kpi-grid">"""
    html += f'<div class="kpi danger"><div class="lbl">Total Failures</div><div class="val">{total_failures}</div><div class="sub">across all types &amp; SOCs</div></div>'
    html += f'<div class="kpi warn"><div class="lbl">Models Failed</div><div class="val">{total_models_failed}</div><div class="sub">unique model regressions</div></div>'
    html += f'<div class="kpi danger"><div class="lbl">HTP Failures</div><div class="val">{htp_failures}</div><div class="sub">on HTP runtime</div></div>'
    html += f'<div class="kpi"><div class="lbl">CPU Failures</div><div class="val">{cpu_failures}</div><div class="sub">on CPU runtime</div></div>'
    html += f'<div class="kpi purple"><div class="lbl">SOCs Affected</div><div class="val">{len(soc_counts)}</div><div class="sub">unique SoC platforms</div></div>'
    html += f'<div class="kpi danger"><div class="lbl">Top SOC</div><div class="val">{top_soc_count}</div><div class="sub">{top_soc}</div></div>'
    html += f'<div class="kpi warn"><div class="lbl">Failure Clusters</div><div class="val">{len(cluster_counts)}</div><div class="sub">distinct issue groups</div></div>'
    html += f'<div class="kpi success"><div class="lbl">Test Types</div><div class="val">{len(type_data)}</div><div class="sub">affected test categories</div></div>'
    html += "</div>"

    # ── Executive Summary ─────────────────────────────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">Executive Summary of Failures</div>'
    html += "<table class='exec-summary'>"
    html += "<tr><th style='width:50%'>SOC / Runtime Summary</th><th style='width:50%'>Model Summary</th></tr>"
    html += f"<tr><td>{exec_soc_html}</td><td>{exec_model_html}</td></tr>"
    html += "</table>"
    html += "</div>"

    # ── Failure Distribution Charts ───────────────────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">Failure Distribution</div>'
    html += '<div class="chart-grid-2col">'

    # SOC bar chart
    html += horizontal_bar_chart("Failures by SOC", soc_counts, PALETTE_DANGER)

    # Runtime donut
    html += css_donut_chart("Failures by Runtime", runtime_counts, PALETTE_BLUE)

    html += "</div>"
    html += '<div class="chart-grid-2col">'

    # Type bar chart (flatten type/runtime)
    type_totals = {t: sum(rts.values()) for t, rts in type_runtime_counts.items()}
    html += horizontal_bar_chart("Failures by Test Type", type_totals, PALETTE_MIXED)

    # Top clusters bar chart (top 10)
    top_clusters = dict(sorted(cluster_counts.items(), key=lambda x: -x[1])[:12])
    html += horizontal_bar_chart("Top Failure Clusters", top_clusters, PALETTE_MIXED)

    html += "</div>"
    html += "</div>"

    # ── SOC × Runtime Heatmap ─────────────────────────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">SOC × Runtime Failure Heatmap</div>'
    html += soc_runtime_heatmap(soc_runtime_matrix, all_socs, all_runtimes)
    html += "</div>"

    # ── Type-based Failures Table ─────────────────────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">Type-based Failures</div>'
    col_headers = ["htp", "htp_fp16", "mcp", "gpu", "gpu_fp16", "cpu", "mcp_x86", "htp_x86", "lpai"]
    html += "<table><tr><th>Type / Runtime</th>"
    for h in col_headers:
        html += f"<th>{h.capitalize()}</th>"
    html += "</tr>"

    for t, rts in type_runtime_counts.items():
        html += f"<tr><th style='background:#f0f4f8;color:#2c3e50'>{t.replace('_', ' ').title()}</th>"
        for rt in col_headers:
            cnt = rts.get(rt, 0)
            if cnt:
                link = f"{SERVER_PREFIX}{QAIRT_FOLDER}/regression_htmls/{PAIR_KEY}/{t}_{rt}_clustered_regression_report.html"
                html += f"<td><a href='{link}'>{cnt}</a></td>"
            else:
                html += "<td style='color:#bbb'>—</td>"
        html += "</tr>"
    html += "</table>"
    html += "</div>"

    # ── Failed Model Report ───────────────────────────────────────────────────
    model_link = f"{SERVER_PREFIX}{QAIRT_FOLDER}/regression_htmls/{PAIR_KEY}/model_failures_regression_report.html"
    html += '<div class="section">'
    html += '<div class="section-title">Failed Model Report</div>'
    html += f'<p><a href="{model_link}">&#128196; View Full Model Failure Report</a> &nbsp;—&nbsp; <b>{total_models_failed}</b> models failed</p>'
    html += "</div>"

    # ── Runtime-based Failures with LLM Summaries ────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">Runtime-based Failures with AI Insights</div>'
    html += (
        "<table><tr><th style='width:120px'>Runtime</th><th style='width:80px'>Failures</th><th>AI Summary</th></tr>"
    )
    for rt in runtime_order:
        cnt = runtime_counts.get(rt, 0)
        link = f"{SERVER_PREFIX}{QAIRT_FOLDER}/regression_htmls/{PAIR_KEY}/{rt}_clustered_regression_report.html"
        summary_html = runtime_qgenie.get(rt, "<p>—</p>")
        html += (
            f"<tr>"
            f"<td><b>{rt.upper()}</b></td>"
            f"<td><a href='{link}'>{cnt}</a></td>"
            f'<td class="qgenie-summary"><ul>{summary_html}</ul></td>'
            f"</tr>"
        )
    html += "</table>"
    html += "</div>"

    # ── SOC-based Failures with LLM Summaries ────────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">SOC-based Failures with AI Insights</div>'
    html += "<table><tr><th style='width:140px'>SOC</th><th style='width:80px'>Failures</th><th>AI Summary</th></tr>"
    for soc in soc_order:
        cnt = soc_counts.get(soc, 0)
        link = f"{SERVER_PREFIX}{QAIRT_FOLDER}/regression_htmls/{PAIR_KEY}/{soc}_clustered_regression_report.html"
        summary_html = soc_qgenie.get(soc, "<p>—</p>")
        html += (
            f"<tr>"
            f"<td><b>{soc}</b></td>"
            f"<td><a href='{link}'>{cnt}</a></td>"
            f'<td class="qgenie-summary"><ul>{summary_html}</ul></td>'
            f"</tr>"
        )
    html += "</table>"
    html += "</div>"

    # ── Cluster Breakdown ─────────────────────────────────────────────────────
    html += '<div class="section">'
    html += '<div class="section-title">Failure Cluster Breakdown</div>'
    html += "<table><tr><th>Test Type</th><th>Runtime</th><th>Cluster</th><th>Count</th></tr>"
    for t, runtimes in type_data.items():
        for rt, clusters in runtimes.items():
            for cluster_name, entries in sorted(clusters.items(), key=lambda x: -len(x[1])):
                html += (
                    f"<tr>"
                    f"<td>{t.replace('_', ' ').title()}</td>"
                    f"<td>{rt.upper()}</td>"
                    f'<td><span class="cluster-badge">{cluster_name}</span></td>'
                    f"<td><b>{len(entries)}</b></td>"
                    f"</tr>"
                )
    html += "</table>"
    html += "</div>"

    # ── Footer ────────────────────────────────────────────────────────────────
    html += f"""
<div class="footer">
  Generated from saved regression artifacts &mdash; no LLM re-generation performed.<br>
  Source: <code>{REGRESSION_ARTIFACTS_PATH}</code><br>
  Report date: 2026-03-31
</div>
</div>
</body>
</html>
"""

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_HTML_PATH), exist_ok=True)
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nEnhanced HTML written to:\n  {OUTPUT_HTML_PATH}")
    print(f"Server URL:\n  {SERVER_PREFIX}{OUTPUT_HTML_PATH}")
    return OUTPUT_HTML_PATH


if __name__ == "__main__":
    build_enhanced_report()
