"""
enhanced_consolidated_report.py
--------------------------------
Standalone script to generate an enhanced QAIRT Analysis HTML report.

Usage:
    python src/enhanced_consolidated_report.py --qairt_id qaisw-v2.46.0.260319041023_nightly

    # Skip LLM calls (metrics-only, much faster):
    python src/enhanced_consolidated_report.py --qairt_id qaisw-v2.46.0.260319041023_nightly --no_llm

    # Save LLM summaries to a JSON cache so re-runs skip the LLM calls:
    python src/enhanced_consolidated_report.py --qairt_id qaisw-v2.46.0.260319041023_nightly --cache_llm

The script reads the regression_artifacts joblib produced by the existing
consolidated_reports_analysis pipeline and generates a richer HTML report
with:
  - Executive KPI dashboard
  - LLM executive summary (Gemini 2.5 Pro via QGenie)
  - BU breakdown cards with per-BU LLM summaries
  - Failure heatmap (BU x Runtime)
  - SOC / Runtime / Test-type analysis with bar charts
  - Cluster analysis with per-cluster-class LLM summaries
  - Model failure ranking
  - DSP type breakdown
  - Gerrits table (grouped by repo)
  - Full run-ID detail table

Does NOT modify any existing file in the pipeline.
"""

import argparse
import json
import logging
import os
import re
from collections import OrderedDict, defaultdict
from html import escape

import joblib

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

CONSOLIDATED_REPORTS_PATH = "/prj/qct/webtech_hyd19/CONSOLIDATED_REPORTS"

BU_COLORS = {
    "Auto": "#e74c3c",
    "Compute (Windows)": "#2980b9",
    "GenAI": "#8e44ad",
    "Mobile/IOT/XR": "#27ae60",
    "Unknown": "#95a5a6",
}

ALL_RUNTIMES_ORDERED = ["cpu", "gpu", "gpu_fp16", "htp", "htp_fp16", "mcp", "mcp_x86", "lpai"]
BU_ORDER = ["Auto", "Compute (Windows)", "Mobile/IOT/XR", "GenAI"]

REPORT_CSS = """
<style>
:root{--primary:#00629B;--secondary:#2471a3;--bg:#f0f2f5;--white:#fff;--muted:#6c757d;--border:#dee2e6;--danger:#c0392b;--warning:#d68910;--success:#1e8449;--shadow:0 2px 8px rgba(0,0,0,.08)}
*{box-sizing:border-box}
body{font-family:'Segoe UI',Roboto,Arial,sans-serif;background:var(--bg);color:#1a1a2e;margin:0;padding:0;line-height:1.6}
.report-header{background:linear-gradient(135deg,#00629B,#003d73);color:#fff;padding:28px 40px}
.report-header h1{margin:0 0 4px;font-size:1.8em;font-weight:700}
.report-header .sub{opacity:.85;font-size:.92em}
.report-header .meta{margin-top:10px;display:flex;gap:16px;flex-wrap:wrap}
.report-header .meta span{background:rgba(255,255,255,.15);padding:3px 12px;border-radius:20px;font-size:.8em}
.toc{background:#fff;border-bottom:2px solid var(--primary);padding:0 40px;display:flex;gap:0;overflow-x:auto;position:sticky;top:0;z-index:100;box-shadow:0 2px 4px rgba(0,0,0,.08)}
.toc a{display:inline-block;padding:11px 16px;color:var(--primary);text-decoration:none;font-size:.85em;font-weight:600;border-bottom:3px solid transparent;white-space:nowrap;transition:all .15s}
.toc a:hover{border-bottom-color:var(--primary);background:#f1faff}
.container{max-width:1400px;margin:0 auto;padding:28px 36px}
.section{background:#fff;border-radius:10px;box-shadow:var(--shadow);padding:24px 28px;margin-bottom:24px}
.section-title{font-size:1.15em;font-weight:700;color:var(--primary);margin:0 0 18px;padding-bottom:10px;border-bottom:2px solid #e9ecef;display:flex;align-items:center;gap:8px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}
.kpi{background:#fff;border-radius:10px;padding:18px;box-shadow:var(--shadow);border-left:5px solid var(--primary)}
.kpi.d{border-left-color:#e74c3c}.kpi.w{border-left-color:#e67e22}.kpi.s{border-left-color:#27ae60}.kpi.p{border-left-color:#8e44ad}
.kpi .lbl{font-size:.76em;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:5px}
.kpi .val{font-size:2em;font-weight:800;color:var(--primary);line-height:1}
.kpi.d .val{color:#e74c3c}.kpi.w .val{color:#e67e22}.kpi.s .val{color:#27ae60}.kpi.p .val{color:#8e44ad}
.kpi .sub{font-size:.76em;color:var(--muted);margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:.88em}
th{background:var(--primary);color:#fff;padding:9px 13px;text-align:left;font-size:.8em;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
td{padding:9px 13px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:#f1faff}
tr:nth-child(even) td{background:#fafafa}
tr:nth-child(even):hover td{background:#f1faff}
.tw{border:1px solid var(--border);border-radius:8px;overflow:hidden}
.bu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}
.bu-card{background:#fff;border-radius:10px;box-shadow:var(--shadow);overflow:hidden}
.bu-hdr{padding:14px 18px;color:#fff;font-weight:700;font-size:.95em;display:flex;justify-content:space-between;align-items:center}
.bu-body{padding:14px 18px}
.bu-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f5f5f5;font-size:.86em}
.bu-row:last-child{border-bottom:none}
.bu-lbl{color:var(--muted)}
.bu-val{font-weight:700}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.74em;font-weight:700}
.bd{background:#fde8ea;color:#c0392b}.bw{background:#fef9e7;color:#d68910}.bs{background:#e9f7ef;color:#1e8449}.bi{background:#eaf4fb;color:#1a5276}.bp{background:#f5eef8;color:#7d3c98}
.heatmap-table th{font-size:.76em;padding:7px 9px}
.heatmap-table td{text-align:center;padding:7px 9px;font-weight:700;font-size:.86em}
.heat-0{background:#f8f9fa;color:#bbb}
.heat-1{background:#fff3cd;color:#856404}
.heat-2{background:#ffd6a5;color:#7d4e00}
.heat-3{background:#ffb3b3;color:#8b0000}
.heat-4{background:#ff6b6b;color:#fff}
.heat-5{background:#c0392b;color:#fff}
.llm-section{background:#f8fbff;border:1px solid #cce0f5;border-radius:8px;padding:18px 22px;margin-top:16px}
.llm-section h4{margin:0 0 10px;color:#00629B;font-size:.92em;display:flex;align-items:center;gap:7px}
.llm-section ul{margin:0;padding-left:20px;line-height:1.85}
.llm-section li{margin-bottom:6px;font-size:.88em}
details summary::-webkit-details-marker{display:none}
.footer{text-align:center;color:var(--muted);font-size:.8em;padding:20px;margin-top:4px}
a{color:var(--secondary);text-decoration:none;font-weight:500}
a:hover{text-decoration:underline}
@media(max-width:900px){.two-col{grid-template-columns:1fr}.container{padding:14px}}
</style>
"""


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_joblib(qairt_id: str):
    """
    Load the regression analysis joblib for a given qairt_id.
    Handles the src module dependency by injecting minimal stubs so the
    pickled RegressionAnalysisReport objects can be unpickled without
    running the full pipeline environment.
    """
    import sys
    import types

    # Build a minimal src package so pickle can resolve the classes
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [os.path.join(os.path.dirname(__file__))]
    src_pkg.__package__ = "src"
    sys.modules.setdefault("src", src_pkg)

    for stub_name in ["src.get_prev_testplan_id", "src.regression_api_call"]:
        if stub_name not in sys.modules:
            sys.modules[stub_name] = types.ModuleType(stub_name)
    sys.modules["src.get_prev_testplan_id"].iterate_db_get_testplan = (
        lambda *a, **kw: (None, None, None, None)
    )
    sys.modules["src.regression_api_call"].get_two_run_ids_cluster_info = (
        lambda *a, **kw: {}
    )

    # Import the real module to get the actual classes for unpickling
    import src.consolidated_reports_analysis as cra_real

    # Temporarily replace the module entry with a stub that exposes the real
    # classes so joblib can resolve them during unpickling
    cra_stub = types.ModuleType("src.consolidated_reports_analysis")
    cra_stub.OrderedDefaultDict = cra_real.OrderedDefaultDict
    cra_stub.RegressionAnalysisReport = cra_real.RegressionAnalysisReport
    sys.modules["src.consolidated_reports_analysis"] = cra_stub

    artifact_path = os.path.join(
        CONSOLIDATED_REPORTS_PATH,
        qairt_id,
        "regression_artifacts",
        f"{qairt_id}_regression_analysis_object.joblib",
    )
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(f"Joblib artifact not found: {artifact_path}")

    obj = joblib.load(artifact_path, mmap_mode="r")

    # Restore the real module
    sys.modules["src.consolidated_reports_analysis"] = cra_real
    return obj


# ─── Classification ───────────────────────────────────────────────────────────

def classify_run_id(run_id: str) -> str:
    r = run_id.lower()
    if "win" in r:
        return "Compute (Windows)"
    if "auto" in r:
        return "Auto"
    if "llm" in r:
        return "GenAI"
    if "pt" in r:
        return "Mobile/IOT/XR"
    return "Unknown"


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(obj: dict) -> dict:
    """
    Walk all RegressionAnalysisReport objects and build every metric dict
    needed by the HTML builder.
    """
    total_run_ids = 0
    total_failures = 0
    total_models_failed = 0
    total_model_failure_entries = 0
    total_gerrits = 0

    bu_stats = defaultdict(
        lambda: {
            "run_ids": [],
            "total_failures": 0,
            "model_failures": 0,
            "model_failure_entries": 0,
            "socs": set(),
            "runtimes": set(),
            "type_counts": defaultdict(int),
            "cluster_classes": defaultdict(int),
            "dsp_types": defaultdict(int),
            "top_models": defaultdict(int),
            "top_clusters": defaultdict(int),
            # error lists for LLM calls
            "error_summary_list": [],
        }
    )

    soc_failure_map = defaultdict(int)
    runtime_failure_map = defaultdict(int)
    type_failure_map = defaultdict(int)
    cluster_class_map = defaultdict(int)
    cluster_name_map = defaultdict(int)
    dsp_type_map = defaultdict(int)
    model_failure_map = defaultdict(int)
    gerrit_set = set()
    gerrit_list = []
    heatmap = defaultdict(lambda: defaultdict(int))

    combined_errors = []
    cluster_class_errors = defaultdict(list)

    for run_id, run_obj in obj.items():
        if not run_obj:
            continue
        rd = run_obj.regression_data
        if not rd or rd.get("status") != 200:
            continue

        total_run_ids += 1
        bu = classify_run_id(run_id)
        bu_stats[bu]["run_ids"].append(run_id)

        # Collect pre-computed error strings from the joblib object
        if run_obj.error_summary_list:
            combined_errors.extend(run_obj.error_summary_list)
            bu_stats[bu]["error_summary_list"].extend(run_obj.error_summary_list)

        # Type-based failures
        for t, runtimes in rd.get("type", {}).items():
            if not isinstance(runtimes, dict):
                continue
            for rt, clusters in runtimes.items():
                if not isinstance(clusters, dict):
                    continue
                cnt = sum(len(v) for v in clusters.values() if isinstance(v, list))
                total_failures += cnt
                bu_stats[bu]["total_failures"] += cnt
                bu_stats[bu]["type_counts"][f"{t}/{rt}"] += cnt
                runtime_failure_map[rt] += cnt
                type_failure_map[t] += cnt
                heatmap[bu][rt] += cnt
                for cluster_name, entries in clusters.items():
                    cluster_name_map[cluster_name] += len(entries)
                    bu_stats[bu]["top_clusters"][cluster_name] += len(entries)
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        soc = entry.get("soc_name", "")
                        if soc:
                            soc_failure_map[soc] += 1
                            bu_stats[bu]["socs"].add(soc)
                        cc = entry.get("cluster_class", "")
                        if cc:
                            cluster_class_map[cc] += 1
                            bu_stats[bu]["cluster_classes"][cc] += 1
                        dt = entry.get("dsp_type", "")
                        if dt:
                            dsp_type_map[dt] += 1
                            bu_stats[bu]["dsp_types"][dt] += 1
                        rt2 = entry.get("runtime", "")
                        if rt2:
                            bu_stats[bu]["runtimes"].add(rt2)
                        reason = entry.get("reason", "")
                        if cc and reason:
                            cluster_class_errors[cc].append(reason)

        # Model-based failures
        for model_name, entries in rd.get("model", {}).items():
            if not entries:
                continue
            total_models_failed += 1
            bu_stats[bu]["model_failures"] += 1
            total_model_failure_entries += len(entries)
            bu_stats[bu]["model_failure_entries"] += len(entries)
            model_failure_map[model_name] += len(entries)
            bu_stats[bu]["top_models"][model_name] += len(entries)
            for entry in entries:
                soc = entry.get("soc_name", "")
                if soc:
                    soc_failure_map[soc] += 1
                    bu_stats[bu]["socs"].add(soc)
                rt = entry.get("runtime", "")
                if rt:
                    runtime_failure_map[rt] += 1
                    bu_stats[bu]["runtimes"].add(rt)

        # Gerrits
        for _, runtime_data in rd.get("gerrit_info", {}).items():
            if not isinstance(runtime_data, dict):
                continue
            for _, gerrit_entries in runtime_data.items():
                if not isinstance(gerrit_entries, list):
                    continue
                for g in gerrit_entries:
                    url = g.get("commit_url", "")
                    if url and url not in gerrit_set:
                        gerrit_set.add(url)
                        gerrit_list.append(g)
                        total_gerrits += 1

    return dict(
        total_run_ids=total_run_ids,
        total_failures=total_failures,
        total_models_failed=total_models_failed,
        total_model_failure_entries=total_model_failure_entries,
        total_gerrits=total_gerrits,
        bu_stats=bu_stats,
        soc_failure_map=soc_failure_map,
        runtime_failure_map=runtime_failure_map,
        type_failure_map=type_failure_map,
        cluster_class_map=cluster_class_map,
        cluster_name_map=cluster_name_map,
        dsp_type_map=dsp_type_map,
        model_failure_map=model_failure_map,
        gerrit_list=gerrit_list,
        heatmap=heatmap,
        combined_errors=combined_errors,
        cluster_class_errors=cluster_class_errors,
    )


# ─── LLM summaries ───────────────────────────────────────────────────────────

def _is_empty_summary(s: str) -> bool:
    return not s or s.strip().lower() in ("no logs to provide summary", "no logs", "")


def generate_llm_summaries(data: dict, cache_path: str | None = None) -> dict:
    """
    Call get_cummilative_sumary for:
      - overall executive summary
      - per-BU summary (short)
      - per-cluster-class summary (short, capped at 80 errors)

    If cache_path is given and the file exists, load from cache instead of
    calling the LLM. Results are saved to cache_path after generation.
    """
    # Try loading from cache first
    if cache_path and os.path.isfile(cache_path):
        logger.info(f"Loading LLM summaries from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    from src.consolidated_reports_analysis import get_cummilative_sumary

    results: dict = {"executive": "", "bu": {}, "cluster_class": {}}

    logger.info("Generating executive LLM summary (%d errors)...", len(data["combined_errors"]))
    results["executive"] = get_cummilative_sumary(
        data["combined_errors"], filter=True, short_summary=False
    )

    for bu, stats in data["bu_stats"].items():
        errs = stats["error_summary_list"]
        logger.info("Generating BU LLM summary for %s (%d errors)...", bu, len(errs))
        results["bu"][bu] = get_cummilative_sumary(errs, filter=True, short_summary=True)

    for cc, errs in data["cluster_class_errors"].items():
        sample = errs[:80]
        logger.info("Generating cluster-class LLM summary for %s (%d errors)...", cc, len(sample))
        results["cluster_class"][cc] = get_cummilative_sumary(
            sample, filter=True, short_summary=True
        )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("LLM summaries cached to: %s", cache_path)

    return results


# ─── HTML helpers ─────────────────────────────────────────────────────────────

def llm_box(summary_html: str, title: str = "LLM Analysis", collapsed: bool = True) -> str:
    """Wrap an LLM summary in a collapsible <details> block."""
    if _is_empty_summary(summary_html):
        return (
            '<p style="color:#aaa;font-size:.82em;font-style:italic;margin-top:10px">'
            "&#129302; No LLM summary available (filtered logs were empty)</p>"
        )
    open_attr = "" if collapsed else "open"
    return f"""
<details {open_attr} style="margin-top:12px">
  <summary style="cursor:pointer;font-weight:700;color:#00629B;font-size:.9em;
    padding:8px 12px;background:#f0f7ff;border-radius:6px;border:1px solid #cce0f5;
    list-style:none;display:flex;align-items:center;gap:8px">
    <span style="font-size:1.1em">&#129302;</span> {title}
    <span style="margin-left:auto;font-size:.8em;color:#888;font-weight:400">click to expand/collapse</span>
  </summary>
  <div style="background:#fafcff;border:1px solid #cce0f5;border-top:none;
    border-radius:0 0 6px 6px;padding:16px 20px">
    <ul style="margin:0;padding-left:20px;line-height:1.8">{summary_html}</ul>
  </div>
</details>"""


def bar_chart_html(data: list, color: str = "#00629B", max_width: int = 280) -> str:
    if not data:
        return ""
    max_val = max(v for _, v in data)
    rows = ""
    for label, val in data:
        w = int((val / max_val) * max_width) if max_val else 0
        rows += (
            f'<tr>'
            f'<td style="width:190px;white-space:nowrap;font-size:.84em;padding:4px 8px 4px 0">{escape(str(label))}</td>'
            f'<td style="padding:4px 0">'
            f'<div style="background:{color};height:16px;width:{w}px;border-radius:3px;display:inline-block;vertical-align:middle"></div>'
            f'<span style="margin-left:8px;font-size:.84em;font-weight:700">{val:,}</span>'
            f'</td></tr>'
        )
    return f'<table style="border:none;margin-bottom:0">{rows}</table>'


def donut_html(data: list) -> str:
    if not data:
        return ""
    total = sum(v for _, v in data)
    if not total:
        return ""
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6", "#1abc9c"]
    stops, cum = [], 0
    for i, (_, val) in enumerate(data):
        s = cum / total * 100
        cum += val
        e = cum / total * 100
        stops.append(f"{colors[i % len(colors)]} {s:.1f}% {e:.1f}%")
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">'
        f'<div style="width:11px;height:11px;background:{colors[i % len(colors)]};border-radius:2px;flex-shrink:0"></div>'
        f'<span style="font-size:.82em">{escape(str(lbl))}: <b>{val:,}</b> ({val/total*100:.1f}%)</span></div>'
        for i, (lbl, val) in enumerate(data)
    )
    gradient = ", ".join(stops)
    return (
        f'<div style="display:flex;align-items:center;gap:20px">'
        f'<div style="position:relative;flex-shrink:0">'
        f'<div style="width:110px;height:110px;border-radius:50%;background:conic-gradient({gradient})"></div>'
        f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        f'width:56px;height:56px;background:white;border-radius:50%;display:flex;align-items:center;'
        f'justify-content:center;font-size:.72em;font-weight:800;color:#333;text-align:center">'
        f'{total:,}<br>total</div></div>'
        f'<div>{legend}</div></div>'
    )


def heat_class(val: int, max_val: int) -> str:
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


# ─── HTML builder ─────────────────────────────────────────────────────────────

def build_html(qairt_id: str, data: dict, llm: dict, obj: dict) -> str:
    from datetime import date as _date

    today = _date.today().isoformat()

    d = data  # shorthand

    top_socs = sorted(d["soc_failure_map"].items(), key=lambda x: -x[1])[:10]
    top_runtimes = sorted(d["runtime_failure_map"].items(), key=lambda x: -x[1])
    top_types = sorted(d["type_failure_map"].items(), key=lambda x: -x[1])
    top_clusters = sorted(d["cluster_name_map"].items(), key=lambda x: -x[1])[:10]
    top_models = sorted(d["model_failure_map"].items(), key=lambda x: -x[1])[:10]
    top_cluster_classes = sorted(d["cluster_class_map"].items(), key=lambda x: -x[1])
    top_dsps = sorted(d["dsp_type_map"].items(), key=lambda x: -x[1])

    all_heatmap_vals = [d["heatmap"][bu][rt] for bu in BU_ORDER for rt in ALL_RUNTIMES_ORDERED]
    max_heat = max(all_heatmap_vals) if all_heatmap_vals else 1

    html = (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>QAIRT Analysis Report - {qairt_id}</title>'
        f'{REPORT_CSS}</head><body>'
    )

    # ── Header ────────────────────────────────────────────────────────────
    html += f"""
<div class="report-header">
  <h1>&#128202; QAIRT Analysis Report</h1>
  <div class="sub">{qairt_id}</div>
  <div class="meta">
    <span>&#128197; Generated: {today}</span>
    <span>&#128279; Run IDs: {d['total_run_ids']}</span>
    <span>&#128268; Gerrits: {d['total_gerrits']}</span>
    <span>&#9888; Total Failures: {d['total_failures']:,}</span>
  </div>
</div>"""

    # ── Nav ───────────────────────────────────────────────────────────────
    html += """<div class="toc">
  <a href="#overview">Overview</a>
  <a href="#bu">BU Breakdown</a>
  <a href="#heatmap">Heatmap</a>
  <a href="#soc">SOC Analysis</a>
  <a href="#runtime">Runtime Analysis</a>
  <a href="#clusters">Cluster Analysis</a>
  <a href="#models">Model Analysis</a>
  <a href="#dsp">DSP Analysis</a>
  <a href="#gerrits">Gerrits</a>
  <a href="#runids">Run IDs</a>
</div>
<div class="container">"""

    # ── Section 1: Overview + Executive LLM Summary ───────────────────────
    exec_summary = llm.get("executive", "")
    html += f"""
<div id="overview" class="section">
  <div class="section-title">&#128202; Executive Overview</div>
  <div class="kpi-grid">
    <div class="kpi d"><div class="lbl">Total Failures</div><div class="val">{d['total_failures']:,}</div><div class="sub">Across all run IDs</div></div>
    <div class="kpi w"><div class="lbl">Models Failed</div><div class="val">{d['total_models_failed']:,}</div><div class="sub">{d['total_model_failure_entries']:,} failure entries</div></div>
    <div class="kpi"><div class="lbl">Run IDs Processed</div><div class="val">{d['total_run_ids']}</div><div class="sub">{len(d['bu_stats'])} Business Units</div></div>
    <div class="kpi s"><div class="lbl">Gerrits Merged</div><div class="val">{d['total_gerrits']}</div><div class="sub">Unique commits</div></div>
    <div class="kpi p"><div class="lbl">SOCs Impacted</div><div class="val">{len(d['soc_failure_map'])}</div><div class="sub">{len(d['runtime_failure_map'])} runtimes tested</div></div>
    <div class="kpi w"><div class="lbl">Failure Clusters</div><div class="val">{len(d['cluster_name_map'])}</div><div class="sub">Unique issue patterns</div></div>
  </div>"""

    if not _is_empty_summary(exec_summary):
        html += f"""
  <div class="llm-section" style="margin-top:20px">
    <h4>&#129302; LLM Executive Summary
      <span style="font-size:.8em;color:#888;font-weight:400">&mdash; Gemini 2.5 Pro analysis across all {d['total_run_ids']} run IDs</span>
    </h4>
    <ul>{exec_summary}</ul>
  </div>"""
    else:
        html += '<p style="color:#aaa;font-size:.82em;font-style:italic;margin-top:16px">&#129302; No LLM executive summary available.</p>'

    html += "</div>"

    # ── Section 2: BU Breakdown ───────────────────────────────────────────
    html += '<div id="bu" class="section"><div class="section-title">&#127970; Business Unit Breakdown</div><div class="bu-grid">'

    for bu in BU_ORDER:
        if bu not in d["bu_stats"]:
            continue
        stats = d["bu_stats"][bu]
        color = BU_COLORS.get(bu, "#95a5a6")
        top3c = sorted(stats["top_clusters"].items(), key=lambda x: -x[1])[:3]
        top3m = sorted(stats["top_models"].items(), key=lambda x: -x[1])[:3]
        cc_counts = dict(stats["cluster_classes"])
        sdk_cnt = cc_counts.get("sdk_issue", 0)
        env_cnt = cc_counts.get("env_issue", 0)
        setup_cnt = cc_counts.get("setup_issue", 0)
        bu_summary = llm.get("bu", {}).get(bu, "")
        has_summary = not _is_empty_summary(bu_summary)

        badges = (
            (f"<span class='badge bd'>SDK: {sdk_cnt}</span> " if sdk_cnt else "")
            + (f"<span class='badge bw'>Env: {env_cnt}</span> " if env_cnt else "")
            + (f"<span class='badge bs'>Setup: {setup_cnt}</span>" if setup_cnt else "")
        )
        cluster_rows = "".join(
            f'<div class="bu-row"><span class="bu-lbl" style="font-size:.82em">{escape(c)}</span>'
            f'<span class="bu-val">{n:,}</span></div>'
            for c, n in top3c
        )
        model_rows = "".join(
            f'<div class="bu-row"><span class="bu-lbl" style="font-size:.82em">{escape(m)}</span>'
            f'<span class="bu-val">{n:,}</span></div>'
            for m, n in top3m
        )
        llm_block = (
            llm_box(bu_summary, f"LLM Analysis &mdash; {bu}", collapsed=False)
            if has_summary
            else '<p style="color:#aaa;font-size:.82em;font-style:italic;margin-top:10px">&#129302; No LLM summary (all errors filtered as infra noise)</p>'
        )

        html += f"""
    <div class="bu-card">
      <div class="bu-hdr" style="background:{color}">
        <span>{bu}</span>
        <span style="font-size:1.35em;font-weight:800">{stats['total_failures']:,}</span>
      </div>
      <div class="bu-body">
        <div class="bu-row"><span class="bu-lbl">Run IDs</span><span class="bu-val">{len(stats['run_ids'])}</span></div>
        <div class="bu-row"><span class="bu-lbl">Models Failed</span><span class="bu-val">{stats['model_failures']:,}</span></div>
        <div class="bu-row"><span class="bu-lbl">Failure Entries</span><span class="bu-val">{stats['model_failure_entries']:,}</span></div>
        <div class="bu-row"><span class="bu-lbl">SOCs Impacted</span><span class="bu-val">{len(stats['socs'])}</span></div>
        <div class="bu-row"><span class="bu-lbl">Runtimes</span><span class="bu-val" style="font-size:.82em">{", ".join(sorted(stats['runtimes']))}</span></div>
        <div class="bu-row"><span class="bu-lbl">Classification</span><span>{badges}</span></div>
        <div style="margin-top:10px;font-size:.78em;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px">Top Clusters</div>
        {cluster_rows}
        <div style="margin-top:8px;font-size:.78em;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px">Top Failing Models</div>
        {model_rows}
        {llm_block}
      </div>
    </div>"""

    html += "</div></div>"

    # ── Section 3: Heatmap ────────────────────────────────────────────────
    html += """
<div id="heatmap" class="section">
  <div class="section-title">&#128293; Failure Heatmap: BU &times; Runtime</div>
  <p style="color:var(--muted);font-size:.86em;margin-top:-10px">Cell values = total failure count. Color intensity = severity.</p>
  <div class="tw"><table class="heatmap-table"><tr><th>BU / Runtime</th>"""
    for rt in ALL_RUNTIMES_ORDERED:
        html += f"<th>{rt.upper()}</th>"
    html += "<th>TOTAL</th></tr>"
    for bu in BU_ORDER:
        if bu not in d["bu_stats"]:
            continue
        row_total = sum(d["heatmap"][bu][rt] for rt in ALL_RUNTIMES_ORDERED)
        html += f'<tr><td style="font-weight:700;background:{BU_COLORS.get(bu,"#95a5a6")};color:#fff">{bu}</td>'
        for rt in ALL_RUNTIMES_ORDERED:
            val = d["heatmap"][bu][rt]
            html += f'<td class="{heat_class(val, max_heat)}">{val:,}</td>'
        html += f'<td style="font-weight:800;background:#f8f9fa">{row_total:,}</td></tr>'
    html += '<tr style="background:#f0f2f5"><td style="font-weight:700">TOTAL</td>'
    for rt in ALL_RUNTIMES_ORDERED:
        ct = sum(d["heatmap"][bu][rt] for bu in BU_ORDER)
        html += f'<td style="font-weight:700">{ct:,}</td>'
    gt = sum(sum(d["heatmap"][bu][rt] for rt in ALL_RUNTIMES_ORDERED) for bu in BU_ORDER)
    html += f'<td style="font-weight:800">{gt:,}</td></tr></table></div>'
    html += '<div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap"><span style="font-size:.8em;color:var(--muted)">Intensity:</span>'
    for cls, lbl in [("heat-0","0"),("heat-1","Low"),("heat-2","Medium"),("heat-3","High"),("heat-4","Very High"),("heat-5","Critical")]:
        html += f'<span class="badge {cls}" style="border-radius:4px">{lbl}</span>'
    html += "</div></div>"

    # ── Section 4: SOC Analysis ───────────────────────────────────────────
    total_soc_f = sum(v for _, v in top_socs)
    html += f"""
<div id="soc" class="section">
  <div class="section-title">&#128241; SOC Analysis</div>
  <div class="two-col">
    <div>
      <h4 style="margin-bottom:10px">Top 10 SOCs by Failure Count</h4>
      {bar_chart_html(top_socs, color="#e74c3c")}
    </div>
    <div><div class="tw"><table>
      <tr><th>SOC Name</th><th>Failures</th><th>Share</th></tr>"""
    for soc, cnt in top_socs:
        pct = cnt / total_soc_f * 100 if total_soc_f else 0
        bar = f'<div style="background:#e74c3c;height:6px;border-radius:3px;width:{int(pct)}%;display:inline-block"></div>'
        html += f'<tr><td><b>{escape(soc)}</b></td><td>{cnt:,}</td><td style="min-width:110px">{bar}<small style="margin-left:4px">{pct:.1f}%</small></td></tr>'
    html += "</table></div></div></div></div>"

    # ── Section 5: Runtime Analysis ───────────────────────────────────────
    total_rt_f = sum(v for _, v in top_runtimes)
    html += f"""
<div id="runtime" class="section">
  <div class="section-title">&#9881; Runtime &amp; Test Type Analysis</div>
  <div class="two-col">
    <div><h4 style="margin-bottom:10px">Failures by Runtime</h4>{bar_chart_html(top_runtimes, color="#2980b9")}</div>
    <div><h4 style="margin-bottom:10px">Failures by Test Type</h4>{bar_chart_html(top_types, color="#8e44ad")}</div>
  </div>
  <div class="tw" style="margin-top:16px"><table>
    <tr><th>Runtime</th><th>Failures</th><th>% of Total</th><th>Distribution</th></tr>"""
    for rt, cnt in top_runtimes:
        pct = cnt / total_rt_f * 100 if total_rt_f else 0
        bar = f'<div style="background:#2980b9;height:8px;border-radius:4px;width:{int(pct)}%;display:inline-block"></div>'
        html += f'<tr><td><b>{rt.upper()}</b></td><td>{cnt:,}</td><td>{pct:.1f}%</td><td style="min-width:180px">{bar}</td></tr>'
    html += "</table></div></div>"

    # ── Section 6: Cluster Analysis + LLM per-class ───────────────────────
    class_meta = {
        "sdk_issue":   ("bd", "&#128308; SDK Issues",   "SDK / Product bug — needs engineering fix"),
        "env_issue":   ("bw", "&#128993; Env Issues",   "Environment / infra issue — test setup problem"),
        "setup_issue": ("bs", "&#128994; Setup Issues", "Test configuration issue — script/config error"),
    }
    max_c = top_clusters[0][1] if top_clusters else 1
    html += """
<div id="clusters" class="section">
  <div class="section-title">&#128202; Failure Cluster Analysis</div>
  <div class="two-col">
    <div>
      <h4 style="margin-bottom:10px">Top 10 Failure Clusters</h4>
      <div class="tw"><table><tr><th>Cluster Name</th><th>Count</th><th>Bar</th></tr>"""
    for cluster, cnt in top_clusters:
        bw = int(cnt / max_c * 200)
        html += f'<tr><td style="font-size:.84em">{escape(cluster)}</td><td><b>{cnt:,}</b></td><td><div style="background:#e74c3c;height:8px;border-radius:4px;width:{bw}px;display:inline-block"></div></td></tr>'
    html += f"""</table></div>
    </div>
    <div>
      <h4 style="margin-bottom:10px">Failure Classification</h4>
      {donut_html(top_cluster_classes)}
      <div class="tw" style="margin-top:14px"><table>
        <tr><th>Class</th><th>Count</th><th>Meaning</th></tr>"""
    for cls, cnt in top_cluster_classes:
        badge_cls, label, meaning = class_meta.get(cls, ("bi", cls, "Other"))
        html += f'<tr><td><span class="badge {badge_cls}">{escape(cls)}</span></td><td><b>{cnt:,}</b></td><td style="font-size:.83em;color:var(--muted)">{meaning}</td></tr>'
    html += """</table></div>
    </div>
  </div>
  <div style="margin-top:20px">
    <h4 style="color:#444;margin-bottom:12px">&#129302; LLM Analysis by Failure Class</h4>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px">"""
    for cc, cnt in top_cluster_classes:
        badge_cls, label, _ = class_meta.get(cc, ("bi", cc, ""))
        summary = llm.get("cluster_class", {}).get(cc, "")
        has_s = not _is_empty_summary(summary)
        content = (
            f'<ul style="margin:0;padding-left:18px;font-size:.86em;line-height:1.8">{summary}</ul>'
            if has_s
            else '<p style="color:#aaa;font-size:.82em;font-style:italic;margin:0">No LLM summary (errors filtered as infra noise)</p>'
        )
        html += f"""
      <div style="background:#fafcff;border:1px solid #cce0f5;border-radius:8px;padding:16px 18px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
          <span class="badge {badge_cls}" style="font-size:.82em">{label}</span>
          <span style="font-size:.82em;color:var(--muted)">{cnt:,} failures</span>
        </div>
        {content}
      </div>"""
    html += "</div></div></div>"

    # ── Section 7: Model Analysis ─────────────────────────────────────────
    max_m = top_models[0][1] if top_models else 1
    html += f"""
<div id="models" class="section">
  <div class="section-title">&#129302; Model Failure Analysis</div>
  <div class="tw"><table>
    <tr><th>#</th><th>Model Name</th><th>Failure Entries</th><th>Severity</th><th>Distribution</th></tr>"""
    for i, (model, cnt) in enumerate(top_models, 1):
        bw = int(cnt / max_m * 240)
        sev = (
            "<span class='badge bd'>Critical</span>" if cnt > 200
            else "<span class='badge bw'>High</span>" if cnt > 50
            else "<span class='badge bi'>Medium</span>"
        )
        html += f'<tr><td style="color:var(--muted)">{i}</td><td style="font-family:monospace;font-size:.83em">{escape(model)}</td><td><b>{cnt:,}</b></td><td>{sev}</td><td><div style="background:#8e44ad;height:8px;border-radius:4px;width:{bw}px;display:inline-block"></div></td></tr>'
    html += f"""</table></div>
  <p style="color:var(--muted);font-size:.83em;margin-top:10px">
    Showing top {len(top_models)} of {d['total_models_failed']:,} total failing models.
    Total failure entries: <b>{d['total_model_failure_entries']:,}</b>
  </p>
</div>"""

    # ── Section 8: DSP Analysis ───────────────────────────────────────────
    total_dsp = sum(v for _, v in top_dsps)
    html += f"""
<div id="dsp" class="section">
  <div class="section-title">&#128187; DSP Type Analysis</div>
  <div class="two-col">
    <div><h4 style="margin-bottom:10px">Failures by DSP Type</h4>{bar_chart_html(top_dsps, color="#1abc9c")}</div>
    <div><div class="tw"><table>
      <tr><th>DSP Type</th><th>Failures</th><th>% Share</th></tr>"""
    for dsp, cnt in top_dsps:
        pct = cnt / total_dsp * 100 if total_dsp else 0
        html += f'<tr><td><b>{escape(dsp)}</b></td><td>{cnt:,}</td><td>{pct:.1f}%</td></tr>'
    html += f'<tr style="background:#f8f9fa"><td><b>Total</b></td><td><b>{total_dsp:,}</b></td><td>100%</td></tr>'
    html += "</table></div></div></div></div>"

    # ── Section 9: Gerrits ────────────────────────────────────────────────
    repo_gerrits: dict = defaultdict(list)
    for g in d["gerrit_list"]:
        repo_gerrits[g.get("repository_name", "Unknown")].append(g)
    html += f'<div id="gerrits" class="section"><div class="section-title">&#128268; Gerrits Merged ({d["total_gerrits"]} unique)</div>'
    for repo, gerrits in sorted(repo_gerrits.items()):
        html += f'<h4 style="color:#444;margin-bottom:8px">&#128193; {escape(repo)} <span class="badge bi">{len(gerrits)}</span></h4>'
        html += '<div class="tw" style="margin-bottom:14px"><table><tr><th>Commit</th><th>Author</th><th>Reviewers</th><th>Approved By</th></tr>'
        for g in gerrits:
            url = g.get("commit_url", "")
            msg = g.get("commit_message", "")[:80]
            author = (g.get("gerrit_raised_by") or [{}])[0].get("name", "N/A")
            reviewers = ", ".join(r.get("name", "") for r in g.get("gerrit_reviewed_by", []))
            approvers = ", ".join(a.get("name", "") for a in g.get("gerrit_approved_by", []))
            link = f'<a href="{escape(url)}" target="_blank">{escape(msg)}</a>' if url else escape(msg)
            html += f'<tr><td style="font-size:.83em">{link}</td><td style="font-size:.83em">{escape(author)}</td><td style="font-size:.8em;color:var(--muted)">{escape(reviewers)}</td><td style="font-size:.8em;color:var(--muted)">{escape(approvers)}</td></tr>'
        html += "</table></div>"
    html += "</div>"

    # ── Section 10: Run ID Details ────────────────────────────────────────
    html += '<div id="runids" class="section"><div class="section-title">&#128196; Run ID Details</div><div class="tw"><table>'
    html += "<tr><th>Run ID</th><th>BU</th><th>Type Failures</th><th>Models Failed</th><th>SOCs</th><th>Runtimes</th><th>Gerrits</th></tr>"
    for run_id, run_obj in obj.items():
        if not run_obj:
            continue
        rd = run_obj.regression_data
        bu = classify_run_id(run_id)
        bc = BU_COLORS.get(bu, "#95a5a6")
        if not rd or rd.get("status") != 200:
            html += f'<tr><td style="font-family:monospace;font-size:.8em">{escape(run_id)}</td><td><span class="badge" style="background:{bc};color:#fff">{bu}</span></td><td colspan="5" style="color:var(--muted);font-style:italic">No data</td></tr>'
            continue
        tc = sum(
            sum(len(v) for v in clusters.values() if isinstance(v, list))
            for t, runtimes in rd.get("type", {}).items()
            if isinstance(runtimes, dict)
            for rt, clusters in runtimes.items()
            if isinstance(clusters, dict)
        )
        mc = len(rd.get("model", {}))
        socs_s: set = set()
        rts_s: set = set()
        for md in rd.get("model", {}).values():
            for e in md:
                if e.get("soc_name"):
                    socs_s.add(e["soc_name"])
                if e.get("runtime"):
                    rts_s.add(e["runtime"])
        gc = 0
        seen: set = set()
        for _, rd2 in rd.get("gerrit_info", {}).items():
            if isinstance(rd2, dict):
                for _, ge in rd2.items():
                    if isinstance(ge, list):
                        for g in ge:
                            u = g.get("commit_url", "")
                            if u and u not in seen:
                                seen.add(u)
                                gc += 1
        soc_str = ", ".join(sorted(socs_s)[:3]) + ("..." if len(socs_s) > 3 else "")
        html += (
            f'<tr>'
            f'<td style="font-family:monospace;font-size:.79em">{escape(run_id)}</td>'
            f'<td><span class="badge" style="background:{bc};color:#fff;font-size:.76em">{bu}</span></td>'
            f'<td><b>{tc:,}</b></td>'
            f'<td><b>{mc:,}</b></td>'
            f'<td style="font-size:.8em">{soc_str}</td>'
            f'<td style="font-size:.8em">{", ".join(sorted(rts_s))}</td>'
            f'<td>{gc}</td>'
            f'</tr>'
        )
    html += "</table></div></div>"

    # ── Footer ────────────────────────────────────────────────────────────
    html += (
        f'<div class="footer">'
        f"Generated by QAIRT Enhanced Analysis Pipeline &bull; {qairt_id} &bull; {today}<br>"
        f"<small>LLM summaries powered by Gemini 2.5 Pro via QGenie</small>"
        f"</div>"
        f"</div></body></html>"
    )

    return html


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Generate enhanced QAIRT analysis HTML report")
    parser.add_argument("--qairt_id", required=True, help="e.g. qaisw-v2.46.0.260319041023_nightly")
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip all LLM calls — produce a metrics-only report",
    )
    parser.add_argument(
        "--cache_llm",
        action="store_true",
        help="Cache LLM summaries to JSON next to the report so re-runs are instant",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <reports_path>/<qairt_id>/<qairt_id>_enhanced.html)",
    )
    args = parser.parse_args()

    qairt_id = args.qairt_id

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(CONSOLIDATED_REPORTS_PATH, qairt_id, f"{qairt_id}_enhanced.html")

    cache_path = None
    if args.cache_llm:
        cache_path = os.path.join(
            CONSOLIDATED_REPORTS_PATH, qairt_id, "regression_artifacts", f"{qairt_id}_llm_summaries.json"
        )

    logger.info("Loading joblib artifact for %s ...", qairt_id)
    obj = _load_joblib(qairt_id)

    logger.info("Aggregating metrics ...")
    data = aggregate(obj)

    if args.no_llm:
        logger.info("--no_llm set, skipping LLM calls")
        llm: dict = {"executive": "", "bu": {}, "cluster_class": {}}
    else:
        logger.info("Generating LLM summaries ...")
        llm = generate_llm_summaries(data, cache_path=cache_path)

    logger.info("Building HTML ...")
    html = build_html(qairt_id, data, llm, obj)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    logger.info("Report written to: %s  (%d bytes)", out_path, len(html))
    print(f"\nDone. Report: {out_path}")


if __name__ == "__main__":
    main()
