"""HTML rendering utilities — shared across all report types.

Provides :class:`HTMLRenderer` with static methods for pure HTML generation
and the :data:`REPORT_CSS` stylesheet constant used by all consolidated reports.

Layering
--------
No imports from other ``src.*`` sub-packages.  Only standard-library modules
are used so this module can be imported by any layer without introducing
circular dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape

__all__ = ["HTMLRenderer", "REPORT_CSS"]

REPORT_CSS = """
<style>
    :root {
        --primary: #00629B; /* Qualcomm Blue approx or dark blue */
        --secondary: #3253DC;
        --bg-body: #f5f7fa;
        --bg-container: #ffffff;
        --text-main: #333333;
        --text-muted: #6c757d;
        --border-color: #e9ecef;
        --table-head-bg: #00629B;
        --table-head-text: #ffffff;
        --row-hover: #f1faff;
        --accent-danger: #dc3545;
        --accent-success: #28a745;
        --shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    body {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background-color: var(--bg-body);
        color: var(--text-main);
        margin: 0;
        padding: 20px;
        line-height: 1.6;
    }

    .container {
        max-width: 100%;
        margin: 0 auto;
        background-color: var(--bg-container);
        padding: 40px;
        border-radius: 8px;
        box-shadow: var(--shadow);
    }

    h1, h2, h3, h4 {
        color: var(--primary);
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.75em;
    }

    h1 { font-size: 2.2em; border-bottom: 3px solid var(--primary); padding-bottom: 10px; margin-top: 0; }
    h2 { font-size: 1.8em; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
    h3 { font-size: 1.4em; color: #444; }

    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
        background-color: white;
        border: 1px solid var(--border-color);
    }

    th, td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.95em;
        vertical-align: top;
        word-wrap: break-word;
        overflow-wrap: normal;
    }

    th {
        background-color: var(--table-head-bg);
        color: var(--table-head-text);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85em;
    }

    tr:nth-child(even) { background-color: #f8f9fa; }
    tr:hover { background-color: var(--row-hover); }

    a { color: var(--secondary); text-decoration: none; font-weight: 500; }
    a:hover { text-decoration: underline; color: #003d73; }

    ul { margin: 0; padding-left: 20px; }
    li { margin-bottom: 5px; }

    /* Summary Box / Dashboard Styles */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: var(--shadow);
        border-top: 4px solid var(--primary);
        text-align: center;
        border: 1px solid var(--border-color);
    }

    .card h4 { margin: 0; font-size: 0.9em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
    .card .count { font-size: 2.5em; font-weight: bold; color: var(--primary); margin: 10px 0; }

    .summary-section {
        background-color: #fff;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 30px;
    }

    /* Helper Classes */
    .text-center { text-align: center; }

    /* Specific overrides */
    .qgenie-summary .cell-content, .gerrit-cell .cell-content {
        display: flex;
        flex-direction: column;
        width: 100%;
    }

    .exec-summary th { background-color: #2c3e50; }

    .footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
        color: var(--text-muted);
        font-size: 0.9em;
        text-align: center;
    }

    /* Failure distribution charts */
    .chart-wrap { margin-bottom: 20px; }
    .chart-grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px; }
    .chart-box { background: #f8f9fa; border: 1px solid var(--border-color); border-radius: 6px; padding: 14px 16px; }
    .chart-box h4 { margin: 0 0 12px; font-size: .9em; color: var(--primary); }
    .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: .83em; }
    .bar-row .bar-label { width: 170px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
    .bar-row .bar-track { flex: 1; background: #e9ecef; border-radius: 3px; height: 13px; }
    .bar-row .bar-fill { height: 13px; border-radius: 3px; }
    .bar-row .bar-val { width: 45px; text-align: right; font-weight: 700; flex-shrink: 0; }
    .donut-wrap { display: flex; align-items: center; gap: 18px; }
    .donut-legend { font-size: .82em; line-height: 1.9; }
    .donut-legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }

    /* BU × Runtime heatmap */
    .heatmap-wrap { margin-bottom: 24px; overflow-x: auto; }
    .heatmap-wrap table { border-collapse: collapse; font-size: .88em; }
    .heatmap-wrap th { background: var(--table-head-bg); color: var(--table-head-text); padding: 8px 12px; font-size: .8em; text-transform: uppercase; letter-spacing: .5px; }
    .heatmap-wrap td { text-align: center; padding: 8px 12px; font-weight: 700; font-size: .86em; border: 1px solid var(--border-color); }
    .heat-0 { background: #f8f9fa; color: #bbb; }
    .heat-1 { background: #fff3cd; color: #856404; }
    .heat-2 { background: #ffd6a5; color: #7d4e00; }
    .heat-3 { background: #ffb3b3; color: #8b0000; }
    .heat-4 { background: #ff6b6b; color: #fff; }
    .heat-5 { background: #c0392b; color: #fff; }
    .heat-legend { display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; align-items: center; font-size: .8em; color: var(--text-muted); }
    .heat-legend span { padding: 2px 10px; border-radius: 4px; font-weight: 700; }

    /* KPI overview cards */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 28px; }
    .kpi { background: #fff; border-radius: 10px; padding: 18px; box-shadow: var(--shadow); border-left: 5px solid var(--primary); }
    .kpi.d { border-left-color: #e74c3c; }
    .kpi.w { border-left-color: #e67e22; }
    .kpi.s { border-left-color: #27ae60; }
    .kpi.p { border-left-color: #8e44ad; }
    .kpi .lbl { font-size: .76em; color: var(--text-muted); text-transform: uppercase; letter-spacing: .8px; font-weight: 600; margin-bottom: 5px; }
    .kpi .val { font-size: 2em; font-weight: 800; color: var(--primary); line-height: 1; }
    .kpi.d .val { color: #e74c3c; }
    .kpi.w .val { color: #e67e22; }
    .kpi.s .val { color: #27ae60; }
    .kpi.p .val { color: #8e44ad; }
    .kpi .sub { font-size: .76em; color: var(--text-muted); margin-top: 3px; }
    .overview-section { background: #fff; border: 1px solid var(--border-color); border-radius: 8px; padding: 22px 24px; margin-bottom: 28px; }
    .overview-section .section-title { font-size: 1.1em; font-weight: 700; color: var(--primary); margin: 0 0 16px; padding-bottom: 10px; border-bottom: 2px solid var(--border-color); }
</style>
"""


class HTMLRenderer:
    """Shared HTML rendering utilities for report generation.

    All methods are static — this class has no instance state.  Import it
    and call methods on the class directly::

        html = HTMLRenderer.bar_chart_html(data, color="#e74c3c")

    Example:
        renderer = HTMLRenderer  # alias, no instantiation needed
        html = renderer.bar_chart_html([(\"SOC A\", 12), (\"SOC B\", 7)])
    """

    @staticmethod
    def bar_chart_html(data: list, color: str = "#00629B") -> str:
        """Render a horizontal bar chart from ``[(label, count), ...]`` as HTML.

        Args:
            data: List of ``(label, count)`` tuples sorted by descending count.
            color: CSS hex color for the bar fill.  Defaults to Qualcomm blue.

        Returns:
            HTML string containing ``.bar-row`` divs, ready to embed in a
            ``.chart-box`` wrapper.  Returns ``<p><i>No data</i></p>`` when
            *data* is empty.
        """
        if not data:
            return "<p><i>No data</i></p>"
        max_val = max(v for _, v in data)
        rows = ""
        for label, val in data:
            pct = int(val / max_val * 100) if max_val else 0
            rows += (
                f'<div class="bar-row">'
                f'<span class="bar-label" title="{escape(str(label))}">{escape(str(label))}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
                f'<span class="bar-val">{val:,}</span>'
                f"</div>"
            )
        return rows

    @staticmethod
    def donut_html(data: list) -> str:
        """Render a CSS conic-gradient donut chart + legend from ``[(label, count), ...]``.

        Args:
            data: List of ``(label, count)`` tuples.

        Returns:
            HTML string containing the donut + legend wrapped in ``.donut-wrap``.
            Returns ``<p><i>No data</i></p>`` when *data* is empty or sums to zero.
        """
        if not data:
            return "<p><i>No data</i></p>"
        total = sum(v for _, v in data)
        if not total:
            return "<p><i>No data</i></p>"
        colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71"]
        stops, cum = [], 0
        for i, (_, val) in enumerate(data):
            s = cum / total * 100
            cum += val
            e = cum / total * 100
            stops.append(f"{colors[i % len(colors)]} {s:.1f}% {e:.1f}%")
        legend = "".join(
            f'<div><span class="dot" style="background:{colors[i % len(colors)]}"></span>'
            f"{escape(str(lbl))}: <b>{val:,}</b> ({val / total * 100:.1f}%)</div>"
            for i, (lbl, val) in enumerate(data)
        )
        gradient = ", ".join(stops)
        return (
            f'<div class="donut-wrap">'
            f'<div style="position:relative;flex-shrink:0">'
            f'<div style="width:110px;height:110px;border-radius:50%;background:conic-gradient({gradient})"></div>'
            f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
            f"width:56px;height:56px;background:white;border-radius:50%;display:flex;align-items:center;"
            f'justify-content:center;font-size:.72em;font-weight:800;color:#333;text-align:center">'
            f"{total:,}<br>total</div></div>"
            f'<div class="donut-legend">{legend}</div></div>'
        )

    @staticmethod
    def list_to_html_ul(items: list) -> str:
        """Convert a Python list into an HTML ``<ul><li>...</li></ul>`` block.

        Args:
            items: List of strings or HTML snippets to wrap in ``<li>`` tags.

        Returns:
            HTML string ``<ul><li>item1</li><li>item2</li>...</ul>``.
        """
        li_html = "".join(f"<li>{item}</li>" for item in items)
        return f"<ul>{li_html}</ul>"

    @staticmethod
    def runtime_gerrit_row(runtime: str, gerrit_data: dict, rows_span: int = 0) -> str:
        """Build an HTML ``<td>`` cell for a runtime's gerrit entries.

        Groups gerrit commits by repository, deduplicates by commit message,
        and wraps them in a nested ``<ul>`` list.

        Args:
            runtime: Runtime name (e.g. ``"cpu"``, ``"gpu"``, ``"tools"``).
                ``"tools"`` merges the ``quantizer`` and ``converter`` entries.
            gerrit_data: Mapping of ``{runtime: [gerrit_info_dict, ...]}``.
            rows_span: ``rowspan`` attribute value for the ``<td>`` element.

        Returns:
            HTML ``<td rowspan="{rows_span}" ...>`` string ready to embed in a
            table row.  Returns a dash cell when no gerrit entries are found.
        """
        base_html = f'<td rowspan="{rows_span}" class="gerrit-cell"><div class="cell-content">'
        if not gerrit_data:
            return base_html + "-</div></td>"

        if runtime == "tools":
            runtime_gerrits = list(gerrit_data.get("quantizer") or []) + list(gerrit_data.get("converter") or [])
        else:
            runtime_gerrits = list(gerrit_data.get(runtime) or [])

        seen_jiras: dict[str, set] = defaultdict(set)
        items_html: list[str] = []
        repository_based_filteration: dict[str, list] = defaultdict(list)

        for gerrit_info in runtime_gerrits:
            repo = (gerrit_info.get("repository_name") or "").strip()
            repository_based_filteration[repo].append(gerrit_info)

        for repo_name, repo_data in repository_based_filteration.items():
            repo_key = (repo_name or "").lower()
            repo_esc = escape(repo_name or "", quote=True)

            inner_li: list[str] = []
            for data in repo_data:
                url = (data.get("commit_url") or "").strip()
                msg = (data.get("commit_message") or "").strip()

                key = msg.lower()
                if key and key not in seen_jiras[repo_key]:
                    msg_esc = escape(msg, quote=True)
                    url_esc = escape(url, quote=True) if url else ""
                    if url_esc:
                        inner_li.append(f'<li><a href="{url_esc}">{msg_esc}</a></li>')
                    else:
                        inner_li.append(f"<li>{msg_esc}</li>")
                    seen_jiras[repo_key].add(key)

            if inner_li:
                items_html.append(f"<li><b>{repo_esc}</b><ul>{''.join(inner_li)}</ul></li>")

        if not items_html:
            return base_html + "-</div></td>"

        return base_html + f"<ul>{''.join(items_html)}</ul>" + "</div>" + "</td>"
