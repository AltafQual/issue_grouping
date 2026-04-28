"""Teams notification helpers using Power Automate Workflow webhooks.

Setup (one-time, done in Teams UI)
-----------------------------------
1. Go to the target channel → ... → Workflows
2. Search "Send webhook alerts to a channel" → configure → Save
3. Copy the generated webhook URL
4. Set env var: TEAMS_WEBHOOK_URL=https://prod-XX.westus.logic.azure.com/...

Two cards are sent per analysis run
------------------------------------
    send_teams_summary_card(webhook_url, runs)    -- high-level overview
    send_teams_breakdown_card(webhook_url, runs)  -- per-type FAIL/PF/NR detail

Both post an Adaptive Card (v1.4) via httpx.AsyncClient.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from src.logger import AppLogger
from src.stability_report import RunAnalysis

logger = AppLogger().get_logger(__name__)

_ACCENT_RED = "attention"
_ACCENT_GREEN = "good"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _flag_color(has_flags: bool) -> str:
    return _ACCENT_RED if has_flags else _ACCENT_GREEN


def _build_summary_card(runs: list[RunAnalysis]) -> dict[str, Any]:
    """Adaptive Card — overview of all runs in this batch."""
    total_tests = sum(r.total_tests for r in runs)
    total_failed = sum(r.total_failed for r in runs)
    total_flagged = sum(len(r.flagged_types) for r in runs)
    has_flags = total_flagged > 0

    run_id_facts = [{"title": "", "value": r.run_id} for r in runs]

    body = [
        {
            "type": "TextBlock",
            "text": f"{'🔴' if has_flags else '✅'} Nightly Stability Alert — {len(runs)} run(s) analysed",
            "weight": "Bolder",
            "size": "Medium",
            "color": _flag_color(has_flags),
            "wrap": True,
        },
        {"type": "TextBlock", "text": _utc_now(), "size": "Small", "isSubtle": True, "spacing": "None"},
        {"type": "TextBlock", "text": "**Run IDs**", "weight": "Bolder", "spacing": "Medium"},
        {"type": "FactSet", "facts": run_id_facts},
        {
            "type": "ColumnSet",
            "spacing": "Medium",
            "columns": [
                {
                    "type": "Column",
                    "width": "stretch",
                    "items": [
                        {"type": "TextBlock", "text": "Total Tests", "weight": "Bolder", "size": "Small"},
                        {"type": "TextBlock", "text": str(total_tests), "size": "Large", "weight": "Bolder"},
                    ],
                },
                {
                    "type": "Column",
                    "width": "stretch",
                    "items": [
                        {"type": "TextBlock", "text": "FAIL", "weight": "Bolder", "size": "Small"},
                        {
                            "type": "TextBlock",
                            "text": str(total_failed),
                            "size": "Large",
                            "weight": "Bolder",
                            "color": _ACCENT_RED if total_failed else _ACCENT_GREEN,
                        },
                    ],
                },
                {
                    "type": "Column",
                    "width": "stretch",
                    "items": [
                        {"type": "TextBlock", "text": "Flagged Types", "weight": "Bolder", "size": "Small"},
                        {
                            "type": "TextBlock",
                            "text": str(total_flagged),
                            "size": "Large",
                            "weight": "Bolder",
                            "color": _flag_color(has_flags),
                        },
                    ],
                },
            ],
        },
    ]

    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }


def _build_breakdown_card(runs: list[RunAnalysis]) -> dict[str, Any]:
    """Adaptive Card — per-type FAIL / PARENT_FAIL / NOT_RUN breakdown for each run."""
    body: list[dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": "Per-Type Breakdown",
            "weight": "Bolder",
            "size": "Medium",
            "spacing": "None",
        },
        {"type": "TextBlock", "text": _utc_now(), "size": "Small", "isSubtle": True, "spacing": "None"},
    ]

    for run in runs:
        # Run ID header
        body.append(
            {
                "type": "TextBlock",
                "text": f"**{run.run_id}**",
                "weight": "Bolder",
                "spacing": "Medium",
                "color": _flag_color(run.has_flags),
            }
        )

        if not run.type_stats:
            body.append({"type": "TextBlock", "text": "No data", "isSubtle": True, "spacing": "Small"})
            continue

        # Column headers
        body.append(
            {
                "type": "ColumnSet",
                "spacing": "Small",
                "columns": [
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [{"type": "TextBlock", "text": "**Type**", "size": "Small", "weight": "Bolder"}],
                    },
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [{"type": "TextBlock", "text": "**FAIL%**", "size": "Small", "weight": "Bolder"}],
                    },
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [{"type": "TextBlock", "text": "**FAIL**", "size": "Small", "weight": "Bolder"}],
                    },
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [{"type": "TextBlock", "text": "**PF**", "size": "Small", "weight": "Bolder"}],
                    },
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [{"type": "TextBlock", "text": "**NR**", "size": "Small", "weight": "Bolder"}],
                    },
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [{"type": "TextBlock", "text": "**Status**", "size": "Small", "weight": "Bolder"}],
                    },
                ],
            }
        )

        sorted_stats = sorted(run.type_stats.values(), key=lambda s: (-int(s.highlighted), -s.failure_rate))
        for s in sorted_stats:
            pct_str = f"{s.failure_pct:.0f}%" if s.effective_total > 0 else "N/A"
            status = "🔴 HIGH" if s.highlighted else "✅ OK"
            color = _ACCENT_RED if s.highlighted else _ACCENT_GREEN
            body.append(
                {
                    "type": "ColumnSet",
                    "spacing": "None",
                    "columns": [
                        {
                            "type": "Column",
                            "width": "stretch",
                            "items": [{"type": "TextBlock", "text": s.test_type, "size": "Small", "wrap": True}],
                        },
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [{"type": "TextBlock", "text": pct_str, "size": "Small", "color": color}],
                        },
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [{"type": "TextBlock", "text": str(s.failed), "size": "Small"}],
                        },
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [
                                {"type": "TextBlock", "text": str(s.parent_fail), "size": "Small", "isSubtle": True}
                            ],
                        },
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [{"type": "TextBlock", "text": str(s.not_run), "size": "Small", "isSubtle": True}],
                        },
                        {
                            "type": "Column",
                            "width": "auto",
                            "items": [{"type": "TextBlock", "text": status, "size": "Small", "color": color}],
                        },
                    ],
                }
            )

    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }


async def _post_card(webhook_url: str, payload: dict[str, Any], label: str) -> None:
    """POST an Adaptive Card payload to the Teams webhook URL."""
    try:
        async with httpx.AsyncClient(verify=False, timeout=15) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        logger.info(f"Teams {label} card sent (status={resp.status_code})")
    except Exception as exc:
        logger.exception(f"Failed to send Teams {label} card: {exc}")


async def send_teams_summary_card(webhook_url: str, runs: list[RunAnalysis]) -> None:
    """Send the high-level overview card to Teams."""
    await _post_card(webhook_url, _build_summary_card(runs), "summary")


async def send_teams_breakdown_card(webhook_url: str, runs: list[RunAnalysis]) -> None:
    """Send the per-type FAIL/PARENT_FAIL/NOT_RUN breakdown card to Teams."""
    await _post_card(webhook_url, _build_breakdown_card(runs), "breakdown")


if __name__ == "__main__":
    import asyncio
    import json
    import random
    import sys

    import pandas as pd

    from src.stability_report import analyze_type_failures

    # ── build synthetic data (same seed as stability_report.__main__) ──────────
    random.seed(42)
    TYPES = ["converter", "quantizer", "savecontext", "htp", "gpu", "cpu", "lpai"]
    SOCS = ["NordLE", "Kailua", "Lanai"]
    HOSTS = {
        "NordLE": ["hydciqlab01", "hydciqlab02", "hydciqlab03"],
        "Kailua": ["hydciqlab04", "hydciqlab05"],
        "Lanai": ["hydciqlab06"],
    }
    RESULTS = ["PASS", "FAIL", "PARENT_FAIL", "NOT_RUN"]
    RESULT_WEIGHTS = [0.55, 0.28, 0.10, 0.07]

    def _make_df(run_id: str) -> pd.DataFrame:
        rows = []
        for test_type in TYPES:
            for _ in range(random.randint(40, 100)):
                soc = random.choice(SOCS)
                rows.append(
                    {
                        "type": test_type,
                        "result": random.choices(RESULTS, RESULT_WEIGHTS)[0],
                        "soc_name": soc,
                        "host": random.choice(HOSTS[soc]),
                        "reason": "Error: segfault in conv2d" if random.random() < 0.3 else "",
                    }
                )
        return pd.DataFrame(rows)

    run_ids = ["QNN-auto-v2.47-260428001", "QNN-auto-v2.47-260428002"]
    runs: list[RunAnalysis] = [
        RunAnalysis(
            run_id=rid,
            job_info={"status": "RUNNING", "start_time": "2026-04-28 00:15:00"},
            type_stats=analyze_type_failures(_make_df(rid)),
        )
        for rid in run_ids
    ]

    # ── print card payloads as pretty JSON ─────────────────────────────────────
    summary_payload = _build_summary_card(runs)
    breakdown_payload = _build_breakdown_card(runs)

    print("=" * 70)
    print("SUMMARY CARD PAYLOAD")
    print("=" * 70)
    print(json.dumps(summary_payload, indent=2))

    print("\n" + "=" * 70)
    print("BREAKDOWN CARD PAYLOAD")
    print("=" * 70)
    print(json.dumps(breakdown_payload, indent=2))

    # ── live POST to dummy URL (expects graceful failure) ──────────────────────
    dummy_url = sys.argv[1] if len(sys.argv) > 1 else "https://httpbin.org/post"
    print(f"\n{'=' * 70}")
    print(f"POST target: {dummy_url}")
    print("=" * 70)

    async def _run_test() -> None:
        await send_teams_summary_card(dummy_url, runs)
        await send_teams_breakdown_card(dummy_url, runs)

    asyncio.run(_run_test())
    print("\nDone.")
