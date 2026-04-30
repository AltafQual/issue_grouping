"""Microsoft Teams notifier via Power Automate Workflow webhooks.

Provides :class:`TeamsNotifier` plus the low-level Adaptive Card helpers
``send_teams_summary_card`` and ``send_teams_breakdown_card``.

Setup (one-time, in Teams UI)
------------------------------
1. Go to target channel → ``...`` → Workflows
2. Search "Send webhook alerts to a channel" → configure → Save
3. Copy the generated webhook URL
4. Set ``TEAMS_WEBHOOK_URL`` environment variable

Layering
--------
This module imports from ``src.monitoring.hourly_report``,
``src.core.interfaces``, ``src.constants``, and ``src.logger``.
No imports from ``src.teams_helpers``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from src.constants import StabilityReportConfig
from src.core.interfaces import INotifier
from src.logger import AppLogger
from src.monitoring.hourly_report import RunAnalysis

logger = AppLogger().get_logger(__name__)

__all__ = ["TeamsNotifier", "send_teams_summary_card", "send_teams_breakdown_card"]

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
        {"type": "TextBlock", "text": "Per-Type Breakdown", "weight": "Bolder", "size": "Medium", "spacing": "None"},
        {"type": "TextBlock", "text": _utc_now(), "size": "Small", "isSubtle": True, "spacing": "None"},
    ]

    for run in runs:
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


class TeamsNotifier(INotifier):
    """Microsoft Teams notifier using Power Automate Workflow webhooks.

    Posts two Adaptive Cards per :meth:`send` call.

    Args:
        webhook_url: Power Automate webhook URL.  Defaults to
            :attr:`~src.constants.StabilityReportConfig.TEAMS_WEBHOOK_URL`.

    Example::

        notifier = TeamsNotifier(webhook_url="https://...")
        notifier.send(subject="Stability Report", body="...", runs=runs)
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url or StabilityReportConfig.TEAMS_WEBHOOK_URL

    def send(self, subject: str, body: str, **kwargs: Any) -> None:
        """Post Teams Adaptive Cards for the provided runs.

        Args:
            subject: Unused — Teams cards generate their own header.
            body: Unused — Teams cards build their own body from ``runs``.
            **kwargs:
                runs (list[RunAnalysis]): Analysed run objects.  Required.
        """
        runs: list[RunAnalysis] = kwargs.get("runs", [])
        if not runs:
            logger.warning("TeamsNotifier.send() called with empty runs — skipping")
            return
        if not self.webhook_url:
            logger.warning("TEAMS_WEBHOOK_URL is not set — skipping Teams notification")
            return
        asyncio.run(self._send_async(runs))

    async def _send_async(self, runs: list) -> None:
        try:
            await send_teams_summary_card(self.webhook_url, runs)
            await send_teams_breakdown_card(self.webhook_url, runs)
            logger.info(f"Teams notification sent for {len(runs)} run(s)")
        except Exception as e:
            logger.error(f"Teams notification failed: {e}")
