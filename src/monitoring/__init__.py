"""Monitoring package — operational monitoring and notifications.

Modules
-------
hourly_report  — :class:`StabilityMonitor` wrapper + data classes.
teams_notifier    — :class:`TeamsNotifier` for Microsoft Teams Adaptive Cards.
email_notifier    — :class:`EmailNotifier` for HTML email reports.

Layering
--------
``monitoring`` imports from ``src.data``, ``src.utils``, ``src.constants``,
and ``src.logger``.  It must **not** import from ``src.pipeline`` or
``src.clustering``.
"""

from src.monitoring.email_notifier import EmailNotifier
from src.monitoring.hourly_report import StabilityMonitor
from src.monitoring.teams_notifier import TeamsNotifier

__all__ = ["StabilityMonitor", "TeamsNotifier", "EmailNotifier"]
