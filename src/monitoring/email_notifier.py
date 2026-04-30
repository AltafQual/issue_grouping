"""Email notifier for HTML stability reports.

Provides :class:`EmailNotifier` plus the low-level ``send_email_report``
helper that sends HTML via the Qualcomm internal SMTP relay.

Layering
--------
This module imports from ``src.core.interfaces``, ``src.constants``,
and ``src.logger``.  No imports from ``src.email_helpers``.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from src.constants import StabilityReportConfig
from src.core.interfaces import INotifier
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["EmailNotifier", "send_email_report"]

_SMTP_HOST = "smtphost.qualcomm.com"


def send_email_report(subject: str, sender_email: str, recipient_email: str, report: str) -> None:
    """Send an HTML report via SMTP.

    Args:
        subject: Email subject line.
        sender_email: Sender address.
        recipient_email: Recipient address.
        report: Rendered HTML string to send as the email body.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.attach(MIMEText(report, "html"))
    try:
        s = smtplib.SMTP(_SMTP_HOST)
        s.starttls()
        s.sendmail(sender_email, recipient_email, msg.as_string())
        s.quit()
        logger.info(f"Email sent to {recipient_email}")
    except Exception as e:
        logger.exception(f"Failed to send email to {recipient_email}: {e}")


class EmailNotifier(INotifier):
    """Email notifier that sends HTML stability reports via SMTP.

    Args:
        sender: Sender email address.
        recipient: Recipient email address.

    Example::

        notifier = EmailNotifier()
        notifier.send(subject="QNN Stability", body=html_report, run_id="QNN-001")
    """

    def __init__(self, sender: str | None = None, recipient: str | None = None) -> None:
        self.sender = sender or StabilityReportConfig.SENDER
        self.recipient = recipient or StabilityReportConfig.RECIPIENT

    def send(self, subject: str, body: str, **kwargs: Any) -> None:
        """Send an HTML report via email.

        Args:
            subject: Email subject line.
            body: HTML-formatted email body.
        """
        send_email_report(
            subject=subject,
            sender_email=self.sender,
            recipient_email=self.recipient,
            report=body,
        )
        logger.info(f"Email notification sent to {self.recipient}")
