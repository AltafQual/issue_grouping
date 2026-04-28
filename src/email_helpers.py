import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

SMTP_HOST = "smtphost.qualcomm.com"


def send_email_report(run_id: str, sender_email: str, recipient_email: str, report: str) -> None:
    """Send an HTML report via email.

    Args:
        run_id: Run ID used as the email subject identifier.
        sender_email: Sender address.
        recipient_email: Recipient address.
        report: Rendered HTML string to send as the email body.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Stability Report: {run_id}"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.attach(MIMEText(report, "html"))
    try:
        s = smtplib.SMTP(SMTP_HOST)
        s.starttls()
        s.sendmail(sender_email, recipient_email, msg.as_string())
        s.quit()
        logger.info(f"Email sent for run_id={run_id} to {recipient_email}")
    except Exception as e:
        logger.exception(f"Failed to send email for run_id={run_id}: {e}")
