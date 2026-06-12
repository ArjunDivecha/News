#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: notify.py
=============================================================================

INPUT FILES:
    (reads from env vars via .env):
    - REPORT_EMAIL_TO          : recipient email address (required)
    - SMTP_HOST / SMTP_PORT    : optional SMTP config (uses Mail.app if absent)
    - SMTP_USER / SMTP_PASS    : optional SMTP credentials

OUTPUT FILES:
    (none - sends an email with the report PDF attached)

VERSION: 1.0
LAST UPDATED: 2026-06-11
AUTHOR: Arjun Divecha

DESCRIPTION:
    Sends the daily report PDF via email. Two transport methods:

    1. AppleScript Mail.app (default, zero config when REPORT_EMAIL_TO is set):
       Opens Mail.app, composes an email with the PDF attached, and sends it.
       Uses your already-configured Mac Mail account — no SMTP credentials.

    2. SMTP (when SMTP_HOST is set in .env):
       Connects directly to an SMTP server with optional authentication.
       Use this for Gmail, Fastmail, or any SMTP relay.

    The email subject includes the report date and a STALE flag if
    holdings were stale.

DEPENDENCIES:
    - subprocess (stdlib, for osascript)
    - smtplib, email.mime (stdlib, for SMTP fallback)

USAGE:
    python3 report/notify.py <pdf_path> --date 2026-06-11 [--stale]
=============================================================================
"""

import argparse
import os
import smtplib
import subprocess
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT_DIR


def _load_dotenv():
    """Load .env from repo root (safe to call multiple times)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")


def send_via_mail_app(pdf_path: Path, to_email: str, subject: str, body: str) -> bool:
    """
    Send email via macOS Mail.app using AppleScript.
    This is the zero-config default — uses your existing Mail.app account.
    """
    pdf_abs = str(pdf_path.resolve())

    script = f'''
    tell application "Mail"
        set newMessage to make new outgoing message with properties {{
            subject: "{subject}",
            content: "{body}",
            visible: false
        }}
        tell newMessage
            make new to recipient at end of to recipients with properties {{address: "{to_email}"}}
            make new attachment with properties {{file name:"{pdf_abs}"}} at after last paragraph
        end tell
        send newMessage
    end tell
    '''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  !! AppleScript failed: {result.stderr.strip()}")
        return False
    return True


def send_via_smtp(pdf_path: Path, to_email: str, subject: str, body: str) -> bool:
    """
    Send email via SMTP. Requires SMTP_HOST in .env.
    Uses STARTTLS on port 587 by default.
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")
    from_email = os.getenv("SMTP_FROM", user)

    if not host:
        raise RuntimeError("SMTP_HOST not set in .env")

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_email or user
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))

    with open(pdf_path, "rb") as f:
        attachment = MIMEApplication(f.read(), _subtype="pdf")
        attachment.add_header(
            "Content-Disposition", "attachment",
            filename=pdf_path.name)
        msg.attach(attachment)

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)

    return True


def send_report(pdf_path: str, report_date: str,
                stale: bool = False) -> bool:
    """
    Send the report PDF to the configured recipient.

    Returns True if sent successfully, False otherwise.
    """
    _load_dotenv()

    to_email = os.getenv("REPORT_EMAIL_TO")
    if not to_email:
        print("  notify: REPORT_EMAIL_TO not set in .env — skipping email")
        return False

    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"  !! notify: PDF not found: {pdf}")
        return False

    stale_tag = " [STALE]" if stale else ""
    subject = f"Daily Market Report — {report_date}{stale_tag}"
    body = f"Daily unified market & portfolio report for {report_date}.{stale_tag}\n\nGenerated automatically by report/main.py."

    # Choose transport
    smtp_host = os.getenv("SMTP_HOST")
    if smtp_host:
        print(f"  notify: sending via SMTP ({smtp_host}) to {to_email}...")
        try:
            send_via_smtp(pdf, to_email, subject, body)
            print(f"  notify: email sent via SMTP")
            return True
        except Exception as e:
            print(f"  !! SMTP failed: {e}")
            return False
    else:
        print(f"  notify: sending via Mail.app to {to_email}...")
        try:
            ok = send_via_mail_app(pdf, to_email, subject, body)
            if ok:
                print(f"  notify: email sent via Mail.app")
            else:
                print(f"  !! Mail.app send failed")
            return ok
        except Exception as e:
            print(f"  !! Mail.app failed: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email the daily report PDF")
    parser.add_argument("pdf_path", help="Path to the report PDF")
    parser.add_argument("--date", required=True, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--stale", action="store_true",
                        help="Mark report as using stale holdings")
    args = parser.parse_args()

    ok = send_report(args.pdf_path, args.date, stale=args.stale)
    sys.exit(0 if ok else 1)
