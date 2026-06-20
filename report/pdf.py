#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: pdf.py
=============================================================================

INPUT FILES:
    (none - takes report markdown string from main.py)

OUTPUT FILES:
    - outputs/unified/Unified_Report_<date>.pdf  (via PrinceXML)
    - outputs/unified/Unified_Report_<date>.html (intermediate, kept)

VERSION: 2.0
LAST UPDATED: 2026-06-19
AUTHOR: Arjun Divecha

DESCRIPTION:
    Renders the report markdown to a Tufte-inspired, LIGHT-MODE PDF via
    PrinceXML. Design principles:
      - High data-ink ratio: tables use whitespace, not zebra stripes
      - Layering: primary data dark, grids/borders whisper-light
      - Cover page with navy background (from legacy reports)
      - Gold callout box for Executive Summary (most important thing)
      - Section headers with left accent border
      - CONFIDENTIAL footer with page numbers
      - Every element earns its ink

DEPENDENCIES:
    - markdown (python), prince (binary at /opt/homebrew/bin/prince)

USAGE:
    from pdf import render_pdf
=============================================================================
"""

import subprocess
import sys
from pathlib import Path

import markdown as md_lib

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Tufte-inspired CSS — high data-ink, layered hierarchy, earned elements only
# ---------------------------------------------------------------------------
CSS = """
/* ── Page & Print ──────────────────────────────────────────────────── */
@page {
    size: letter;
    margin: 20mm 18mm 22mm 18mm;
    @bottom-left {
        content: "CONFIDENTIAL";
        font-size: 7pt; color: #999; letter-spacing: 1.5pt;
        font-family: "Helvetica Neue", Helvetica, sans-serif;
    }
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-size: 7.5pt; color: #888;
        font-family: "Helvetica Neue", Helvetica, sans-serif;
    }
}
@page :first { margin: 0; @bottom-left { content: none; } @bottom-right { content: none; } }

/* ── Cover Page ────────────────────────────────────────────────────── */
.cover-page {
    page: cover;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    height: 100vh; width: 100vw;
    background: linear-gradient(160deg, #0f1f3d 0%, #1a3055 40%, #1f3a5f 100%);
    color: #ffffff;
    text-align: center;
    page-break-after: always;
    padding: 0; margin: 0;
}
.cover-page h1 {
    font-family: "Helvetica Neue", Helvetica, sans-serif;
    font-size: 36pt; font-weight: 800;
    letter-spacing: 2pt; line-height: 1.15;
    margin: 0 0 18px 0; color: #ffffff;
    border: none;
}
.cover-page .cover-date {
    font-size: 16pt; font-weight: 300;
    color: rgba(255,255,255,0.85);
    margin-bottom: 60px;
}
.cover-page .cover-meta {
    font-size: 10pt; color: rgba(255,255,255,0.6);
    letter-spacing: 0.5pt;
}
.cover-page .cover-confidential {
    font-size: 9pt; color: #7b9ec2;
    letter-spacing: 2.5pt; margin-top: 8px;
}

/* ── Body Typography ───────────────────────────────────────────────── */
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10pt; line-height: 1.5;
    color: #1a1a1a; background: #ffffff;
    max-width: 100%;
}

/* ── Headings — left accent border, minimal ink ────────────────────── */
h1 {
    font-size: 16pt; font-weight: 600;
    color: #1f3a5f;
    border-bottom: 2px solid #1f3a5f;
    border-left: 4px solid #c8a84e;
    padding: 4px 0 5px 10px;
    margin-top: 0;
}
h2 {
    font-size: 12pt; font-weight: 600;
    color: #1f3a5f;
    border-left: 3.5px solid #1f3a5f;
    border-bottom: none;
    padding: 2px 0 2px 9px;
    margin-top: 20px; margin-bottom: 8px;
}
h3 {
    font-size: 10pt; font-weight: 600;
    color: #2c4f7c;
    margin-top: 14px; margin-bottom: 4px;
}

/* ── Prose — tight, justified, recedes behind tables ───────────────── */
p {
    margin: 5px 0; text-align: justify;
    color: #2a2a2a; font-size: 9.5pt; line-height: 1.5;
}

/* ── Executive Summary callout — gold accent (legacy "MOST IMPORTANT") */
.exec-callout {
    background: #fdf8e8;
    border-left: 4px solid #c8a84e;
    padding: 10px 14px;
    margin: 10px 0 14px 0;
    font-size: 10pt; line-height: 1.5;
    color: #1a1a1a;
}
.exec-callout strong { color: #8b6914; }

/* ── Stale holdings banner ─────────────────────────────────────────── */
.stale-banner {
    background: #fff3cd; border-left: 4px solid #b8860b;
    color: #7a5b00; padding: 8px 14px; font-weight: 600;
    margin: 10px 0; font-size: 9pt;
}

/* ── Tables — Tufte style: whitespace separation, minimal rules ───── */
table {
    border-collapse: collapse; width: 100%;
    margin: 8px 0 12px 0;
    font-size: 8.2pt; line-height: 1.35;
}
thead {
    border-bottom: 1.5px solid #1f3a5f;
}
th {
    background: none;
    color: #1f3a5f; font-weight: 600;
    text-align: right;
    padding: 3px 6px 4px 6px;
    font-size: 7.5pt;
    letter-spacing: 0.3pt;
    text-transform: uppercase;
    border: none;
}
th:first-child { text-align: left; }
td {
    padding: 2.5px 6px;
    text-align: right;
    border: none;
    border-bottom: 0.5px solid #e8ecf0;
    color: #1a1a1a;
}
td:first-child { text-align: left; font-weight: 500; }

/* Last row in table — heavier bottom rule (Tufte double-rule) */
tbody tr:last-child td {
    border-bottom: 1.2px solid #1f3a5f;
}

/* Bold rows (TOTAL, leader/laggard) — stand out without color */
tr td strong, tr th strong { color: #0d1f3a; }

/* No zebra striping — whitespace does the work */

/* ── Lists ─────────────────────────────────────────────────────────── */
ul, ol { margin: 4px 0; padding-left: 18px; font-size: 9.5pt; }
li { margin: 2px 0; }

/* ── Horizontal rules — thin, receding ─────────────────────────────── */
hr { border: none; border-top: 0.5px solid #d0d5dc; margin: 16px 0; }

/* ── Metadata line ─────────────────────────────────────────────────── */
.header-meta {
    color: #888; font-size: 8pt;
    margin-bottom: 14px; letter-spacing: 0.3pt;
}

/* ── Utility: keep tables from splitting awkwardly across pages ────── */
table { page-break-inside: auto; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
h2, h3 { page-break-after: avoid; }
"""

# ---------------------------------------------------------------------------
# HTML template — cover page + body
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>

<!-- Cover Page -->
<div class="cover-page">
    <h1>DAILY MARKET<br>&amp; PORTFOLIO<br>REPORT</h1>
    <div class="cover-date">{date_display}</div>
    <div class="cover-meta">
        <div>Investment Committee Brief</div>
        <div class="cover-confidential">CONFIDENTIAL</div>
    </div>
</div>

<!-- Report Body -->
<div class="header-meta">Generated {generated} &middot; Model: {model}</div>
{stale_banner}
{body}

</body>
</html>"""


def _format_date_display(date: str) -> str:
    """Convert 2026-06-18 to 'June 18, 2026'."""
    try:
        from datetime import datetime
        dt = datetime.strptime(date, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y").replace(" 0", " ")
    except Exception:
        return date


def _post_process_html(html: str) -> str:
    """
    Post-process the rendered HTML to add Tufte-inspired enhancements:
    - Wrap the first paragraph after Executive Summary h2 in a gold callout
    """
    # Find the Executive Summary section and wrap its first <p> in a callout
    import re
    pattern = r'(<h2[^>]*>Executive Summary</h2>\s*)((<p>.*?</p>))'
    replacement = r'\1<div class="exec-callout">\3</div>'
    html = re.sub(pattern, replacement, html, count=1, flags=re.DOTALL)
    return html


def render_pdf(report_md: str, date: str, generated: str, model: str,
               output_dir: Path, stale: bool = False,
               stale_msg: str = "") -> dict:
    """
    Write Unified_Report_<date>.md/.html/.pdf into output_dir.

    Returns dict of artifact paths. Raises RuntimeError if Prince fails
    (md/html are already on disk by then).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"Unified_Report_{date}"
    md_path = base.with_suffix(".md")
    html_path = base.with_suffix(".html")
    pdf_path = base.with_suffix(".pdf")

    # 1. markdown (the canonical artifact - never lost)
    md_path.write_text(report_md)

    # 2. html
    body = md_lib.markdown(report_md, extensions=["tables", "smarty"])
    banner = (f'<div class="stale-banner">STALE HOLDINGS &mdash; {stale_msg}</div>'
              if stale else "")
    date_display = _format_date_display(date)
    html = HTML_TEMPLATE.format(
        title=f"Daily Market &amp; Portfolio Report &mdash; {date}",
        css=CSS, generated=generated, model=model,
        date_display=date_display,
        stale_banner=banner, body=body)

    # Post-process for callout boxes
    html = _post_process_html(html)

    html_path.write_text(html)

    # 3. pdf via Prince
    result = subprocess.run(
        ["prince", str(html_path), "-o", str(pdf_path)],
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(
            f"PrinceXML failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-400:]}\n"
            f"Markdown and HTML were saved:\n  {md_path}\n  {html_path}")

    return {"md": md_path, "html": html_path, "pdf": pdf_path}
