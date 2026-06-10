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

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Renders the report markdown to a clean, LIGHT-MODE PDF via PrinceXML.
    The .md and .html are always written first, so a Prince failure never
    loses the report content (fail-loud, but artifacts preserved).

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

CSS = """
@page {
    size: letter;
    margin: 22mm 18mm;
    @bottom-center { content: counter(page) " / " counter(pages);
                     font-size: 8pt; color: #888; }
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10.5pt; line-height: 1.55; color: #1a1a1a;
    background: #ffffff; max-width: 100%;
}
h1 { font-size: 17pt; border-bottom: 2.5px solid #1f3a5f;
     padding-bottom: 6px; color: #1f3a5f; }
h2 { font-size: 13pt; color: #1f3a5f; margin-top: 22px;
     border-bottom: 1px solid #d5dce6; padding-bottom: 3px; }
h3 { font-size: 11pt; color: #2c4f7c; }
p { margin: 7px 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
        font-size: 8.6pt; }
th { background: #eef2f7; color: #1f3a5f; text-align: right;
     padding: 4px 7px; border-bottom: 1.5px solid #1f3a5f; }
th:first-child, td:first-child { text-align: left; }
td { padding: 3px 7px; border-bottom: 0.5px solid #e3e8ee;
     text-align: right; }
tr:nth-child(even) td { background: #f7f9fb; }
strong { color: #143050; }
.header-meta { color: #667; font-size: 9pt; margin-bottom: 18px; }
.stale-banner { background: #fff3cd; border: 1.5px solid #b8860b;
                color: #7a5b00; padding: 8px 12px; font-weight: bold;
                margin: 12px 0; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>
<h1>{title}</h1>
<div class="header-meta">Generated {generated} &middot; Model: {model}</div>
{stale_banner}
{body}
</body>
</html>"""


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
    banner = (f'<div class="stale-banner">STALE HOLDINGS - {stale_msg}</div>'
              if stale else "")
    html = HTML_TEMPLATE.format(
        title=f"Daily Market &amp; Portfolio Report &mdash; {date}",
        css=CSS, generated=generated, model=model,
        stale_banner=banner, body=body)
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
