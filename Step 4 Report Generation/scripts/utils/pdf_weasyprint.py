#!/usr/bin/env python3
"""
=============================================================================
PDF GENERATION UTILITIES - WeasyPrint Version (Premium Design)
=============================================================================

INPUT FILES:
- Markdown report files (.md)

OUTPUT FILES:
- Professional PDF reports (.pdf)

VERSION: 4.0.0
CREATED: 2026-01-31

PURPOSE:
Convert markdown reports to premium, institutional-grade PDFs.
Bloomberg Terminal-inspired design with modern typography.

DEPENDENCIES:
pip install weasyprint markdown

USAGE:
    from utils.pdf_weasyprint import convert_report
    pdf_path = convert_report('report.md')
=============================================================================
"""

import re
import markdown
from pathlib import Path
from typing import Optional
from datetime import datetime

# Try to import WeasyPrint
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("WARNING: weasyprint not available. Install with: pip install weasyprint")


# =============================================================================
# PREMIUM CSS STYLESHEET
# =============================================================================

REPORT_CSS = """
/* ============================================
   FONTS - Premium Typography
   ============================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ============================================
   CSS VARIABLES - Design System
   ============================================ */

:root {
    /* Typography */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    
    /* Primary Colors - Deep Navy */
    --navy-900: #0a1628;
    --navy-800: #0f2240;
    --navy-700: #1a365d;
    --navy-600: #234e82;
    --navy-500: #2d6cb5;
    --navy-100: #e0e7f1;
    --navy-50: #f0f4f8;
    
    /* Accent - Electric Blue */
    --accent-500: #3b82f6;
    --accent-400: #60a5fa;
    --accent-300: #93c5fd;
    
    /* Semantic Colors */
    --green-600: #059669;
    --green-500: #10b981;
    --green-100: #d1fae5;
    --green-50: #ecfdf5;
    
    --red-600: #dc2626;
    --red-500: #ef4444;
    --red-100: #fee2e2;
    --red-50: #fef2f2;
    
    --amber-500: #f59e0b;
    --amber-100: #fef3c7;
    
    /* Neutrals */
    --gray-900: #0f172a;
    --gray-800: #1e293b;
    --gray-700: #334155;
    --gray-600: #475569;
    --gray-500: #64748b;
    --gray-400: #94a3b8;
    --gray-300: #cbd5e1;
    --gray-200: #e2e8f0;
    --gray-100: #f1f5f9;
    --gray-50: #f8fafc;
}

/* ============================================
   PAGE SETUP - Premium Layout
   ============================================ */

@page {
    size: letter;
    margin: 0.6in 0.5in 0.7in 0.5in;
    
    @top-left {
        content: none;
    }
    
    @top-center {
        content: none;
    }
    
    @top-right {
        content: none;
    }
    
    @bottom-left {
        content: "CONFIDENTIAL";
        font-family: var(--font-sans);
        font-size: 6.5pt;
        font-weight: 500;
        letter-spacing: 0.15em;
        color: var(--gray-400);
        text-transform: uppercase;
    }
    
    @bottom-center {
        content: "Daily Market Wrap — " string(report-date);
        font-family: var(--font-sans);
        font-size: 7pt;
        color: var(--gray-500);
    }
    
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: var(--font-mono);
        font-size: 8pt;
        font-weight: 500;
        color: var(--navy-700);
    }
}

@page :first {
    margin-top: 0.4in;
    
    @bottom-left { content: none; }
    @bottom-center { content: none; }
    @bottom-right {
        content: "Page " counter(page);
        font-family: var(--font-mono);
        font-size: 8pt;
        color: var(--gray-400);
    }
}

/* ============================================
   BASE STYLES
   ============================================ */

* {
    box-sizing: border-box;
}

html {
    font-family: var(--font-sans);
    font-size: 9pt;
    line-height: 1.6;
    color: var(--gray-800);
    font-weight: 400;
    -webkit-font-smoothing: antialiased;
}

body {
    margin: 0;
    padding: 0;
}

/* ============================================
   HEADER - Title Block
   ============================================ */

h1 {
    font-size: 28pt;
    font-weight: 800;
    color: var(--navy-900);
    margin: 0 0 0.05in 0;
    padding: 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
    string-set: report-date attr(data-date);
}

h1 + h2 {
    font-size: 12pt;
    font-weight: 500;
    color: var(--gray-500);
    margin: 0 0 0.25in 0;
    padding-bottom: 0.18in;
    border-bottom: 3px solid var(--navy-800);
    letter-spacing: 0.02em;
}

/* ============================================
   SECTION HEADINGS
   ============================================ */

h3 {
    font-size: 11pt;
    font-weight: 700;
    color: var(--navy-800);
    margin: 0.28in 0 0.1in 0;
    padding: 0.08in 0 0.08in 0.15in;
    background: linear-gradient(90deg, var(--navy-50) 0%, transparent 70%);
    border-left: 4px solid var(--accent-500);
    border-radius: 0 4px 4px 0;
    page-break-after: avoid;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    font-size: 9.5pt;
}

h4 {
    font-size: 9pt;
    font-weight: 700;
    color: var(--gray-700);
    margin: 0.15in 0 0.06in 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ============================================
   EXECUTIVE SUMMARY - Premium Callout
   ============================================ */

blockquote {
    background: linear-gradient(135deg, var(--navy-900) 0%, var(--navy-700) 100%);
    color: white;
    margin: 0.15in 0;
    padding: 0.2in 0.25in;
    border-radius: 8px;
    border: none;
    page-break-inside: avoid;
    box-shadow: 0 8px 24px rgba(10, 22, 40, 0.35);
    position: relative;
    overflow: hidden;
}

blockquote::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-400) 0%, var(--accent-500) 50%, var(--green-500) 100%);
}

blockquote p {
    margin: 0.05in 0;
    text-align: left;
    color: white;
    font-size: 10.5pt;
    line-height: 1.55;
}

blockquote strong {
    color: #fcd34d;
    font-weight: 700;
}

blockquote p:first-child {
    font-size: 11pt;
    font-weight: 600;
}

/* ============================================
   PARAGRAPHS & TEXT
   ============================================ */

p {
    margin: 0.07in 0;
    text-align: justify;
    hyphens: auto;
    line-height: 1.55;
}

p strong:first-child {
    color: var(--navy-700);
}

/* Narrative sections */
h3 + p, h4 + p, table + p {
    margin-top: 0.1in;
}

/* ============================================
   LISTS - Clean Bullets
   ============================================ */

ul, ol {
    margin: 0.1in 0;
    padding-left: 0.25in;
}

li {
    margin: 0.04in 0;
    padding-left: 0.05in;
    line-height: 1.5;
}

li strong:first-child {
    color: var(--navy-700);
}

/* Numbered lists */
ol li {
    margin: 0.06in 0;
}

ol li strong:first-child {
    color: var(--gray-800);
    font-weight: 600;
}

/* ============================================
   TABLES - Premium Data Display
   ============================================ */

table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 0.12in 0;
    font-size: 8pt;
    page-break-inside: avoid;
    font-variant-numeric: tabular-nums;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

thead {
    display: table-header-group;
}

th {
    background: var(--navy-800);
    color: white;
    font-weight: 600;
    text-align: center;
    padding: 0.09in 0.07in;
    border: none;
    font-size: 7pt;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

th:first-child {
    text-align: left;
    padding-left: 0.12in;
}

td {
    padding: 0.065in 0.07in;
    border: none;
    border-bottom: 1px solid var(--gray-200);
    text-align: center;
    vertical-align: middle;
    font-size: 8pt;
    background: white;
}

/* First column - category names */
td:first-child {
    text-align: left;
    font-weight: 500;
    color: var(--gray-800);
    padding-left: 0.12in;
}

/* Zebra stripes - subtle */
tr:nth-child(even) td {
    background-color: var(--gray-50);
}

/* Last row */
tr:last-child td {
    border-bottom: none;
}

/* Bold rows (important data) */
tr td strong {
    font-weight: 700;
    color: var(--gray-900);
}

/* ============================================
   CONDITIONAL FORMATTING - Returns
   ============================================ */

.positive {
    color: var(--green-600) !important;
    font-weight: 600;
}

.negative {
    color: var(--red-600) !important;
    font-weight: 600;
}

.extreme-positive {
    color: var(--green-600) !important;
    font-weight: 700;
    background: linear-gradient(90deg, var(--green-50) 0%, transparent 100%) !important;
}

.extreme-negative {
    color: var(--red-600) !important;
    font-weight: 700;
    background: linear-gradient(90deg, var(--red-50) 0%, transparent 100%) !important;
}

.neutral {
    color: var(--gray-500);
}

/* ============================================
   Z-SCORE BADGES
   ============================================ */

.z-extreme {
    display: inline-block;
    background: linear-gradient(135deg, var(--red-600) 0%, #b91c1c 100%);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7pt;
    font-weight: 600;
    font-family: var(--font-mono);
    box-shadow: 0 1px 2px rgba(220, 38, 38, 0.3);
}

.z-warning {
    display: inline-block;
    background: linear-gradient(135deg, var(--amber-500) 0%, #d97706 100%);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7pt;
    font-weight: 600;
    font-family: var(--font-mono);
    box-shadow: 0 1px 2px rgba(245, 158, 11, 0.3);
}

.z-positive {
    display: inline-block;
    background: linear-gradient(135deg, var(--green-500) 0%, var(--green-600) 100%);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7pt;
    font-weight: 600;
    font-family: var(--font-mono);
    box-shadow: 0 1px 2px rgba(16, 185, 129, 0.3);
}

/* ============================================
   HORIZONTAL RULES - Subtle Dividers
   ============================================ */

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, var(--gray-300) 0%, var(--gray-200) 40%, transparent 100%);
    margin: 0.2in 0;
}

/* ============================================
   CODE / MONOSPACE
   ============================================ */

code {
    font-family: var(--font-mono);
    font-size: 8pt;
    background: var(--gray-100);
    padding: 2px 5px;
    border-radius: 3px;
    color: var(--navy-700);
}

/* ============================================
   FOOTER / METADATA
   ============================================ */

.report-footer, p:last-child em, article > p:last-child {
    margin-top: 0.25in;
    padding-top: 0.1in;
    border-top: 1px solid var(--gray-200);
    font-size: 7pt;
    color: var(--gray-400);
    font-style: italic;
    text-align: center;
}

/* ============================================
   SPECIAL ELEMENTS
   ============================================ */

/* Key metrics highlight */
.metric {
    display: inline-block;
    background: var(--navy-50);
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 8.5pt;
    font-weight: 600;
    color: var(--navy-700);
}

/* Highlight text */
mark {
    background: var(--amber-100);
    padding: 1px 4px;
    border-radius: 2px;
}

/* ============================================
   PAGE BREAK CONTROL
   ============================================ */

h3 {
    page-break-after: avoid;
}

table, blockquote, ul, ol {
    page-break-inside: avoid;
}

h3 + table,
h3 + ul,
h3 + blockquote,
h3 + p {
    page-break-before: avoid;
}

/* ============================================
   PRINT OPTIMIZATIONS
   ============================================ */

@media print {
    body {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }
}

/* ============================================
   EMOJI STYLING
   ============================================ */

/* Make emojis slightly larger and consistent */
"""


# =============================================================================
# MARKDOWN TO HTML CONVERSION
# =============================================================================

def colorize_returns(html: str) -> str:
    """Color percentage values based on sign and magnitude."""
    
    def replace_percentage(match):
        full_match = match.group(0)
        
        if '<' in full_match or 'class=' in full_match:
            return full_match
        
        try:
            clean = full_match.replace('+', '').replace('%', '').replace('*', '').strip()
            value = float(clean)
            
            if value > 2.0:
                css_class = 'extreme-positive'
            elif value > 0:
                css_class = 'positive'
            elif value < -2.0:
                css_class = 'extreme-negative'
            elif value < 0:
                css_class = 'negative'
            else:
                css_class = 'neutral'
            
            return f'<span class="{css_class}">{full_match}</span>'
        except ValueError:
            return full_match
    
    pattern = r'(?<![<"\w])([+-]?\d+\.?\d*%)'
    return re.sub(pattern, replace_percentage, html)


def enhance_z_scores(html: str) -> str:
    """Style z-score indicators with colored badges."""
    # Red circle = extreme negative
    html = re.sub(
        r'🔴\s*\*{0,2}([+-]?\d+\.?\d*)\*{0,2}',
        r'<span class="z-extreme">🔴 \1</span>',
        html
    )
    # Orange circle = warning
    html = re.sub(
        r'🟠\s*\*{0,2}([+-]?\d+\.?\d*)\*{0,2}',
        r'<span class="z-warning">🟠 \1</span>',
        html
    )
    # Green circle = positive
    html = re.sub(
        r'🟢\s*\*{0,2}([+-]?\d+\.?\d*)\*{0,2}',
        r'<span class="z-positive">🟢 \1</span>',
        html
    )
    return html


def enhance_bold_in_tables(html: str) -> str:
    """Convert any remaining markdown bold to HTML."""
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    return html


def add_table_wrapper(html: str) -> str:
    """Add wrapper classes to tables."""
    return html.replace('<table>', '<table class="data-table">')


def markdown_to_html(md_text: str, report_date: str = None) -> str:
    """Convert markdown to styled HTML."""
    
    # Pre-process: Ensure tables have blank lines before and after
    # This fixes cases where tables immediately follow text like "**Title:**"
    lines = md_text.split('\n')
    processed_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_line = stripped.startswith('|') and stripped.endswith('|')
        is_separator = stripped.startswith('|') and set(stripped.replace('|', '').replace('-', '').replace(':', '').strip()) == set()
        
        if is_table_line or is_separator:
            # Starting a table - add blank line before if needed
            if not in_table and processed_lines and processed_lines[-1].strip() != '':
                processed_lines.append('')
            in_table = True
            processed_lines.append(line)
        else:
            # Ending a table - add blank line after if needed
            if in_table and stripped != '':
                processed_lines.append('')
            in_table = False
            processed_lines.append(line)
    
    md_text = '\n'.join(processed_lines)
    
    # Note: Do NOT use 'nl2br' extension - it breaks table parsing
    md = markdown.Markdown(extensions=[
        'tables',
        'fenced_code',
    ])
    
    body_html = md.convert(md_text)
    
    # Apply enhancements
    body_html = enhance_bold_in_tables(body_html)
    body_html = colorize_returns(body_html)
    body_html = enhance_z_scores(body_html)
    body_html = add_table_wrapper(body_html)
    
    # Extract date
    if not report_date:
        date_match = re.search(
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            md_text
        )
        if date_match:
            report_date = date_match.group(0)
        else:
            report_date = datetime.now().strftime('%B %d, %Y')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Daily Market Wrap - {report_date}</title>
    <style>
{REPORT_CSS}
    </style>
</head>
<body>
    <article>
        {body_html}
    </article>
</body>
</html>
"""
    
    # Add data-date attribute to h1
    html = re.sub(r'<h1>', f'<h1 data-date="{report_date}">', html, count=1)
    
    return html


# =============================================================================
# PDF GENERATION
# =============================================================================

def markdown_to_pdf(md_text: str, output_path: str,
                   report_date: str = None) -> bool:
    """Convert markdown to premium PDF."""
    if not WEASYPRINT_AVAILABLE:
        print("ERROR: weasyprint not installed")
        return False
    
    try:
        html_content = markdown_to_html(md_text, report_date)
        html_doc = HTML(string=html_content)
        html_doc.write_pdf(output_path)
        return True
    except Exception as e:
        print(f"ERROR creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_report(md_path: str, pdf_path: Optional[str] = None) -> Optional[str]:
    """Convert markdown report to premium PDF."""
    md_path = Path(md_path)
    
    if not md_path.exists():
        print(f"ERROR: File not found: {md_path}")
        return None
    
    if pdf_path is None:
        pdf_path = md_path.with_suffix('.pdf')
    else:
        pdf_path = Path(pdf_path)
    
    md_text = md_path.read_text()
    
    # Extract date from filename
    name_parts = md_path.stem.split('_')
    report_date = None
    if len(name_parts) >= 3:
        potential_date = name_parts[2]
        if re.match(r'\d{4}-\d{2}-\d{2}', potential_date):
            try:
                dt = datetime.strptime(potential_date, '%Y-%m-%d')
                report_date = dt.strftime('%B %d, %Y')
            except ValueError:
                pass
    
    success = markdown_to_pdf(md_text, str(pdf_path), report_date)
    
    if success:
        return str(pdf_path)
    return None


# =============================================================================
# TESTING
# =============================================================================

def test_pdf_generation() -> bool:
    """Test PDF generation."""
    print("\nTesting Premium PDF Generation...")
    print("-" * 50)
    
    if not WEASYPRINT_AVAILABLE:
        print("SKIPPED: weasyprint not installed")
        return False
    
    sample_md = """# DAILY MARKET WRAP
## January 30, 2026

---

### 0. EXECUTIVE SYNTHESIS ⭐ START HERE

> **THE SINGLE MOST IMPORTANT THING:**
> Metals collapsed -10.6% in a single session—a 🔴 z-score of -4.2—creating the most lopsided commodities selloff in recent memory.

**KEY TAKEAWAYS:**
1. **Metals implosion is historically extreme:** The -10.6% single-day move is a multi-sigma event.
2. **Defensive rotation worked:** Low Volatility factors (+0.39%) posted gains.
3. **Overbought exhaustion evident:** 193 assets (20%) entered with RSI > 70.

---

### 1. FLASH HEADLINES

- **🔴 EXTREME:** Metals (-10.6%) posted a -4.2 z-score day
- **🟢 UNUSUAL:** Energy (+0.18%) was the ONLY positive commodity
- **⚠️ DIVERGENCE:** India (+0.48%) only positive regional performer

---

### 2. TIER-1 ASSET CLASS DASHBOARD

| Category | 1-Day | Streak | 60d Pctl | Z-Score | Regime |
|----------|-------|--------|----------|---------|--------|
| Fixed Income | -0.09% | -3 | 15% | ⚪ -0.2 | Defensive |
| Currencies | -0.11% | -3 | 33% | ⚪ -0.1 | Pullback |
| Equities | -0.85% | -3 | 7% | ⚪ -0.5 | Weak |
| **Commodities** | **-5.12%** | -1 | 0% | 🔴 **-4.2** | **CRISIS** |

**NARRATIVE:**
Risk-off dominated with all Tier-1 categories negative. The commodities collapse was concentrated in Metals.

---

*Report generated: January 30, 2026 | 970 assets analyzed*
"""
    
    test_path = Path(__file__).parent.parent.parent / "outputs" / "test_premium.pdf"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = markdown_to_pdf(sample_md, str(test_path), "January 30, 2026")
    
    if success and test_path.exists():
        print(f"SUCCESS: PDF created at {test_path}")
        print(f"  Size: {test_path.stat().st_size:,} bytes")
        return True
    else:
        print("FAILED: PDF creation failed")
        return False


if __name__ == "__main__":
    test_pdf_generation()
