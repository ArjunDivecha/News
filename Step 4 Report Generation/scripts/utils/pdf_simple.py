"""
=============================================================================
SCRIPT NAME: pdf_simple.py
=============================================================================

DESCRIPTION:
    Converts a markdown (.md) file to a professionally styled PDF using
    WeasyPrint. This is a fallback PDF generator used when PrinceXML
    structured output is not available.

    The script preprocesses the markdown to ensure proper table parsing,
    converts it to HTML via Python-Markdown (with 'tables', 'fenced_code',
    and 'toc' extensions), wraps the result in a full HTML document, then
    renders it to PDF using WeasyPrint with a built-in CSS stylesheet
    designed for investment committee reports.

    The embedded stylesheet includes: letter-sized pages with page numbers,
    a "Confidential - For Investment Committee Use Only" footer, professional
    table formatting with alternating row colors and monospaced right-aligned
    numeric columns, color-coded positive/negative values, executive summary
    blockquote styling, and print-optimized page break behavior.

INPUT FILES:
    <user-provided-path>/report.md
        Path to the markdown input file, provided as a command-line argument
        (sys.argv[1]) when run as a script, or passed directly to the
        convert_report() function when used as a library module. Supports
        pipe-delimited tables and fenced code blocks.

OUTPUT FILES:
    <user-provided-path>/report.pdf
        PDF file generated alongside the input markdown file (same path,
        .pdf extension substituted for .md). Uses the investment committee
        report template described above.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - markdown
    - weasyprint
    - pathlib (stdlib)
    - typing (stdlib)

USAGE:
    # As a script:
    python pdf_simple.py /path/to/report.md

    # As a library module:
    from utils.pdf_simple import convert_report
    pdf_path = convert_report('/path/to/report.md')

NOTES:
    - WeasyPrint must be installed separately: pip install weasyprint
    - If WeasyPrint is unavailable, the script falls back gracefully by
      returning None from convert_report() (no hard import error).
    - All errors are caught and printed to stderr; the script exits with
      code 1 on failure rather than raising an unhandled exception.
    - The generated PDF includes a "Confidential - For Investment Committee
      Use Only" footer on every page and page numbers in the top-right.
=============================================================================
"""

import markdown
from pathlib import Path
from typing import Optional

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


# Professional CSS styling for reports
REPORT_CSS = """
@page {
    size: letter;
    margin: 0.75in;
    @top-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
    }
    @bottom-center {
        content: "Confidential - For Investment Committee Use Only";
        font-size: 8pt;
        color: #999;
    }
}

body {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #333;
    max-width: 7.5in;
}

h1 {
    font-size: 18pt;
    color: #1a365d;
    border-bottom: 2px solid #2c5282;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 16px;
}

h2 {
    font-size: 14pt;
    color: #2c5282;
    margin-top: 24px;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    color: #4a5568;
    margin-top: 16px;
    margin-bottom: 8px;
}

p {
    margin-bottom: 10px;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th {
    background-color: #2c5282;
    color: white;
    font-weight: bold;
    padding: 8px 6px;
    text-align: left;
    border: 1px solid #1a365d;
}

td {
    padding: 6px;
    border: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #f7fafc;
}

tr:hover {
    background-color: #edf2f7;
}

/* Numeric values right-aligned */
td:nth-child(n+2) {
    text-align: right;
    font-family: 'Courier New', monospace;
}

strong {
    color: #2c5282;
}

ul, ol {
    margin-left: 20px;
    margin-bottom: 12px;
}

li {
    margin-bottom: 4px;
}

code {
    background-color: #f7fafc;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 9pt;
}

/* Positive/negative returns coloring */
.positive { color: #2f855a; }
.negative { color: #c53030; }

/* Section breaks */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 20px 0;
}

/* Executive summary box */
blockquote {
    background-color: #ebf8ff;
    border-left: 4px solid #3182ce;
    padding: 12px 16px;
    margin: 16px 0;
    font-style: italic;
}

/* Print optimization */
@media print {
    h1, h2, h3 {
        page-break-after: avoid;
    }
    table {
        page-break-inside: avoid;
    }
    tr {
        page-break-inside: avoid;
    }
}
"""


def preprocess_markdown(md_content: str) -> str:
    """
    Preprocess markdown to fix common issues.
    
    - Ensures blank lines around tables for proper parsing
    - Fixes table alignment
    """
    lines = md_content.split('\n')
    result = []
    in_table = False
    
    for i, line in enumerate(lines):
        # Detect table start
        if line.strip().startswith('|') and not in_table:
            # Add blank line before table if needed
            if result and result[-1].strip():
                result.append('')
            in_table = True
        
        # Detect table end
        if in_table and not line.strip().startswith('|') and line.strip():
            # Add blank line after table
            result.append('')
            in_table = False
        
        result.append(line)
    
    return '\n'.join(result)


def convert_report(md_path: str) -> Optional[str]:
    """
    Convert a markdown report to PDF.
    
    Args:
        md_path: Path to markdown file
        
    Returns:
        Path to generated PDF, or None if failed
    """
    if not WEASYPRINT_AVAILABLE:
        print("ERROR: WeasyPrint not installed. Run: pip install weasyprint")
        return None
    
    md_path = Path(md_path)
    
    if not md_path.exists():
        print(f"ERROR: File not found: {md_path}")
        return None
    
    try:
        # Read markdown
        md_content = md_path.read_text()
        
        # Preprocess
        md_content = preprocess_markdown(md_content)
        
        # Convert to HTML
        html_body = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'toc']
        )
        
        # Wrap in full HTML document
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Daily Market Report</title>
</head>
<body>
{html_body}
</body>
</html>
"""
        
        # Generate PDF
        pdf_path = md_path.with_suffix('.pdf')
        
        html_doc = HTML(string=html_content)
        css = CSS(string=REPORT_CSS)
        html_doc.write_pdf(str(pdf_path), stylesheets=[css])
        
        return str(pdf_path)
        
    except Exception as e:
        print(f"ERROR: PDF conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_simple.py <markdown_file>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_path = convert_report(md_file)
    
    if pdf_path:
        print(f"PDF generated: {pdf_path}")
    else:
        print("PDF generation failed")
        sys.exit(1)
