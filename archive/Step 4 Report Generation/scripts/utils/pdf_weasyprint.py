"""
WeasyPrint PDF Converter for Markdown Reports
"""

from pathlib import Path
from typing import Optional
import markdown
from weasyprint import HTML, CSS


def convert_report(md_path: str, pdf_path: Optional[str] = None) -> Optional[str]:
    """
    Convert markdown report to PDF using WeasyPrint.
    
    Args:
        md_path: Path to markdown file
        pdf_path: Optional output PDF path (defaults to same name with .pdf)
        
    Returns:
        Path to generated PDF, or None if failed
    """
    md_path = Path(md_path)
    
    if not md_path.exists():
        print(f"ERROR: Markdown file not found: {md_path}")
        return None
    
    if pdf_path is None:
        pdf_path = md_path.with_suffix('.pdf')
    else:
        pdf_path = Path(pdf_path)
    
    try:
        # Read markdown
        md_content = md_path.read_text()
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'nl2br']
        )
        
        # Add CSS styling
        css = """
        @page {
            size: letter;
            margin: 1in;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1a1a1a;
        }
        h1 {
            font-size: 24pt;
            font-weight: 700;
            margin-top: 0;
            margin-bottom: 0.5em;
            color: #000;
        }
        h2 {
            font-size: 18pt;
            font-weight: 600;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            color: #000;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3em;
        }
        h3 {
            font-size: 14pt;
            font-weight: 600;
            margin-top: 1.2em;
            margin-bottom: 0.5em;
            color: #333;
        }
        h4 {
            font-size: 12pt;
            font-weight: 600;
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #555;
        }
        p {
            margin: 0.5em 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 10pt;
        }
        th {
            background-color: #f5f5f5;
            font-weight: 600;
            text-align: left;
            padding: 8px;
            border-bottom: 2px solid #ddd;
        }
        td {
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background-color: #fafafa;
        }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: "SF Mono", Monaco, "Courier New", monospace;
            font-size: 9pt;
        }
        blockquote {
            border-left: 4px solid #0066cc;
            padding-left: 1em;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }
        ul, ol {
            margin: 0.5em 0;
            padding-left: 2em;
        }
        li {
            margin: 0.3em 0;
        }
        strong {
            font-weight: 600;
            color: #000;
        }
        hr {
            border: none;
            border-top: 1px solid #ddd;
            margin: 1.5em 0;
        }
        """
        
        # Wrap in full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Daily Market Wrap</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        HTML(string=full_html).write_pdf(
            str(pdf_path),
            stylesheets=[CSS(string=css)]
        )
        
        return str(pdf_path)
        
    except Exception as e:
        print(f"ERROR converting to PDF: {e}")
        return None
