#!/usr/bin/env python3
"""
=============================================================================
PDF GENERATION UTILITIES
=============================================================================

PURPOSE:
Convert markdown reports to professionally styled PDFs.

USAGE:
    from utils.pdf import markdown_to_pdf, convert_report
=============================================================================
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

# Try to import reportlab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: reportlab not available. Install with: pip install reportlab")


# Professional color scheme
COLORS = {
    'primary': HexColor('#1a365d') if REPORTLAB_AVAILABLE else None,  # Dark blue
    'secondary': HexColor('#2c5282') if REPORTLAB_AVAILABLE else None,  # Medium blue
    'accent': HexColor('#3182ce') if REPORTLAB_AVAILABLE else None,  # Light blue
    'positive': HexColor('#276749') if REPORTLAB_AVAILABLE else None,  # Green
    'negative': HexColor('#c53030') if REPORTLAB_AVAILABLE else None,  # Red
    'text': HexColor('#2d3748') if REPORTLAB_AVAILABLE else None,  # Dark gray
    'light_bg': HexColor('#f7fafc') if REPORTLAB_AVAILABLE else None,  # Light gray
}


def create_styles():
    """Create custom paragraph styles for the report."""
    if not REPORTLAB_AVAILABLE:
        return None
    
    styles = getSampleStyleSheet()
    
    def safe_add(name, **kwargs):
        """Safely add or update a style."""
        if name in styles.byName:
            # Update existing
            for k, v in kwargs.items():
                if k != 'parent':
                    setattr(styles[name], k, v)
        else:
            styles.add(ParagraphStyle(name=name, **kwargs))
    
    # Title style
    safe_add('ReportTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=COLORS['primary'],
        spaceAfter=20,
        alignment=TA_CENTER,
    )
    
    # Section heading
    safe_add('SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=COLORS['secondary'],
        spaceBefore=15,
        spaceAfter=8,
    )
    
    # Subsection heading
    safe_add('SubsectionHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=COLORS['secondary'],
        spaceBefore=10,
        spaceAfter=6,
    )
    
    # Body text
    styles['BodyText'].fontSize = 10
    styles['BodyText'].textColor = COLORS['text']
    styles['BodyText'].spaceBefore = 4
    styles['BodyText'].spaceAfter = 4
    styles['BodyText'].leading = 14
    
    # Bullet point
    safe_add('Bullet',
        parent=styles['Normal'],
        fontSize=10,
        textColor=COLORS['text'],
        leftIndent=20,
        bulletIndent=10,
        spaceBefore=2,
        spaceAfter=2,
    )
    
    # Table header
    safe_add('TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=COLORS['primary'],
        alignment=TA_CENTER,
    )
    
    # Table cell
    safe_add('TableCell',
        parent=styles['Normal'],
        fontSize=9,
        textColor=COLORS['text'],
        alignment=TA_CENTER,
    )
    
    return styles


def parse_markdown_table(table_text: str) -> list:
    """Parse a markdown table into a list of rows."""
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]
    
    if len(lines) < 2:
        return []
    
    rows = []
    for i, line in enumerate(lines):
        if '---' in line or '|---' in line:
            continue  # Skip separator row
        
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]  # Remove empty cells from edges
        
        if cells:
            rows.append(cells)
    
    return rows


def create_table(rows: list, styles) -> Table:
    """Create a styled table from rows."""
    if not REPORTLAB_AVAILABLE or not rows:
        return None
    
    # Convert rows to Paragraphs for better text handling
    data = []
    for i, row in enumerate(rows):
        style_name = 'TableHeader' if i == 0 else 'TableCell'
        data.append([Paragraph(str(cell), styles[style_name]) for cell in row])
    
    # Calculate column widths
    num_cols = len(rows[0]) if rows else 0
    col_width = 6.5 * inch / max(num_cols, 1)
    
    table = Table(data, colWidths=[col_width] * num_cols)
    
    # Style the table
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), COLORS['light_bg']),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), COLORS['light_bg']]),
    ])
    
    table.setStyle(table_style)
    return table


def markdown_to_elements(md_text: str, styles) -> list:
    """Convert markdown text to reportlab flowable elements."""
    if not REPORTLAB_AVAILABLE:
        return []
    
    elements = []
    lines = md_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            elements.append(Spacer(1, 6))
            i += 1
            continue
        
        # Title (# )
        if line.startswith('# '):
            text = line[2:].strip()
            elements.append(Paragraph(text, styles['ReportTitle']))
            i += 1
            continue
        
        # Section heading (### )
        if line.startswith('### '):
            text = line[4:].strip()
            elements.append(Paragraph(text, styles['SectionHeading']))
            i += 1
            continue
        
        # Subsection heading (## )
        if line.startswith('## '):
            text = line[3:].strip()
            elements.append(Paragraph(text, styles['SubsectionHeading']))
            i += 1
            continue
        
        # Bold heading (** at start)
        if line.startswith('**') and line.endswith('**'):
            text = line[2:-2].strip()
            elements.append(Paragraph(f"<b>{text}</b>", styles['SubsectionHeading']))
            i += 1
            continue
        
        # Table (starts with |)
        if line.startswith('|'):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            table_text = '\n'.join(table_lines)
            rows = parse_markdown_table(table_text)
            if rows:
                table = create_table(rows, styles)
                if table:
                    elements.append(Spacer(1, 6))
                    elements.append(table)
                    elements.append(Spacer(1, 6))
            continue
        
        # Bullet point
        if line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            # Handle bold within bullet
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            elements.append(Paragraph(f"• {text}", styles['Bullet']))
            i += 1
            continue
        
        # Regular paragraph
        # Handle inline formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
        
        elements.append(Paragraph(text, styles['BodyText']))
        i += 1
    
    return elements


def markdown_to_pdf(md_text: str, output_path: str, 
                   title: Optional[str] = None,
                   date: Optional[str] = None) -> bool:
    """
    Convert markdown text to a PDF file.
    
    Args:
        md_text: Markdown content
        output_path: Path for output PDF
        title: Optional title (added to header)
        date: Optional date (added to header)
        
    Returns:
        True if successful, False otherwise
    """
    if not REPORTLAB_AVAILABLE:
        print("ERROR: reportlab not installed")
        return False
    
    try:
        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        # Get styles
        styles = create_styles()
        
        # Build elements
        elements = []
        
        # Add header if provided
        if title or date:
            header_text = title or "Market Report"
            if date:
                header_text += f" - {date}"
            elements.append(Paragraph(header_text, styles['ReportTitle']))
            elements.append(Spacer(1, 20))
        
        # Convert markdown to elements
        md_elements = markdown_to_elements(md_text, styles)
        elements.extend(md_elements)
        
        # Build PDF
        doc.build(elements)
        
        return True
        
    except Exception as e:
        print(f"ERROR creating PDF: {e}")
        return False


def convert_report(md_path: str, pdf_path: Optional[str] = None) -> Optional[str]:
    """
    Convert a markdown report file to PDF.
    
    Args:
        md_path: Path to markdown file
        pdf_path: Output PDF path (default: same name with .pdf)
        
    Returns:
        Path to PDF if successful, None otherwise
    """
    md_path = Path(md_path)
    
    if not md_path.exists():
        print(f"ERROR: File not found: {md_path}")
        return None
    
    if pdf_path is None:
        pdf_path = md_path.with_suffix('.pdf')
    else:
        pdf_path = Path(pdf_path)
    
    # Read markdown
    md_text = md_path.read_text()
    
    # Extract title and date from filename if possible
    # Format: daily_wrap_YYYY-MM-DD_provider.md
    name_parts = md_path.stem.split('_')
    date = None
    if len(name_parts) >= 3:
        potential_date = '_'.join(name_parts[2:5]) if len(name_parts) >= 5 else name_parts[2]
        if re.match(r'\d{4}-\d{2}-\d{2}', potential_date):
            date = potential_date
    
    title = "Daily Market Wrap"
    
    # Convert
    success = markdown_to_pdf(md_text, str(pdf_path), title, date)
    
    if success:
        return str(pdf_path)
    return None


def test_pdf_generation() -> bool:
    """Test PDF generation with sample content."""
    print("\nTesting PDF Generation...")
    print("-" * 40)
    
    if not REPORTLAB_AVAILABLE:
        print("SKIPPED: reportlab not installed")
        return True  # Not a failure, just not available
    
    # Sample markdown
    sample_md = """# Daily Market Wrap - 2026-01-30

### 1. FLASH HEADLINES

- **Equities down 1.5%** on risk-off sentiment
- Fixed Income outperforms with +0.5% return
- Vol indices spike 15% intraday

### 2. TIER-1 ASSET CLASS DASHBOARD

| Category | 1-Day | 1-Week | YTD |
|----------|-------|--------|-----|
| Equities | -1.5% | -2.1% | +3.2% |
| Fixed Income | +0.5% | +0.8% | +1.1% |
| Commodities | -0.8% | -1.2% | +5.4% |

This is a paragraph with **bold text** and *italic text*.

### 3. KEY OBSERVATIONS

- Risk-off rotation continues for third consecutive day
- Value vs Growth spread widening to 60-day extreme
- Emerging markets showing unusual strength despite US weakness
"""
    
    # Create test PDF
    test_path = Path(__file__).parent.parent.parent / "outputs" / "test_pdf.pdf"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = markdown_to_pdf(sample_md, str(test_path), "Test Report", "2026-01-30")
    
    if success and test_path.exists():
        print(f"PASSED: PDF created at {test_path}")
        print(f"  Size: {test_path.stat().st_size:,} bytes")
        # Clean up test file
        test_path.unlink()
        return True
    else:
        print("FAILED: PDF creation failed")
        return False


if __name__ == "__main__":
    test_pdf_generation()
