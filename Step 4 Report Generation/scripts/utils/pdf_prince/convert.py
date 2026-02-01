"""
PrinceXML PDF Converter - Sell-Side Quality PDFs

Requires PrinceXML to be installed:
- Download from: https://www.princexml.com/download/
- Install and ensure 'prince' command is in PATH
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template, Environment, FileSystemLoader

# Import chart generation
from . import charts

# Check if PrinceXML is available
PRINCE_AVAILABLE = False
try:
    result = subprocess.run(
        ['prince', '--version'],
        capture_output=True,
        timeout=5
    )
    if result.returncode == 0:
        PRINCE_AVAILABLE = True
except (FileNotFoundError, subprocess.TimeoutExpired):
    PRINCE_AVAILABLE = False


def load_css() -> str:
    """Load CSS stylesheet."""
    css_path = Path(__file__).parent / 'templates' / 'styles.css'
    return css_path.read_text()


def render_html(data: Dict[str, Any], css_content: Optional[str] = None) -> str:
    """
    Render HTML from structured data using Jinja2 template.
    
    Args:
        data: Structured report data dict
        css_content: Optional CSS content (loads default if None)
        
    Returns:
        Rendered HTML string
    """
    if css_content is None:
        css_content = load_css()
    
    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template('report.html')
    
    # Generate charts from data
    chart_data = charts.generate_all_charts(data)
    
    # Prepare template data
    template_data = {
        'report_date': data.get('report_date', 'Unknown Date'),
        'executive_synthesis': data.get('executive_synthesis', {}),
        'flash_headlines': data.get('flash_headlines', []),
        'sections': data.get('sections', []),
        'css_content': css_content,
        'charts': chart_data,  # Add charts
    }
    
    html = template.render(**template_data)
    return html


def generate_pdf_prince(html_content: str, output_path: str) -> bool:
    """
    Generate PDF using PrinceXML.
    
    Args:
        html_content: HTML string to convert
        output_path: Output PDF path
        
    Returns:
        True if successful, False otherwise
    """
    if not PRINCE_AVAILABLE:
        print("ERROR: PrinceXML not found. Install from https://www.princexml.com/download/")
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write HTML to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        temp_html = f.name
    
    try:
        # Run PrinceXML
        cmd = [
            'prince',
            temp_html,
            '-o', str(output_path),
            '--pdf-profile=PDF/X-1a:2003',  # Print quality
            '--pdf-output-intent=sRGB.icc',  # Color profile
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            text=True
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"PrinceXML error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: PrinceXML timed out")
        return False
    except Exception as e:
        print(f"ERROR running PrinceXML: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            Path(temp_html).unlink()
        except:
            pass


def convert_from_data(data: Dict[str, Any], output_path: str) -> Optional[str]:
    """
    Convert structured data to PDF using PrinceXML.
    
    Args:
        data: Structured report data dict
        output_path: Output PDF path
        
    Returns:
        Path to generated PDF, or None if failed
    """
    output_path = Path(output_path)
    
    # Render HTML
    html_content = render_html(data)
    
    # Generate PDF
    success = generate_pdf_prince(html_content, str(output_path))
    
    if success and output_path.exists():
        return str(output_path)
    return None


def convert_report(json_path: str, pdf_path: Optional[str] = None) -> Optional[str]:
    """
    Convert JSON report file to PDF.
    
    Args:
        json_path: Path to JSON file with structured report data
        pdf_path: Optional output PDF path
        
    Returns:
        Path to generated PDF, or None if failed
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return None
    
    if pdf_path is None:
        pdf_path = json_path.with_suffix('.pdf')
    else:
        pdf_path = Path(pdf_path)
    
    # Load JSON data
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        return None
    
    # Extract date from filename if not in data
    if 'report_date' not in data:
        name_parts = json_path.stem.split('_')
        if len(name_parts) >= 3:
            potential_date = name_parts[2]
            from datetime import datetime
            try:
                dt = datetime.strptime(potential_date, '%Y-%m-%d')
                data['report_date'] = dt.strftime('%B %d, %Y')
            except ValueError:
                data['report_date'] = 'Unknown Date'
    
    return convert_from_data(data, str(pdf_path))
