"""
PrinceXML PDF Converter - Portfolio Reports
============================================

Creates sell-side quality PDFs from portfolio data using PrinceXML.
Falls back to WeasyPrint if PrinceXML is not available.

Requires PrinceXML to be installed:
- Download from: https://www.princexml.com/download/
- Install and ensure 'prince' command is in PATH
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Check for Jinja2
try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("WARNING: Jinja2 not installed. Install with: pip install jinja2")

# Import chart generation
try:
    from . import charts
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("WARNING: Charts module not available")

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


def build_sections(data: Dict[str, Any]) -> List[Dict]:
    """Build Phase 1 style sections from portfolio data."""
    sections = []
    
    # Get summary data
    summary = data.get('summary', {})
    
    # 1. Portfolio At A Glance
    sections.append({
        'title': 'PORTFOLIO AT A GLANCE',
        'narrative': f"Portfolio returned {summary.get('portfolio_return_1d', 0):+.2f}% today with "
                    f"${summary.get('gross_exposure', 0):,.0f} gross exposure across "
                    f"{summary.get('long_count', 0)} long and {summary.get('short_count', 0)} short positions.",
        'tables': [{
            'title': 'Portfolio Metrics',
            'headers': ['Metric', 'Value', 'Context'],
            'column_widths': ['40%', '30%', '30%'],
            'column_alignments': ['left', 'right', 'left'],
            'rows': [
                ['Portfolio Return (1D)', f"{summary.get('portfolio_return_1d', 0):+.2f}%", 
                 '🟢 Positive' if summary.get('portfolio_return_1d', 0) >= 0 else '🔴 Negative'],
                ['Gross Exposure', f"${summary.get('gross_exposure', 0):,.0f}", 'Total capital at risk'],
                ['Net Exposure', f"${summary.get('net_exposure', 0):,.0f}", 
                 f"{summary.get('net_exposure', 0) / summary.get('gross_exposure', 1) * 100:.0f}% net long"],
                ['Long Positions', str(summary.get('long_count', 0)), 'Active long bets'],
                ['Short Positions', str(summary.get('short_count', 0)), 'Hedges/shorts'],
                ['Total Unrealized P&L', f"${summary.get('total_open_pnl', 0):,.0f}", 
                 'In the money' if summary.get('total_open_pnl', 0) >= 0 else 'Underwater'],
            ]
        }]
    })
    
    # 2. Top Contributors & Detractors
    top_contributors = summary.get('top_contributors', [])
    top_detractors = summary.get('top_detractors', [])
    
    if isinstance(top_contributors, str):
        top_contributors = json.loads(top_contributors)
    if isinstance(top_detractors, str):
        top_detractors = json.loads(top_detractors)
    
    contributor_rows = []
    for c in top_contributors[:5]:
        contributor_rows.append([
            c.get('symbol', ''),
            f"{c.get('return_1d', 0):+.2f}%",
            f"{c.get('weight', 0) * 100:.1f}%",
            f"🟢 {c.get('contribution', 0) * 10000:+.0f}bp",
            'Top contributor'
        ])
    
    contributor_rows.append(['...' for _ in range(5)])
    
    for d in top_detractors[:5]:
        contributor_rows.append([
            d.get('symbol', ''),
            f"{d.get('return_1d', 0):+.2f}%",
            f"{d.get('weight', 0) * 100:.1f}%",
            f"🔴 {d.get('contribution', 0) * 10000:+.0f}bp",
            'Top detractor'
        ])
    
    sections.append({
        'title': 'TOP CONTRIBUTORS & DETRACTORS',
        'narrative': data.get('contributors_narrative', ''),
        'tables': [{
            'title': 'Performance Attribution',
            'headers': ['Position', 'Return', 'Weight', 'Contribution', 'Signal'],
            'column_widths': ['20%', '15%', '15%', '20%', '30%'],
            'column_alignments': ['left', 'right', 'right', 'right', 'left'],
            'rows': contributor_rows
        }]
    })
    
    # 3. Regional Exposure
    regional_breakdown = []
    aggregates = data.get('aggregates', [])
    for agg in aggregates:
        if agg.get('dimension_type') == 'region':
            regional_breakdown.append(agg)
    
    if regional_breakdown:
        regional_rows = []
        for r in sorted(regional_breakdown, key=lambda x: abs(x.get('total_weight', 0)), reverse=True):
            regional_rows.append([
                r.get('dimension_value', ''),
                f"{r.get('total_weight', 0) * 100:.1f}%",
                str(r.get('holding_count', 0)),
                f"{'🟢' if r.get('weighted_return_1d', 0) >= 0 else '🔴'} {r.get('weighted_return_1d', 0):+.2f}%",
                f"{r.get('contribution_1d', 0) * 10000:+.0f}bp"
            ])
        
        sections.append({
            'title': 'REGIONAL EXPOSURE',
            'narrative': data.get('regional_narrative', ''),
            'tables': [{
                'title': 'Regional Performance Breakdown',
                'headers': ['Region', 'Weight', '# Holdings', 'Wtd Return', 'Contribution'],
                'column_widths': ['25%', '15%', '15%', '20%', '25%'],
                'column_alignments': ['left', 'right', 'center', 'right', 'right'],
                'rows': regional_rows
            }]
        })
    
    # 4. Holdings Detail
    holdings = data.get('holdings', [])
    if holdings:
        holdings_rows = []
        for h in sorted(holdings, key=lambda x: abs(x.get('contribution_1d', 0)), reverse=True)[:20]:
            holdings_rows.append([
                h.get('symbol', ''),
                h.get('position_type', 'LONG'),
                f"{abs(h.get('weight', 0)) * 100:.1f}%",
                f"${abs(h.get('market_value_usd', 0)):,.0f}",
                f"{'🟢' if h.get('return_1d', 0) >= 0 else '🔴'} {h.get('return_1d', 0):+.2f}%",
                f"{h.get('contribution_1d', 0) * 100:+.0f}bp",
                f"${h.get('open_pnl', 0):,.0f}"
            ])
        
        sections.append({
            'title': 'HOLDINGS DETAIL',
            'tables': [{
                'title': 'Top 20 Positions by Contribution',
                'headers': ['Symbol', 'Type', 'Weight', 'Mkt Value', 'Return', 'Contribution', 'Unreal P&L'],
                'column_widths': ['12%', '10%', '12%', '18%', '12%', '18%', '18%'],
                'column_alignments': ['left', 'center', 'right', 'right', 'right', 'right', 'right'],
                'rows': holdings_rows
            }]
        })
    
    # 5. Risk & Concentration
    risk_alerts = data.get('risk_alerts', [])
    risk_narrative = data.get('risk_narrative', '')
    
    if risk_alerts or risk_narrative:
        sections.append({
            'title': 'RISK & CONCENTRATION',
            'narrative': risk_narrative,
            'tables': []  # Risk alerts handled separately in template
        })
    
    return sections


def prepare_template_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for the Jinja2 template.
    
    Extracts and formats data for rendering.
    """
    # Format date
    report_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    try:
        dt = datetime.strptime(report_date, '%Y-%m-%d')
        formatted_date = dt.strftime('%B %d, %Y')
    except:
        formatted_date = report_date
    
    # Get summary data
    summary = data.get('summary', {})
    
    # Get contributors/detractors from JSON if needed
    top_contributors = summary.get('top_contributors', [])
    top_detractors = summary.get('top_detractors', [])
    
    if isinstance(top_contributors, str):
        top_contributors = json.loads(top_contributors)
    if isinstance(top_detractors, str):
        top_detractors = json.loads(top_detractors)
    
    # Get regional breakdown from aggregates
    aggregates = data.get('aggregates', [])
    regional_breakdown = []
    sector_breakdown = []
    
    for agg in aggregates:
        if agg.get('dimension_type') == 'region':
            regional_breakdown.append({
                'region': agg['dimension_value'],
                'weight': agg.get('total_weight', 0) * 100,
                'count': agg.get('holding_count', 0),
                'return_1d': agg.get('weighted_return_1d', 0),
                'contribution': agg.get('contribution_1d', 0),
            })
        elif agg.get('dimension_type') == 'tier2':
            sector_breakdown.append({
                'sector': agg['dimension_value'],
                'weight': agg.get('total_weight', 0) * 100,
                'return_1d': agg.get('weighted_return_1d', 0),
                'contribution': agg.get('contribution_1d', 0),
            })
    
    # Sort by weight
    regional_breakdown.sort(key=lambda x: -abs(x['weight']))
    sector_breakdown.sort(key=lambda x: -abs(x['weight']))
    
    # Get holdings detail
    holdings = data.get('holdings', [])
    holdings_detail = []
    
    for h in holdings:
        holdings_detail.append({
            'symbol': h.get('symbol', ''),
            'position_type': h.get('position_type', 'LONG'),
            'weight': abs(h.get('weight', 0)) * 100,
            'market_value': abs(h.get('market_value_usd', 0)),
            'return_1d': h.get('return_1d', 0),
            'contribution': h.get('contribution_1d', 0),
            'open_pnl': h.get('open_pnl', 0),
        })
    
    # Sort by absolute contribution
    holdings_detail.sort(key=lambda x: -abs(x['contribution']))
    
    # Build risk alerts
    risk_alerts = []
    
    # Check for concentration
    if holdings_detail:
        top_weight = holdings_detail[0]['weight'] if holdings_detail else 0
        if top_weight > 10:
            risk_alerts.append({
                'severity': 'warning',
                'title': 'CONCENTRATION WARNING',
                'detail': f"Top position is {top_weight:.1f}% of portfolio"
            })
    
    # Generate charts
    chart_data = {}
    if CHARTS_AVAILABLE:
        chart_data = charts.generate_all_charts({
            'top_contributors': top_contributors[:5],
            'top_detractors': top_detractors[:5],
            'regional_breakdown': regional_breakdown,
            'holdings_detail': holdings_detail[:20],
        })
    
    # Build data structure for Phase 1 template
    return {
        'report_date': formatted_date,
        'executive_synthesis': {
            'single_most_important': data.get('executive_summary', 'Portfolio performance summary.'),
            'key_takeaways': data.get('key_takeaways', []),
            'what_to_watch': data.get('what_to_watch', []),
        },
        'flash_headlines': [],  # Could add market headlines here
        'sections': build_sections(data),
        'css_content': load_css(),
    }


def render_html(data: Dict[str, Any]) -> str:
    """
    Render HTML from structured data using Jinja2 template.
    
    Args:
        data: Portfolio report data dict
        
    Returns:
        Rendered HTML string
    """
    if not JINJA2_AVAILABLE:
        raise ImportError("Jinja2 is required for PDF generation")
    
    # Set up Jinja2 environment
    template_dir = Path(__file__).parent / 'templates'
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Use Phase 1 style template for consistent look
    template = env.get_template('report_phase1_style.html')
    
    # Prepare data
    template_data = prepare_template_data(data)
    
    return template.render(**template_data)


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
        print("WARNING: PrinceXML not found. Install from https://www.princexml.com/download/")
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


def generate_pdf_weasyprint(html_content: str, output_path: str) -> bool:
    """
    Fallback: Generate PDF using WeasyPrint.
    
    Args:
        html_content: HTML string to convert
        output_path: Output PDF path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from weasyprint import HTML, CSS
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        HTML(string=html_content).write_pdf(str(output_path))
        return True
        
    except ImportError:
        print("ERROR: WeasyPrint not installed")
        return False
    except Exception as e:
        print(f"ERROR generating PDF with WeasyPrint: {e}")
        return False


def convert_to_pdf(data: Dict[str, Any], output_path: str) -> Optional[str]:
    """
    Convert portfolio data to PDF.
    
    Tries PrinceXML first, falls back to WeasyPrint.
    
    Args:
        data: Portfolio report data
        output_path: Output PDF path
        
    Returns:
        Path to generated PDF, or None if failed
    """
    output_path = Path(output_path)
    
    # Render HTML
    html_content = render_html(data)
    
    # Save HTML for debugging
    html_path = output_path.with_suffix('.html')
    html_path.write_text(html_content)
    
    # Try PrinceXML first
    if PRINCE_AVAILABLE:
        success = generate_pdf_prince(html_content, str(output_path))
        if success and output_path.exists():
            return str(output_path)
    
    # Fallback to WeasyPrint
    print("Falling back to WeasyPrint...")
    success = generate_pdf_weasyprint(html_content, str(output_path))
    
    if success and output_path.exists():
        return str(output_path)
    
    return None


# For CLI testing
if __name__ == "__main__":
    print(f"PrinceXML available: {PRINCE_AVAILABLE}")
    print(f"Jinja2 available: {JINJA2_AVAILABLE}")
    print(f"Charts available: {CHARTS_AVAILABLE}")
