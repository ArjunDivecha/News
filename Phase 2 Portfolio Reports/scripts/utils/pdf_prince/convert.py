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


def _fmt_currency(value: Any, decimals: int = 0) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "—"
    if decimals > 0:
        return f"${val:,.{decimals}f}"
    return f"${val:,.0f}"


def _fmt_pct(value: Any, decimals: int = 2, signed: bool = True) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "—"
    if signed:
        return f"{val:+.{decimals}f}%"
    return f"{val:.{decimals}f}%"


def _fmt_bp(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{val * 10000:+.0f}bp"


def _strip_html(text: str) -> str:
    if not text:
        return ""
    import re
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def _weighted_return(items: List[Dict[str, Any]]) -> float:
    total_w = 0.0
    total_r = 0.0
    for item in items:
        w = abs(float(item.get('weight', 0) or 0))
        r = float(item.get('return_1d', 0) or 0)
        total_w += w
        total_r += w * r
    if total_w == 0:
        return 0.0
    return total_r / total_w


def build_sections(data: Dict[str, Any]) -> List[Dict]:
    """Build Phase 1 style sections from portfolio data."""
    sections = []

    summary = data.get('summary', {})
    aggregates = data.get('aggregates', [])
    holdings = data.get('holdings', [])

    # Contributors/detractors
    top_contributors = summary.get('top_contributors', [])
    top_detractors = summary.get('top_detractors', [])
    if isinstance(top_contributors, str):
        top_contributors = json.loads(top_contributors)
    if isinstance(top_detractors, str):
        top_detractors = json.loads(top_detractors)

    # Aggregates
    regional_breakdown = []
    sector_breakdown = []
    for agg in aggregates:
        if agg.get('dimension_type') == 'region':
            regional_breakdown.append(agg)
        elif agg.get('dimension_type') in ('tier2', 'sector'):
            sector_breakdown.append(agg)

    # 1. Portfolio At A Glance
    gross_exposure = summary.get('gross_exposure', 0)
    net_exposure = summary.get('net_exposure', 0)
    long_count = summary.get('long_count', 0)
    short_count = summary.get('short_count', 0)
    total_pnl = summary.get('total_open_pnl', 0)
    net_long_pct = (net_exposure / gross_exposure * 100) if gross_exposure else 0
    portfolio_return = summary.get('portfolio_return_1d', 0)

    sections.append({
        'title': 'Portfolio At A Glance',
        'narrative': (
            f"Portfolio returned {_fmt_pct(portfolio_return)} today with {_fmt_currency(gross_exposure)} "
            f"gross exposure across {long_count} long and {short_count} short positions."
        ),
        'tables': [{
            'title': None,
            'headers': ['Metric', 'Value', 'Context'],
            'column_widths': ['40%', '30%', '30%'],
            'column_alignments': ['left', 'right', 'left'],
            'rows': [
                ['Portfolio Return (1D)', _fmt_pct(portfolio_return), 'Positive' if portfolio_return >= 0 else 'Negative'],
                ['Gross Exposure', _fmt_currency(gross_exposure), 'Total capital at risk'],
                ['Net Exposure', _fmt_currency(net_exposure), f"{net_long_pct:.0f}% net long"],
                ['Long Positions', str(long_count), 'Active long bets'],
                ['Short Positions', str(short_count), 'Hedges/shorts'],
                ['Total Unrealized P&L', _fmt_currency(total_pnl), 'In the money' if total_pnl >= 0 else 'Underwater'],
            ]
        }]
    })

    # 2. Top Contributors & Detractors
    contributor_rows = []
    for c in top_contributors[:5]:
        contributor_rows.append([
            c.get('symbol', 'N/A'),
            _fmt_pct(c.get('return_1d', 0)),
            f"{float(c.get('weight', 0) or 0) * 100:.1f}%",
            _fmt_bp(c.get('contribution', 0)),
        ])

    detractor_rows = []
    for d in top_detractors[:5]:
        detractor_rows.append([
            d.get('symbol', 'N/A'),
            _fmt_pct(d.get('return_1d', 0)),
            f"{float(d.get('weight', 0) or 0) * 100:.1f}%",
            _fmt_bp(d.get('contribution', 0)),
        ])

    sections.append({
        'title': 'Top Contributors & Detractors',
        'narrative': _strip_html(data.get('contributors_narrative', '')),
        'tables': [
            {
                'title': 'Top Contributors',
                'headers': ['Position', 'Return', 'Weight', 'Contribution'],
                'column_widths': ['30%', '20%', '20%', '30%'],
                'column_alignments': ['left', 'right', 'right', 'right'],
                'rows': contributor_rows or [['—', '—', '—', '—']],
            },
            {
                'title': 'Top Detractors',
                'headers': ['Position', 'Return', 'Weight', 'Contribution'],
                'column_widths': ['30%', '20%', '20%', '30%'],
                'column_alignments': ['left', 'right', 'right', 'right'],
                'rows': detractor_rows or [['—', '—', '—', '—']],
            },
        ]
    })

    # 3. Regional Exposure Analysis
    regional_rows = []
    for r in sorted(regional_breakdown, key=lambda x: abs(x.get('total_weight', 0)), reverse=True)[:10]:
        regional_rows.append([
            r.get('dimension_value', '—'),
            f"{r.get('total_weight', 0) * 100:.1f}%",
            _fmt_pct(r.get('weighted_return_1d', 0)),
            _fmt_bp(r.get('contribution_1d', 0)),
            str(r.get('holding_count', 0)),
        ])

    sections.append({
        'title': 'Regional Exposure Analysis',
        'narrative': _strip_html(data.get('regional_narrative', '')),
        'tables': [{
            'title': None,
            'headers': ['Region', 'Weight', 'Return', 'Contribution', 'Holdings'],
            'column_widths': ['30%', '15%', '15%', '20%', '20%'],
            'column_alignments': ['left', 'right', 'right', 'right', 'right'],
            'rows': regional_rows or [['—', '—', '—', '—', '—']],
        }]
    })

    # 4. Sector/Theme Exposure
    sector_rows = []
    for s in sorted(sector_breakdown, key=lambda x: abs(x.get('total_weight', 0)), reverse=True)[:10]:
        sector_rows.append([
            s.get('dimension_value', '—'),
            f"{s.get('total_weight', 0) * 100:.1f}%",
            _fmt_pct(s.get('weighted_return_1d', 0)),
            _fmt_bp(s.get('contribution_1d', 0)),
        ])

    sections.append({
        'title': 'Sector/Theme Exposure',
        'narrative': _strip_html(data.get('sector_narrative', '')),
        'tables': [{
            'title': None,
            'headers': ['Sector/Theme', 'Weight', 'Return', 'Contribution'],
            'column_widths': ['40%', '20%', '20%', '20%'],
            'column_alignments': ['left', 'right', 'right', 'right'],
            'rows': sector_rows or [['—', '—', '—', '—']],
        }]
    })

    # 5. Long vs Short Analysis
    long_items = [h for h in holdings if (h.get('position_type', 'LONG') or 'LONG').upper() == 'LONG']
    short_items = [h for h in holdings if (h.get('position_type', 'LONG') or 'LONG').upper() == 'SHORT']
    if long_items or short_items:
        long_return = _weighted_return(long_items)
        short_return = _weighted_return(short_items)
        long_contrib = sum(float(h.get('contribution_1d', 0) or 0) for h in long_items)
        short_contrib = sum(float(h.get('contribution_1d', 0) or 0) for h in short_items)
        long_value = summary.get('total_long_value', None)
        short_value = summary.get('total_short_value', None)

        sections.append({
            'title': 'Long vs Short Analysis',
            'narrative': _strip_html(data.get('long_short_narrative', '')),
            'tables': [{
                'title': None,
                'headers': ['Position Type', 'Exposure', 'Return', 'Contribution'],
                'column_widths': ['30%', '30%', '20%', '20%'],
                'column_alignments': ['left', 'right', 'right', 'right'],
                'rows': [
                    ['Long', _fmt_currency(long_value) if long_value is not None else '—',
                     _fmt_pct(long_return), _fmt_bp(long_contrib)],
                    ['Short', _fmt_currency(short_value) if short_value is not None else '—',
                     _fmt_pct(short_return), _fmt_bp(short_contrib)],
                ],
            }]
        })

    # 6. P&L Analysis
    holdings_detail = sorted(
        [{
            'symbol': h.get('symbol', ''),
            'open_pnl': h.get('open_pnl', 0),
            'market_value': h.get('market_value_usd', 0),
            'return_1d': h.get('return_1d', 0),
        } for h in holdings],
        key=lambda x: x['open_pnl'],
    )
    gains = list(reversed(holdings_detail[-5:]))
    losses = holdings_detail[:5]

    gains_rows = [
        [g['symbol'] or '—', _fmt_currency(g['open_pnl']), _fmt_currency(g['market_value']), _fmt_pct(g['return_1d'])]
        for g in gains if g.get('symbol')
    ]
    losses_rows = [
        [l['symbol'] or '—', _fmt_currency(l['open_pnl']), _fmt_currency(l['market_value']), _fmt_pct(l['return_1d'])]
        for l in losses if l.get('symbol')
    ]

    sections.append({
        'title': 'P&L Analysis',
        'narrative': _strip_html(data.get('pnl_narrative', '')),
        'tables': [
            {
                'title': 'Largest Unrealized Gains',
                'headers': ['Position', 'Unrealized P&L', 'Market Value', 'Return'],
                'column_widths': ['30%', '25%', '25%', '20%'],
                'column_alignments': ['left', 'right', 'right', 'right'],
                'rows': gains_rows or [['—', '—', '—', '—']],
            },
            {
                'title': 'Largest Unrealized Losses',
                'headers': ['Position', 'Unrealized P&L', 'Market Value', 'Return'],
                'column_widths': ['30%', '25%', '25%', '20%'],
                'column_alignments': ['left', 'right', 'right', 'right'],
                'rows': losses_rows or [['—', '—', '—', '—']],
            }
        ]
    })

    # 7. Concentration, Risk & Scenario Analysis
    risk_alerts = data.get('risk_alerts', [])
    risk_rows = [[r.get('title', 'RISK'), r.get('detail', '')] for r in risk_alerts]
    sections.append({
        'title': 'Concentration, Risk & Scenario Analysis',
        'narrative': _strip_html(data.get('risk_narrative', '')),
        'tables': [{
            'title': None,
            'headers': ['Risk Flag', 'Detail'],
            'column_widths': ['30%', '70%'],
            'column_alignments': ['left', 'left'],
            'rows': risk_rows or [['—', '—']],
        }]
    })

    # 8. Market Context for Portfolio (optional)
    market_context = _strip_html(data.get('market_context_narrative', ''))
    if market_context:
        sections.append({
            'title': 'Market Context for Portfolio',
            'narrative': market_context,
            'tables': [],
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
    
    summary = data.get('summary', {})

    # Contributors/detractors
    top_contributors = summary.get('top_contributors', [])
    top_detractors = summary.get('top_detractors', [])
    if isinstance(top_contributors, str):
        top_contributors = json.loads(top_contributors)
    if isinstance(top_detractors, str):
        top_detractors = json.loads(top_detractors)

    # Risk alerts (simple concentration check)
    holdings = data.get('holdings', [])
    risk_alerts = data.get('risk_alerts', [])
    if not risk_alerts and holdings:
        try:
            top_weight = max(abs(float(h.get('weight', 0) or 0)) * 100 for h in holdings)
            if top_weight > 10:
                risk_alerts.append({
                    'severity': 'warning',
                    'title': 'CONCENTRATION WARNING',
                    'detail': f"Top position is {top_weight:.1f}% of portfolio"
                })
        except ValueError:
            pass

    # Executive synthesis
    exec_summary = (data.get('executive_summary') or "").strip()
    if not exec_summary:
        exec_summary = f"Portfolio returned {_fmt_pct(summary.get('portfolio_return_1d', 0))} today."
    executive_synthesis = {
        'single_most_important': _strip_html(exec_summary),
        'key_takeaways': data.get('key_takeaways', []),
        'what_to_watch': data.get('what_to_watch', []),
    }

    # Flash headlines derived from portfolio data
    flash_headlines = []
    if top_contributors:
        top_c = top_contributors[0]
        flash_headlines.append(
            f"Top contributor {top_c.get('symbol', 'N/A')} added {_fmt_bp(top_c.get('contribution', 0))}."
        )
    if top_detractors:
        top_d = top_detractors[0]
        flash_headlines.append(
            f"Top detractor {top_d.get('symbol', 'N/A')} detracted {_fmt_bp(top_d.get('contribution', 0))}."
        )

    if not flash_headlines:
        flash_headlines = ["Portfolio performance drivers summarized below."]

    # Charts (kept for compatibility)
    chart_data = {}
    if CHARTS_AVAILABLE:
        chart_data = charts.generate_all_charts({
            'top_contributors': top_contributors[:5],
            'top_detractors': top_detractors[:5],
            'regional_breakdown': [],
            'holdings_detail': [],
        })

    data_for_sections = dict(data)
    data_for_sections['risk_alerts'] = risk_alerts

    return {
        'report_date': formatted_date,
        'executive_synthesis': executive_synthesis,
        'flash_headlines': flash_headlines,
        'sections': build_sections(data_for_sections),
        'css_content': load_css(),
        'charts': chart_data,
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
    template = env.get_template('report.html')
    
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
