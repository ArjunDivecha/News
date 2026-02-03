#!/usr/bin/env python3
"""
=============================================================================
REPORT GENERATOR - Phase 2 Portfolio Reports
=============================================================================

Generates personalized portfolio daily wrap report using Claude Opus 4.5.

USAGE:
    python 04_generate_report.py --portfolio TEST --date 2026-01-31

OUTPUT:
    - Markdown report in outputs/{portfolio_id}/
    - PDF report in outputs/{portfolio_id}/
    - Report record in portfolio_reports table
=============================================================================
"""

import argparse
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import uuid

import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.db import (
    get_db, get_portfolio, get_holdings, get_daily_snapshot,
    get_aggregates, get_portfolio_summary, get_phase1_market_data,
    save_report
)
from utils.llm import generate_report

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = PROJECT_DIR / "prompts"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_prompt_template() -> tuple:
    """Load the portfolio daily wrap prompt template."""
    prompt_path = PROMPTS_DIR / "portfolio_daily_wrap.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    # Split on USER to get system and user parts
    if "USER" in content:
        parts = content.split("USER", 1)
        system_prompt = parts[0].replace("SYSTEM", "").strip()
        user_prompt = parts[1].strip()
    else:
        system_prompt = ""
        user_prompt = content
    
    return system_prompt, user_prompt


def format_portfolio_summary(summary: dict) -> str:
    """Format portfolio summary as text."""
    if not summary:
        return "No portfolio summary available."
    
    lines = []
    lines.append(f"Total Market Value: ${summary.get('total_market_value', 0):,.2f}")
    lines.append(f"Long Exposure: ${summary.get('total_long_value', 0):,.2f}")
    lines.append(f"Short Exposure: ${summary.get('total_short_value', 0):,.2f}")
    lines.append(f"Net Exposure: ${summary.get('net_exposure', 0):,.2f}")
    lines.append(f"Gross Exposure: ${summary.get('gross_exposure', 0):,.2f}")
    lines.append(f"")
    lines.append(f"Holdings: {summary.get('holding_count', 0)} total")
    lines.append(f"  - Long positions: {summary.get('long_count', 0)}")
    lines.append(f"  - Short positions: {summary.get('short_count', 0)}")
    lines.append(f"")
    lines.append(f"Portfolio Return (1D): {summary.get('portfolio_return_1d', 0):+.2f}%")
    ytd = summary.get('portfolio_return_ytd')
    if ytd is not None:
        lines.append(f"Portfolio Return (YTD): {ytd:+.2f}%")
    lines.append(f"Total Unrealized P&L: ${summary.get('total_open_pnl', 0):,.2f}")
    
    return "\n".join(lines)


def format_contributors(contributors: list, label: str) -> str:
    """Format top contributors/detractors as table."""
    if not contributors:
        return f"No {label.lower()} data available."
    
    lines = []
    lines.append(f"| Symbol | Type | Weight | Return | Contribution |")
    lines.append(f"|--------|------|--------|--------|--------------|")
    
    for c in contributors:
        symbol = c.get('symbol', 'N/A')
        pos_type = c.get('position_type', 'LONG')
        weight = c.get('weight', 0) * 100
        ret = c.get('return_1d', 0)
        contrib = c.get('contribution', 0)
        lines.append(f"| {symbol} | {pos_type} | {weight:.1f}% | {ret:+.2f}% | {contrib:+.1f}bp |")
    
    return "\n".join(lines)


def format_aggregates(aggregates: pd.DataFrame, dim_type: str, snapshot: pd.DataFrame = None) -> str:
    """Format aggregates for a specific dimension as table with YTD."""
    dim_data = aggregates[aggregates['dimension_type'] == dim_type].copy()
    
    if dim_data.empty:
        return f"No {dim_type} breakdown available."
    
    # Compute YTD averages from snapshot if available
    ytd_map = {}
    if snapshot is not None and not snapshot.empty:
        # Need to get the dimension field from holdings
        # For tier1/tier2, we need to join with assets table
        # For now, compute weighted YTD from snapshot grouped by the dimension
        conn = get_db()
        
        # Get holdings with their classifications
        holdings_df = pd.read_sql_query("""
            SELECT h.symbol, h.tier1, h.tier2, h.yf_sector, h.country
            FROM portfolio_holdings h
        """, conn)
        conn.close()
        
        # Merge snapshot with holdings to get classifications
        merged = snapshot.merge(holdings_df, on='symbol', how='left')
        
        # Map dimension type to column name
        dim_col_map = {
            'tier1': 'tier1',
            'tier2': 'tier2',
            'sector': 'yf_sector',
            'region': 'country',
        }
        
        dim_col = dim_col_map.get(dim_type)
        if dim_col and dim_col in merged.columns:
            # Compute weighted YTD for each dimension value
            for dim_val in merged[dim_col].dropna().unique():
                subset = merged[merged[dim_col] == dim_val]
                if not subset.empty and subset['weight'].sum() > 0:
                    # Weighted average YTD
                    weighted_ytd = (subset['return_ytd'] * subset['weight']).sum() / subset['weight'].sum()
                    ytd_map[dim_val] = weighted_ytd
    
    # Sort by total weight descending
    dim_data = dim_data.sort_values('total_weight', ascending=False)
    
    lines = []
    lines.append(f"| {dim_type.title()} | Weight | 1D Ret | YTD Ret | Contribution | Holdings |")
    lines.append(f"|{'-'*20}|--------|--------|---------|--------------|----------|")
    
    for _, row in dim_data.iterrows():
        value = row['dimension_value']
        weight = row['total_weight'] * 100
        ret = row['weighted_return_1d']
        ytd = ytd_map.get(value, 0)
        contrib = row['contribution_1d']
        count = row['holding_count']
        
        if weight > 0.5:  # Only show if meaningful weight
            lines.append(f"| {value:<18} | {weight:>5.1f}% | {ret:>+6.2f}% | {ytd:>+6.2f}% | {contrib:>+9.1f}bp | {count:>8} |")
    
    return "\n".join(lines)


def format_holdings_detail(snapshot: pd.DataFrame) -> str:
    """Format holdings detail as table."""
    if snapshot.empty:
        return "No holdings data available."
    
    # Sort by absolute contribution
    snapshot = snapshot.copy()
    snapshot['abs_contrib'] = snapshot['contribution_1d'].abs()
    snapshot = snapshot.sort_values('abs_contrib', ascending=False)
    
    lines = []
    lines.append(f"| Symbol | Type | Weight | Price | 1D Ret | YTD Ret | Contrib | P&L |")
    lines.append(f"|--------|------|--------|-------|--------|---------|---------|-----|")
    
    for _, row in snapshot.head(20).iterrows():  # Top 20 by impact
        symbol = row['symbol']
        pos_type = row['position_type']
        weight = (row['weight'] * 100) if pd.notna(row['weight']) else 0
        price = row['price'] if pd.notna(row['price']) else 0
        ret = row['return_1d'] if pd.notna(row['return_1d']) else 0
        ytd = row['return_ytd'] if pd.notna(row['return_ytd']) else 0
        contrib = row['contribution_1d'] if pd.notna(row['contribution_1d']) else 0
        pnl = row['open_pnl'] if pd.notna(row['open_pnl']) else 0
        
        lines.append(f"| {symbol:<6} | {pos_type:5} | {weight:>5.1f}% | ${price:>7.2f} | {ret:>+5.2f}% | {ytd:>+6.2f}% | {contrib:>+6.1f}bp | ${pnl:>+10,.0f} |")
    
    return "\n".join(lines)


def get_market_context(date: str) -> str:
    """Get Phase 1 market context if available."""
    market_data = get_phase1_market_data(date)
    
    if not market_data:
        return "Phase 1 market data not available for this date."
    
    lines = []
    
    # Category stats
    category_stats = market_data.get('category_stats')
    if category_stats is not None and not category_stats.empty:
        # Tier 1 summary
        tier1 = category_stats[category_stats['category_type'] == 'tier1']
        if not tier1.empty:
            lines.append("ASSET CLASS PERFORMANCE:")
            lines.append("| Category | Avg Return | Count |")
            lines.append("|----------|------------|-------|")
            for _, row in tier1.iterrows():
                lines.append(f"| {row['category_value']} | {row['avg_return']:+.2f}% | {row['count']} |")
            lines.append("")
    
    # Factor returns
    factor_returns = market_data.get('factor_returns')
    if factor_returns is not None and not factor_returns.empty:
        lines.append("FACTOR RETURNS:")
        for _, row in factor_returns.iterrows():
            ret = row['return_1d'] if pd.notna(row['return_1d']) else 0
            lines.append(f"  {row['factor_name']}: {ret:+.2f}%")
    
    if not lines:
        return "Limited Phase 1 market context available."
    
    return "\n".join(lines)


def prepare_prompt_data(portfolio_id: str, date: str) -> dict:
    """Prepare all data for prompt injection."""
    # Load portfolio info
    portfolio = get_portfolio(portfolio_id)
    if not portfolio:
        raise ValueError(f"Portfolio not found: {portfolio_id}")
    
    # Load summary
    summary = get_portfolio_summary(portfolio_id, date)
    if not summary:
        raise ValueError(f"No summary found for {portfolio_id} on {date}")
    
    # Load aggregates
    aggregates = get_aggregates(portfolio_id, date)
    
    # Load snapshot
    snapshot = get_daily_snapshot(portfolio_id, date)
    
    # Get market context
    market_context = get_market_context(date)
    
    # Prepare data dict
    return {
        'date': date,
        'portfolio_id': portfolio_id,
        'portfolio_name': portfolio.get('portfolio_name', portfolio_id),
        'portfolio_summary': format_portfolio_summary(summary),
        'top_contributors': format_contributors(summary.get('top_contributors', []), 'Contributors'),
        'top_detractors': format_contributors(summary.get('top_detractors', []), 'Detractors'),
        'regional_breakdown': format_aggregates(aggregates, 'region', snapshot),
        'tier1_breakdown': format_aggregates(aggregates, 'tier1', snapshot),
        'tier2_breakdown': format_aggregates(aggregates, 'tier2', snapshot),
        'sector_breakdown': format_aggregates(aggregates, 'sector', snapshot),
        'holdings_detail': format_holdings_detail(snapshot),
        'market_context': market_context,
    }


def inject_data_into_prompt(template: str, data: dict) -> str:
    """Inject data into prompt template."""
    result = template
    for key, value in data.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def parse_markdown_narratives(md_content: str) -> dict:
    """Parse LLM-generated markdown to extract narrative sections."""
    import re
    
    narratives = {
        'executive_summary': '',
        'key_takeaways': [],
        'what_to_watch': [],
        'contributors_narrative': '',
        'regional_narrative': '',
        'sector_narrative': '',
        'long_short_narrative': '',
        'pnl_narrative': '',
        'risk_narrative': '',
        'market_context_narrative': '',
    }
    
    # Extract executive summary (the blockquote after EXECUTIVE SYNTHESIS)
    exec_match = re.search(r'EXECUTIVE SYNTHESIS.*?\n\n>\s*\**PORTFOLIO PERFORMANCE[^>]*\*\*:?\s*\n?>\s*(.+?)(?:\n\n|\*\*KEY)', md_content, re.DOTALL | re.IGNORECASE)
    if exec_match:
        narratives['executive_summary'] = exec_match.group(1).strip().replace('> ', '').replace('\n', ' ')
    
    # Extract key takeaways
    takeaways_match = re.search(r'\*\*KEY TAKEAWAYS:\*\*\s*\n((?:\d+\..+?\n)+)', md_content, re.DOTALL)
    if takeaways_match:
        lines = takeaways_match.group(1).strip().split('\n')
        for line in lines:
            cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
            if cleaned:
                narratives['key_takeaways'].append(cleaned)
    
    # Extract what to watch
    watch_match = re.search(r'\*\*WHAT TO WATCH:\*\*\s*\n((?:-.+?\n)+)', md_content, re.DOTALL)
    if watch_match:
        lines = watch_match.group(1).strip().split('\n')
        for line in lines:
            cleaned = line.strip().lstrip('- ')
            if cleaned:
                narratives['what_to_watch'].append(cleaned)
    
    # Extract narrative sections (the paragraphs starting with **NARRATIVE:** or just after tables)
    # Extract narrative sections using more robust logic
    # Strategy: Find the section header, then capture everything until the next major section marker
    # explicitly excluding known tables or subsections if possible, or just grabbing the text blocks.
    
    def extract_rich_narrative(start_marker, end_marker_pattern=r'(?:\n##|\n---)'):
        # Find start of section
        start_match = re.search(rf'{start_marker}', md_content, re.IGNORECASE)
        if not start_match:
            return ''
        
        start_pos = start_match.end()
        
        # Find next major section to cap the search
        remaining_text = md_content[start_pos:]
        end_match = re.search(end_marker_pattern, remaining_text)
        
        if end_match:
            section_text = remaining_text[:end_match.start()]
        else:
            section_text = remaining_text
            
        # Clean up the text: remove tables
        # Remove markdown tables
        text_without_tables = re.sub(r'\|.*\|.*\n\|[-:| ]+\|\n(?:\|.*\|\n)*', '', section_text, flags=re.MULTILINE)
        
        # Remove the "Top 5 Contributors" type subheaders if they remain
        text_without_tables = re.sub(r'\*\*Top 5 [^\n]+\*\*', '', text_without_tables)
        
        # What remains are paragraphs and h3 headers. 
        # Convert h3 headers (### Title) to bold paragraph starts or just keep them?
        # HTML template uses {{ narrative }}, so we can keep HTML-friendly formatting or just text.
        # Let's convert markdown h3 to HTML h4 for the template
        formatted_text = re.sub(r'###\s+(.+)', r'<h4>\1</h4>', text_without_tables)
        
        # Convert bold **Text** to <strong>Text</strong>
        formatted_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted_text)
        
        # Convert newlines to breaks for simple text blocks, but preserve paragraph structure
        # Better: split into paragraphs
        paragraphs = [p.strip() for p in formatted_text.split('\n\n') if p.strip()]
        
        return '\n\n'.join([f'<p>{p}</p>' if not p.startswith('<h') else p for p in paragraphs])

    narratives['contributors_narrative'] = extract_rich_narrative(r'## 2\. TOP CONTRIBUTORS')
    narratives['regional_narrative'] = extract_rich_narrative(r'## 3\. REGIONAL')
    narratives['sector_narrative'] = extract_rich_narrative(r'## 4\. SECTOR')
    narratives['long_short_narrative'] = extract_rich_narrative(r'## 5\. LONG VS SHORT')
    narratives['pnl_narrative'] = extract_rich_narrative(r'## 6\. P&L')
    narratives['risk_narrative'] = extract_rich_narrative(r'## 7\. CONCENTRATION')
    narratives['market_context_narrative'] = extract_rich_narrative(r'## 8\. MARKET CONTEXT')
    
    return narratives


def generate_pdf_prince(portfolio_id: str, date: str, 
                        summary: dict, aggregates: pd.DataFrame,
                        snapshot: pd.DataFrame, pdf_path: Path,
                        md_content: str = None) -> bool:
    """Generate PDF using PrinceXML with professional template."""
    try:
        from utils.pdf_prince.convert import convert_to_pdf, PRINCE_AVAILABLE
        
        if not PRINCE_AVAILABLE:
            return False
        
        # Parse markdown for narratives if provided
        narratives = {}
        if md_content:
            narratives = parse_markdown_narratives(md_content)
        
        # Get top contributors/detractors from summary
        top_contributors = summary.get('top_contributors', [])
        top_detractors = summary.get('top_detractors', [])
        
        if isinstance(top_contributors, str):
            top_contributors = json.loads(top_contributors)
        if isinstance(top_detractors, str):
            top_detractors = json.loads(top_detractors)
        
        # Convert aggregates to list of dicts
        agg_list = aggregates.to_dict('records') if not aggregates.empty else []
        
        # Convert snapshot to list of dicts
        holdings_list = []
        if not snapshot.empty:
            for _, row in snapshot.iterrows():
                holdings_list.append({
                    'symbol': row['symbol'],
                    'position_type': row['position_type'],
                    'weight': row['weight'] if pd.notna(row['weight']) else 0,
                    'market_value_usd': row['market_value_usd'] if pd.notna(row['market_value_usd']) else 0,
                    'return_1d': row['return_1d'] if pd.notna(row['return_1d']) else 0,
                    'contribution_1d': row['contribution_1d'] if pd.notna(row['contribution_1d']) else 0,
                    'open_pnl': row['open_pnl'] if pd.notna(row['open_pnl']) else 0,
                })
        
        # Build data structure for template
        data = {
            'portfolio_id': portfolio_id,
            'date': date,
            'summary': summary,
            'aggregates': agg_list,
            'holdings': holdings_list,
            'executive_summary': narratives.get('executive_summary') or f"Portfolio returned {summary.get('portfolio_return_1d', 0):+.2f}% today.",
            'key_takeaways': narratives.get('key_takeaways', []),
            'what_to_watch': narratives.get('what_to_watch', []),
            'contributors_narrative': narratives.get('contributors_narrative', ''),
            'regional_narrative': narratives.get('regional_narrative', ''),
            'sector_narrative': narratives.get('sector_narrative', ''),
            'long_short_narrative': narratives.get('long_short_narrative', ''),
            'pnl_narrative': narratives.get('pnl_narrative', ''),
            'risk_narrative': narratives.get('risk_narrative', ''),
            'market_context_narrative': narratives.get('market_context_narrative', ''),
        }
        
        result = convert_to_pdf(data, str(pdf_path))
        return result is not None
        
    except Exception as e:
        print(f"  ⚠️  PrinceXML PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def markdown_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Convert markdown to PDF using WeasyPrint (fallback)."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        
        # Read markdown
        md_content = md_path.read_text()
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code']
        )
        
        # Wrap in HTML document with styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #1a1a2e;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 8px;
                }}
                h1 {{ font-size: 24px; }}
                h2 {{ font-size: 20px; }}
                h3 {{ font-size: 16px; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 16px 0;
                    font-size: 12px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #1a1a2e;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                blockquote {{
                    background: #f0f4f8;
                    border-left: 4px solid #1a1a2e;
                    margin: 16px 0;
                    padding: 12px 20px;
                }}
                code {{
                    background: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                }}
                ul, ol {{
                    margin: 8px 0;
                }}
                li {{
                    margin: 4px 0;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #eee;
                    margin: 24px 0;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        HTML(string=full_html).write_pdf(str(pdf_path))
        return True
        
    except Exception as e:
        print(f"  ⚠️  PDF generation failed: {e}")
        return False


def generate_portfolio_report(portfolio_id: str, date: str,
                              verbose: bool = True) -> dict:
    """
    Generate portfolio daily wrap report.
    
    Args:
        portfolio_id: Portfolio identifier
        date: Target date (YYYY-MM-DD)
        verbose: Print progress
        
    Returns:
        Dict with generation results
    """
    if verbose:
        print("=" * 70)
        print("PORTFOLIO REPORT GENERATION")
        print("=" * 70)
        print(f"\nPortfolio: {portfolio_id}")
        print(f"Date: {date}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load prompt template
    if verbose:
        print("\n[1] Loading prompt template...")
    system_prompt, user_template = load_prompt_template()
    
    # Prepare data
    if verbose:
        print("[2] Preparing portfolio data...")
    data = prepare_prompt_data(portfolio_id, date)
    
    # Inject data into prompt
    user_prompt = inject_data_into_prompt(user_template, data)
    
    if verbose:
        print(f"    Prompt length: {len(user_prompt):,} characters")
    
    # Generate report
    if verbose:
        print("[3] Generating report with Claude Opus 4.5...")
    
    result = generate_report(system_prompt, user_prompt, max_tokens=8000)
    
    if 'error' in result:
        raise Exception(f"Report generation failed: {result['error']}")
    
    content = result['content']
    
    if verbose:
        print(f"    ✓ Generated {len(content):,} characters")
        print(f"    Tokens: {result.get('tokens_input', 0):,} in / {result.get('tokens_output', 0):,} out")
        print(f"    Time: {result.get('time_ms', 0):,}ms")
    
    # Create output directory
    output_dir = OUTPUT_DIR / portfolio_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save markdown
    md_filename = f"portfolio_wrap_{date}.md"
    md_path = output_dir / md_filename
    md_path.write_text(content)
    
    if verbose:
        print(f"\n[4] Saved markdown: {md_path}")
    
    # Generate PDF - try PrinceXML first, then WeasyPrint
    if verbose:
        print("[5] Generating PDF...")
    
    pdf_filename = f"portfolio_wrap_{date}.pdf"
    pdf_path = output_dir / pdf_filename
    
    # Get raw data for PrinceXML template
    summary = get_portfolio_summary(portfolio_id, date)
    aggregates = get_aggregates(portfolio_id, date)
    snapshot = get_daily_snapshot(portfolio_id, date)
    
    # Try PrinceXML first (produces professional sell-side quality)
    pdf_success = generate_pdf_prince(portfolio_id, date, summary, aggregates, snapshot, pdf_path, md_content=content)
    
    if not pdf_success:
        if verbose:
            print("    Falling back to WeasyPrint...")
        pdf_success = markdown_to_pdf(md_path, pdf_path)
    
    if verbose and pdf_success:
        print(f"    ✓ Saved PDF: {pdf_path}")
    
    # Save to database
    report_id = str(uuid.uuid4())[:8]
    save_report(
        report_id=report_id,
        portfolio_id=portfolio_id,
        report_date=date,
        content_md=content,
        model_name=result.get('model', 'unknown'),
        pdf_path=str(pdf_path) if pdf_success else None,
        tokens_input=result.get('tokens_input'),
        tokens_output=result.get('tokens_output'),
        generation_time_ms=result.get('time_ms'),
    )
    
    if verbose:
        print(f"\n[6] Saved report to database (ID: {report_id})")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  Markdown: {md_path}")
        if pdf_success:
            print(f"  PDF: {pdf_path}")
        print(f"\nReport preview (first 500 chars):")
        print("-" * 40)
        print(content[:500])
        print("-" * 40)
    
    return {
        'report_id': report_id,
        'portfolio_id': portfolio_id,
        'date': date,
        'md_path': str(md_path),
        'pdf_path': str(pdf_path) if pdf_success else None,
        'content_length': len(content),
        'tokens_input': result.get('tokens_input'),
        'tokens_output': result.get('tokens_output'),
        'generation_time_ms': result.get('time_ms'),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate portfolio daily wrap report'
    )
    parser.add_argument('--portfolio', required=True,
                        help='Portfolio ID')
    parser.add_argument('--date', required=True,
                        help='Target date (YYYY-MM-DD)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        result = generate_portfolio_report(
            portfolio_id=args.portfolio,
            date=args.date,
            verbose=not args.quiet
        )
        
        print("\n✓ Report generation successful")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
