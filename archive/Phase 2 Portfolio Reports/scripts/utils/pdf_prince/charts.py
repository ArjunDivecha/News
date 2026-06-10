"""
=============================================================================
PRINCEXML PDF GENERATOR - Portfolio Charts
=============================================================================

PURPOSE:
Generate SVG charts for Portfolio Wrap reports using Matplotlib.

CHARTS:
- Horizontal bar chart for Top Contributors/Detractors
- Regional allocation pie chart
- Sector exposure bar chart

VERSION: 1.0.0
CREATED: 2026-02-01
=============================================================================
"""

import io
import base64
from typing import List, Dict, Optional, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Color scheme matching PrinceXML template
COLORS = {
    'positive': '#10B981',      # Green
    'positive_light': '#34D399',
    'negative': '#EF4444',      # Red
    'negative_light': '#FCA5A5',
    'neutral': '#6B7280',       # Gray
    'neutral_bg': '#F3F4F6',
    'text_primary': '#111827',
    'text_secondary': '#6B7280',
    'text_muted': '#9CA3AF',
    'border': '#E5E7EB',
    'navy_800': '#0f2240',
    'accent_500': '#3b82f6',
}

# Region colors for consistent visualization
REGION_COLORS = {
    'US': '#3b82f6',
    'Europe': '#8b5cf6',
    'Asia': '#f59e0b',
    'Japan': '#ef4444',
    'EM': '#10b981',
    'Global': '#6366f1',
    'LATAM': '#f97316',
    'Other': '#94a3b8',
}


def setup_matplotlib():
    """Configure matplotlib for professional output."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 11,
        'axes.unicode_minus': False,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['border'],
        'axes.linewidth': 0.5,
        'grid.color': COLORS['border'],
        'grid.linewidth': 0.5,
    })


def generate_contribution_chart(
    contributors: List[Dict],
    detractors: List[Dict],
    title: str = "Top Contributors & Detractors",
    width: float = 7.5,
    height: float = 4.0
) -> str:
    """
    Generate horizontal bar chart showing top contributors and detractors.
    
    Args:
        contributors: List of dicts with symbol, contribution keys
        detractors: List of dicts with symbol, contribution keys
        title: Chart title
        width: Figure width in inches
        height: Figure height in inches
        
    Returns:
        Base64-encoded SVG data URI
    """
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Combine and sort data (detractors first, then contributors)
    all_data = []
    for d in detractors[:5]:
        all_data.append({
            'symbol': d.get('symbol', 'Unknown'),
            'contribution': float(d.get('contribution', 0) or 0) * 100,  # Convert to bp
            'type': 'detractor'
        })
    for c in reversed(contributors[:5]):
        all_data.append({
            'symbol': c.get('symbol', 'Unknown'),
            'contribution': float(c.get('contribution', 0) or 0) * 100,  # Convert to bp
            'type': 'contributor'
        })
    
    if not all_data:
        return ""
    
    labels = [d['symbol'] for d in all_data]
    values = [d['contribution'] for d in all_data]
    colors = [COLORS['positive'] if d['type'] == 'contributor' else COLORS['negative'] for d in all_data]
    
    # Create bars
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width_val = bar.get_width()
        label_x = width_val + (abs(width_val) * 0.05 + 2) if width_val >= 0 else width_val - (abs(width_val) * 0.05 + 2)
        ha = 'left' if width_val >= 0 else 'right'
        
        label_text = f'{val:+.0f}bp'
        
        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            label_text,
            va='center', ha=ha,
            fontsize=9,
            fontweight='bold',
            color=COLORS['text_primary']
        )
    
    # Style axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight='medium')
    ax.set_xlabel('Contribution (bp)', fontsize=9, color=COLORS['text_secondary'])
    ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['navy_800'], pad=15)
    
    # Zero line
    ax.axvline(x=0, color=COLORS['text_muted'], linewidth=1.5, linestyle='-')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['border'])
    
    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save to SVG
    buffer = io.BytesIO()
    plt.savefig(buffer, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig)
    
    buffer.seek(0)
    svg_data = buffer.getvalue().decode('utf-8')
    
    encoded = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def generate_regional_chart(
    regional_data: List[Dict],
    title: str = "Regional Allocation",
    width: float = 7.5,
    height: float = 3.5
) -> str:
    """
    Generate horizontal bar chart for regional exposure.
    
    Args:
        regional_data: List of dicts with region, weight, return_1d keys
        title: Chart title
        
    Returns:
        Base64-encoded SVG data URI
    """
    setup_matplotlib()
    
    if not regional_data:
        return ""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
    
    # Sort by weight
    data = sorted(regional_data, key=lambda x: abs(x.get('weight', 0) or 0))
    
    labels = [d.get('region', 'Unknown') for d in data]
    weights = [abs(float(d.get('weight', 0) or 0)) for d in data]
    returns = [float(d.get('return_1d', 0) or 0) for d in data]
    
    region_colors = [REGION_COLORS.get(r, COLORS['neutral']) for r in labels]
    
    y_pos = np.arange(len(labels))
    
    # Left chart - Weight allocation
    bars1 = ax1.barh(y_pos, weights, color=region_colors, height=0.6, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Weight (%)', fontsize=9, color=COLORS['text_secondary'])
    ax1.set_title('Allocation', fontsize=10, fontweight='bold', color=COLORS['navy_800'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Add labels
    for bar, val in zip(bars1, weights):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=8, color=COLORS['text_primary'])
    
    # Right chart - Returns
    colors2 = [COLORS['positive'] if r >= 0 else COLORS['negative'] for r in returns]
    bars2 = ax2.barh(y_pos, returns, color=colors2, height=0.6, edgecolor='white', linewidth=0.5)
    ax2.set_yticks([])
    ax2.set_xlabel('1-Day Return (%)', fontsize=9, color=COLORS['text_secondary'])
    ax2.set_title('Performance', fontsize=10, fontweight='bold', color=COLORS['navy_800'])
    ax2.axvline(x=0, color=COLORS['text_muted'], linewidth=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Add labels
    for bar, val in zip(bars2, returns):
        offset = 0.1 if val >= 0 else -0.1
        ha = 'left' if val >= 0 else 'right'
        ax2.text(val + offset, bar.get_y() + bar.get_height()/2, 
                f'{val:+.2f}%', va='center', ha=ha, fontsize=8, color=COLORS['text_primary'])
    
    fig.suptitle(title, fontsize=12, fontweight='bold', color=COLORS['navy_800'], y=1.02)
    plt.tight_layout()
    
    # Save to SVG
    buffer = io.BytesIO()
    plt.savefig(buffer, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig)
    
    buffer.seek(0)
    svg_data = buffer.getvalue().decode('utf-8')
    
    encoded = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def generate_holdings_heatmap(
    holdings: List[Dict],
    title: str = "Holdings Return Heatmap",
    width: float = 7.5,
    height: float = 4.5,
    max_holdings: int = 20
) -> str:
    """
    Generate a treemap-style heatmap of holdings by return.
    
    Args:
        holdings: List of dicts with symbol, weight, return_1d keys
        title: Chart title
        max_holdings: Maximum number of holdings to display
        
    Returns:
        Base64-encoded SVG data URI
    """
    setup_matplotlib()
    
    if not holdings:
        return ""
    
    # Sort by absolute contribution and take top N
    sorted_holdings = sorted(holdings, key=lambda x: abs(x.get('contribution', 0) or 0), reverse=True)[:max_holdings]
    
    # Sort again by return for display
    sorted_holdings = sorted(sorted_holdings, key=lambda x: x.get('return_1d', 0) or 0)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    labels = [h.get('symbol', '?') for h in sorted_holdings]
    returns = [float(h.get('return_1d', 0) or 0) for h in sorted_holdings]
    weights = [abs(float(h.get('weight', 0) or 0)) * 100 for h in sorted_holdings]  # Convert to %
    
    # Color based on return
    colors = [COLORS['positive'] if r >= 0 else COLORS['negative'] for r in returns]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, returns, color=colors, height=0.7, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val, wt in zip(bars, returns, weights):
        width_val = bar.get_width()
        # Put label outside bar
        if width_val >= 0:
            label_x = width_val + 0.2
            ha = 'left'
        else:
            label_x = width_val - 0.2
            ha = 'right'
        
        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f'{val:+.1f}%',
            va='center', ha=ha,
            fontsize=8,
            fontweight='medium',
            color=COLORS['text_primary']
        )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('1-Day Return (%)', fontsize=9, color=COLORS['text_secondary'])
    ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['navy_800'], pad=12)
    
    ax.axvline(x=0, color=COLORS['text_muted'], linewidth=1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['border'])
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none', dpi=150)
    plt.close(fig)
    
    buffer.seek(0)
    svg_data = buffer.getvalue().decode('utf-8')
    
    encoded = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def generate_all_charts(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Generate all available charts from portfolio data.
    
    Args:
        data: Portfolio report data dict containing:
            - top_contributors: List of contributor dicts
            - top_detractors: List of detractor dicts
            - regional_breakdown: List of regional exposure dicts
            - holdings_detail: List of holding dicts
            
    Returns:
        Dict with chart data URIs
    """
    charts = {}
    
    # Contributors/Detractors chart
    contributors = data.get('top_contributors', [])
    detractors = data.get('top_detractors', [])
    if contributors or detractors:
        charts['contribution'] = generate_contribution_chart(contributors, detractors)
    
    # Regional chart
    regional = data.get('regional_breakdown', [])
    if regional:
        charts['regional'] = generate_regional_chart(regional)
    
    # Holdings heatmap
    holdings = data.get('holdings_detail', [])
    if holdings:
        charts['holdings'] = generate_holdings_heatmap(holdings)
    
    return charts
