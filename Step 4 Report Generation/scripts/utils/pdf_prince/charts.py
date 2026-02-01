"""
=============================================================================
PRINCEXML PDF GENERATOR - Chart Generation
=============================================================================

PURPOSE:
Generate SVG charts using Matplotlib for embedding in PDF reports.

CHARTS:
- Horizontal bar chart for Asset Class performance
- Horizontal bar chart for Sector performance
- Horizontal bar chart for Strategy performance
- RSI distribution chart
- Momentum trends chart

VERSION: 1.0.0
CREATED: 2026-01-31
=============================================================================
"""

import io
import base64
from typing import List, Dict, Optional
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


def generate_bar_chart(
    data: List[Dict],
    title: str,
    value_key: str = 'return_1d',
    label_key: str = 'category',
    width: float = 7.5,
    height: float = 3.5,
    color_scheme: str = 'return'  # 'return', 'neutral', 'gradient'
) -> str:
    """
    Generate horizontal bar chart.
    
    Args:
        data: List of dicts with value and label keys
        title: Chart title
        value_key: Key for numeric values
        label_key: Key for labels
        width: Figure width in inches
        height: Figure height in inches
        color_scheme: 'return' (green/red), 'neutral' (gray), 'gradient'
        
    Returns:
        Base64-encoded SVG data URI
    """
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Sort by value
    data = sorted(data, key=lambda x: x.get(value_key, 0) or 0)
    
    labels = [str(d.get(label_key, 'Unknown')) for d in data]
    values = [float(d.get(value_key, 0) or 0) for d in data]
    
    # Determine colors based on scheme
    colors = []
    if color_scheme == 'return':
        for v in values:
            if v > 2.0:
                colors.append(COLORS['positive'])
            elif v > 0:
                colors.append(COLORS['positive_light'])
            elif v < -2.0:
                colors.append(COLORS['negative'])
            elif v < 0:
                colors.append(COLORS['negative_light'])
            else:
                colors.append(COLORS['neutral'])
    elif color_scheme == 'neutral':
        colors = [COLORS['neutral']] * len(values)
    else:  # gradient
        # Use gradient from negative to positive
        max_abs = max(abs(v) for v in values) if values else 1
        for v in values:
            if v >= 0:
                intensity = min(v / max_abs, 1.0) if max_abs > 0 else 0
                colors.append(plt.cm.Greens(0.3 + 0.7 * intensity))
            else:
                intensity = min(abs(v) / max_abs, 1.0) if max_abs > 0 else 0
                colors.append(plt.cm.Reds(0.3 + 0.7 * intensity))
    
    # Create bars
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width_val = bar.get_width()
        label_x = width_val + (abs(width_val) * 0.05) if width_val >= 0 else width_val - (abs(width_val) * 0.05)
        ha = 'left' if width_val >= 0 else 'right'
        
        # Format value
        if abs(val) >= 1:
            label_text = f'{val:+.2f}%'
        else:
            label_text = f'{val:+.3f}%'
        
        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            label_text,
            va='center', ha=ha,
            fontsize=8,
            fontweight='medium',
            color=COLORS['text_primary']
        )
    
    # Style axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('1-Day Return (%)', fontsize=9, color=COLORS['text_secondary'])
    ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['text_primary'], pad=12)
    
    # Zero line
    ax.axvline(x=0, color=COLORS['text_muted'], linewidth=1, linestyle='-')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['border'])
    
    # Grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to SVG
    buffer = io.BytesIO()
    plt.savefig(buffer, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    buffer.seek(0)
    svg_data = buffer.getvalue().decode('utf-8')
    
    # Return as data URI for embedding
    encoded = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def generate_asset_class_chart(sections: List[Dict]) -> Optional[str]:
    """Generate Asset Class performance chart from report sections."""
    for section in sections:
        if section.get('title') == 'Asset Class Dashboard':
            tables = section.get('tables', [])
            if tables:
                table = tables[0]  # First table
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                # Find indices
                try:
                    asset_class_idx = headers.index('Asset Class')
                    return_idx = headers.index('1-Day')
                except ValueError:
                    return None
                
                # Extract data
                chart_data = []
                for row in rows:
                    if len(row) > max(asset_class_idx, return_idx):
                        asset_class = row[asset_class_idx]
                        return_str = row[return_idx]
                        # Parse return (remove % and +)
                        try:
                            return_val = float(return_str.replace('%', '').replace('+', '').strip())
                            chart_data.append({
                                'category': asset_class,
                                'return_1d': return_val
                            })
                        except (ValueError, AttributeError):
                            continue
                
                if chart_data:
                    return generate_bar_chart(
                        chart_data,
                        title="Asset Class Performance",
                        value_key='return_1d',
                        label_key='category'
                    )
    return None


def generate_sector_chart(sections: List[Dict]) -> Optional[str]:
    """Generate Sector performance chart from report sections."""
    for section in sections:
        if section.get('title') == 'Sector Dashboard':
            tables = section.get('tables', [])
            if tables:
                table = tables[0]
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                try:
                    sector_idx = headers.index('Sector')
                    return_idx = headers.index('Avg')  # or '1-Day'
                except ValueError:
                    try:
                        return_idx = headers.index('1-Day')
                    except ValueError:
                        return None
                
                chart_data = []
                for row in rows:
                    if len(row) > max(sector_idx, return_idx):
                        sector = row[sector_idx]
                        return_str = row[return_idx]
                        try:
                            return_val = float(return_str.replace('%', '').replace('+', '').strip())
                            chart_data.append({
                                'sector': sector,
                                'return_1d': return_val
                            })
                        except (ValueError, AttributeError):
                            continue
                
                if chart_data:
                    return generate_bar_chart(
                        chart_data,
                        title="Sector Performance",
                        value_key='return_1d',
                        label_key='sector'
                    )
    return None


def generate_rsi_chart(sections: List[Dict]) -> Optional[str]:
    """Generate RSI distribution chart from Momentum section."""
    for section in sections:
        if 'Momentum' in section.get('title', ''):
            tables = section.get('tables', [])
            for table in tables:
                if 'RSI' in table.get('title', ''):
                    headers = table.get('headers', [])
                    rows = table.get('rows', [])
                    
                    # Look for overbought/oversold columns
                    try:
                        category_idx = headers.index('Category')
                        overbought_idx = headers.index('Overbought (>70)')
                        oversold_idx = headers.index('Oversold (<30)')
                    except ValueError:
                        continue
                    
                    chart_data = []
                    for row in rows:
                        if len(row) > max(category_idx, overbought_idx, oversold_idx):
                            category = row[category_idx]
                            try:
                                overbought = int(row[overbought_idx])
                                oversold = int(row[oversold_idx])
                                net = overbought - oversold
                                chart_data.append({
                                    'category': category,
                                    'net_position': net
                                })
                            except (ValueError, TypeError):
                                continue
                    
                    if chart_data:
                        return generate_bar_chart(
                            chart_data,
                            title="RSI Distribution (Overbought - Oversold)",
                            value_key='net_position',
                            label_key='category',
                            color_scheme='neutral'
                        )
    return None


def generate_all_charts(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Generate all available charts from structured report data.
    
    Args:
        data: Structured report data dict
        
    Returns:
        Dict with chart data URIs (keys: 'asset_class', 'sector', 'rsi')
    """
    sections = data.get('sections', [])
    
    return {
        'asset_class': generate_asset_class_chart(sections),
        'sector': generate_sector_chart(sections),
        'rsi': generate_rsi_chart(sections),
    }
