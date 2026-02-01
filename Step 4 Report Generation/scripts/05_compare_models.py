#!/usr/bin/env python3
"""
=============================================================================
MODEL COMPARISON TOOL (Opus Only)
=============================================================================

PURPOSE:
Generate reports using Claude Opus 4.5. Legacy comparison functionality
kept for historical reference.

USAGE:
    python scripts/05_compare_models.py                    # Today's date (Opus)
    python scripts/05_compare_models.py --date 2026-01-30 # Specific date
    python scripts/05_compare_models.py --structured    # Use structured JSON output
    python scripts/05_compare_models.py --models anthropic  # Opus only (default)

OUTPUT:
    outputs/comparison/{date}/
      - {model}_report.json (structured data)
      - {model}_report.md (markdown fallback)
      - {model}_report.pdf (PDF)
      - comparison_summary.txt (quick comparison)
=============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.db import get_db
from utils.llm import generate_parallel, generate_report as gen_single
import importlib.util
spec = importlib.util.spec_from_file_location("report_gen", SCRIPT_DIR / "03_generate_daily_report.py")
report_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_gen)
prepare_data_summary = report_gen.prepare_data_summary
inject_data_into_prompt = report_gen.inject_data_into_prompt
get_last_trading_day = report_gen.get_last_trading_day
load_prompt_template = report_gen.load_prompt_template

# Import structured parser
try:
    from utils.pdf_prince.parser import parse_structured_report, extract_json_from_response
    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False
    print("WARNING: Structured parser not available")


OUTPUT_DIR = PROJECT_DIR / "outputs" / "comparison"


def load_structured_prompt_template() -> tuple:
    """Load structured prompt template."""
    prompt_path = PROJECT_DIR / "prompts" / "daily_wrap_structured.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Structured prompt not found: {prompt_path}")
    
    content = prompt_path.read_text()
    
    # Split on "USER" marker
    if "USER" in content:
        parts = content.split("USER", 1)
        system_prompt = parts[0].replace("SYSTEM", "").strip()
        user_prompt = parts[1].strip() if len(parts) > 1 else ""
    else:
        system_prompt = content
        user_prompt = ""
    
    return system_prompt, user_prompt


def save_structured_output(provider: str, date: str, llm_response: str, 
                          output_dir: Path) -> Dict[str, Path]:
    """
    Save structured output from LLM.
    
    Returns:
        Dict with paths to saved files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    # Try to extract JSON
    if STRUCTURED_AVAILABLE:
        try:
            data = parse_structured_report(llm_response)
            data['report_date'] = datetime.strptime(date, '%Y-%m-%d').strftime('%B %d, %Y')
            
            # Save JSON
            json_path = output_dir / f"{provider}_report.json"
            json_path.write_text(json.dumps(data, indent=2))
            saved_files['json'] = json_path
            
            # Try to generate PDF with PrinceXML
            try:
                from utils.pdf_prince import convert_from_data
                pdf_path = output_dir / f"{provider}_report.pdf"
                result = convert_from_data(data, str(pdf_path))
                if result:
                    saved_files['pdf'] = Path(result)
            except Exception as e:
                print(f"  WARNING: Could not generate PDF: {e}")
                
        except Exception as e:
            print(f"  WARNING: Could not parse structured output: {e}")
            print(f"  Saving raw response as markdown...")
    
    # Save raw response as markdown fallback
    md_path = output_dir / f"{provider}_report.md"
    md_path.write_text(llm_response)
    saved_files['md'] = md_path
    
    return saved_files


def generate_comparison_summary(date: str, results: Dict[str, Dict], output_dir: Path):
    """Generate a summary file comparing all models."""
    summary_path = output_dir / "comparison_summary.txt"
    
    lines = [
        "=" * 70,
        f"MODEL COMPARISON SUMMARY - {date}",
        "=" * 70,
        "",
    ]
    
    for provider, result in results.items():
        if 'error' in result:
            lines.append(f"{provider.upper()}: ERROR - {result['error']}")
        else:
            content_len = len(result.get('content', ''))
            model = result.get('model', 'unknown')
            tokens_in = result.get('tokens_input') or 0
            tokens_out = result.get('tokens_output') or 0
            time_ms = result.get('time_ms') or 0
            
            lines.append(f"{provider.upper()} ({model}):")
            lines.append(f"  Content: {content_len:,} chars")
            lines.append(f"  Tokens: {tokens_in:,} in / {tokens_out:,} out")
            lines.append(f"  Time: {time_ms:,}ms ({time_ms/1000:.1f}s)")
            
            # Check if structured JSON was extracted
            if STRUCTURED_AVAILABLE:
                try:
                    data = extract_json_from_response(result['content'])
                    if data:
                        sections = len(data.get('sections', []))
                        tables = sum(len(s.get('tables', [])) for s in data.get('sections', []))
                        lines.append(f"  Structured: ✓ ({sections} sections, {tables} tables)")
                    else:
                        lines.append(f"  Structured: ✗ (could not extract JSON)")
                except:
                    lines.append(f"  Structured: ✗ (parse error)")
            
            lines.append("")
    
    lines.append("=" * 70)
    lines.append("FILES SAVED:")
    for provider, result in results.items():
        if 'saved_files' in result:
            for file_type, path in result['saved_files'].items():
                lines.append(f"  {provider}/{file_type}: {path.name}")
    
    summary_path.write_text("\n".join(lines))
    print(f"\nComparison summary saved: {summary_path}")


def compare_models(date: str, providers: List[str], structured: bool = False) -> Dict[str, Any]:
    """
    Generate reports with multiple models for comparison.
    
    Args:
        date: Date string (YYYY-MM-DD)
        providers: List of provider names
        structured: If True, use structured JSON output mode
        
    Returns:
        Dict with results from each provider
    """
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON FOR {date}")
    print(f"{'='*70}")
    print(f"Providers: {', '.join(providers)}")
    print(f"Mode: {'Structured JSON' if structured else 'Markdown'}")
    
    # Load prompt template
    if structured:
        print("\n[1/4] Loading structured prompt template...")
        system_prompt, user_prompt_template = load_structured_prompt_template()
    else:
        print("\n[1/4] Loading markdown prompt template...")
        system_prompt, user_prompt_template = load_prompt_template()
    
    # Prepare data
    print("\n[2/4] Preparing data summary...")
    data = prepare_data_summary(date)
    data['unusual_flags'] = ""  # Simplified for comparison
    
    # Inject data into prompt
    print("\n[3/4] Injecting data into prompt...")
    user_prompt = inject_data_into_prompt(user_prompt_template, data)
    
    # Generate reports
    print(f"\n[4/4] Generating reports from {len(providers)} provider(s)...")
    
    results = {}
    if len(providers) == 1:
        provider = providers[0]
        print(f"  Calling {provider}...")
        result = gen_single(system_prompt, user_prompt, provider, 'daily', 12000)
        results[provider] = result
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Success: {len(result.get('content', ''))} chars")
    else:
        print(f"  Calling {len(providers)} providers in parallel...")
        parallel_results = generate_parallel(system_prompt, user_prompt, providers, 'daily', 12000)
        results = parallel_results
        for provider, result in parallel_results.items():
            if 'error' in result:
                print(f"  [{provider}] ERROR: {result['error']}")
            else:
                print(f"  [{provider}] Success: {len(result.get('content', ''))} chars")
    
    # Save outputs
    print("\n[5/5] Saving outputs...")
    output_dir = OUTPUT_DIR / date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for provider, result in results.items():
        if 'content' in result and result['content']:
            print(f"  Saving {provider} output...")
            if structured:
                saved_files = save_structured_output(provider, date, result['content'], output_dir)
            else:
                # Save as markdown
                md_path = output_dir / f"{provider}_report.md"
                md_path.write_text(result['content'])
                saved_files = {'md': md_path}
            
            result['saved_files'] = saved_files
    
    # Generate comparison summary
    generate_comparison_summary(date, results, output_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare multiple LLM models")
    parser.add_argument("--date", type=str, default=None,
                       help="Date (YYYY-MM-DD). Defaults to last trading day.")
    parser.add_argument("--models", type=str, nargs='+',
                       default=['anthropic'],  # Opus only
                       choices=['anthropic'],  # Opus only
                       help="Model to use (Claude Opus 4.5 only)")
    parser.add_argument("--structured", action="store_true",
                       help="Use structured JSON output mode")
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        # If date explicitly provided, use it directly (don't override)
        report_date = args.date
    else:
        # Only auto-detect last trading day if no date specified
        report_date = get_last_trading_day(datetime.now().strftime('%Y-%m-%d'))
    
    try:
        results = compare_models(report_date, args.models, args.structured)
        
        # Check for errors
        errors = [p for p, r in results.items() if 'error' in r]
        if errors:
            print(f"\n⚠ Some providers failed: {', '.join(errors)}")
            return 1
        
        print("\n✓ Model comparison complete")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
