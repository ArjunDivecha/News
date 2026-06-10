#!/usr/bin/env python3
"""
=============================================================================
LLM UTILITIES - Phase 2 Portfolio Reports
=============================================================================

Wrapper functions for LLM providers.
- Claude Opus 4.6 for report generation
- Claude Haiku for ETF classification

USAGE:
    from utils.llm import generate_report, classify_with_haiku
=============================================================================
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Runtime configuration (Opus default with bounded latency controls)
DEFAULT_ANTHROPIC_TIMEOUT_SECONDS = float(os.getenv('ANTHROPIC_TIMEOUT_SECONDS', '90'))
DEFAULT_REPORT_MODEL = os.getenv('PHASE2_REPORT_MODEL', 'claude-opus-4-6')
DEFAULT_REPORT_MAX_TOKENS = int(os.getenv('PHASE2_REPORT_MAX_TOKENS', '2600'))
FALLBACK_REPORT_MODEL = os.getenv('PHASE2_REPORT_FALLBACK_MODEL', '')

# Import Anthropic
try:
    from anthropic import Anthropic
    _anthropic_client = None
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    _anthropic_client = None

# Model configurations
MODELS = {
    'report': DEFAULT_REPORT_MODEL,
    'report_fallback': FALLBACK_REPORT_MODEL,
    'classify': 'claude-haiku-4-5-20251001',  # ETF classification
}


def _get_anthropic():
    """Lazy load Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None and ANTHROPIC_AVAILABLE:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            # Enforce client-level timeout so long generations fail fast instead
            # of appearing hung forever.
            try:
                _anthropic_client = Anthropic(
                    api_key=api_key,
                    timeout=DEFAULT_ANTHROPIC_TIMEOUT_SECONDS,
                )
            except TypeError:
                # Backward compatibility with older SDK signatures.
                _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def generate_anthropic(system_prompt: str, user_prompt: str,
                       model: str = None, max_tokens: int = 8000) -> Dict:
    """
    Generate text using Anthropic Claude.
    
    Returns:
        Dict with 'content', 'model', 'tokens_input', 'tokens_output', 'time_ms'
    """
    if model is None:
        model = MODELS['report']
        
    client = _get_anthropic()
    if client is None:
        return {'error': 'Anthropic client not available', 'model': model}
    
    start = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        return {
            'content': response.content[0].text,
            'model': model,
            'provider': 'anthropic',
            'tokens_input': response.usage.input_tokens,
            'tokens_output': response.usage.output_tokens,
            'time_ms': elapsed_ms,
        }
        
    except Exception as e:
        return {'error': str(e), 'model': model, 'provider': 'anthropic'}


def generate_report(system_prompt: str, user_prompt: str,
                   max_tokens: int = None) -> Dict:
    """
    Generate a portfolio report using the configured primary model.
    
    Args:
        system_prompt: System instructions
        user_prompt: User prompt with data
        max_tokens: Maximum tokens to generate (defaults to PHASE2_REPORT_MAX_TOKENS)
        
    Returns:
        Dict with generation result
    """
    if max_tokens is None:
        max_tokens = DEFAULT_REPORT_MAX_TOKENS

    primary_model = MODELS['report']
    result = generate_anthropic(system_prompt, user_prompt, primary_model, max_tokens)

    # Fast fallback if primary model fails or times out.
    if 'error' in result:
        fallback_model = MODELS.get('report_fallback')
        if fallback_model and fallback_model != primary_model:
            fallback_result = generate_anthropic(system_prompt, user_prompt, fallback_model, max_tokens)
            if 'error' not in fallback_result:
                fallback_result['fallback_from'] = primary_model
                return fallback_result

    return result


def get_report_model() -> str:
    """Return currently configured report model."""
    return MODELS['report']


def classify_with_haiku(ticker: str, name: str, description: str = None,
                        category: str = None, metadata: dict = None) -> Optional[Dict]:
    """
    Classify an ETF/asset using Claude Haiku.
    
    Args:
        ticker: Asset ticker
        name: Asset name
        description: Asset description (optional)
        category: ETF category from yfinance (optional)
        metadata: Additional metadata dict
        
    Returns:
        Dict with tier1, tier2, tier3_tags or None if failed
    """
    # Handle both module and standalone execution
    try:
        from .taxonomy import HAIKU_SYSTEM_PROMPT
    except ImportError:
        from taxonomy import HAIKU_SYSTEM_PROMPT
    
    # Build user message
    metadata_str = ""
    if metadata:
        for k, v in metadata.items():
            if v:
                metadata_str += f"{k.replace('_', ' ').title()}: {v}\n"
    
    user_message = f"""Classify this asset:

Ticker: {ticker}
Name: {name}
{f'Category: {category}' if category else ''}
{metadata_str}
Description: {description if description else '(No description available)'}

Respond with ONLY valid JSON, no other text."""

    client = _get_anthropic()
    if client is None:
        print(f"  ❌ Anthropic client not available for {ticker}")
        return None
    
    try:
        response = client.messages.create(
            model=MODELS['classify'],
            max_tokens=500,
            system=HAIKU_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ],
        )
        
        response_text = response.content[0].text.strip()
        
        # Extract JSON if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  ❌ API error for {ticker}: {e}")
        return None


def test_llm():
    """Test LLM connectivity."""
    print("Testing LLM utilities...")
    
    # Test Haiku classification
    print("\n[1] Testing Haiku classification...")
    result = classify_with_haiku(
        ticker="EWZ",
        name="iShares MSCI Brazil ETF",
        description="Tracks the MSCI Brazil Index, providing exposure to Brazilian equities",
        category="Latin America Stock"
    )
    if result:
        print(f"  ✓ EWZ classified as: {result.get('tier1')} / {result.get('tier2')}")
        print(f"    Tags: {result.get('tier3_tags')}")
    else:
        print("  ✗ Classification failed")
    
    # Test report generation (short test)
    print("\n[2] Testing Opus report generation...")
    result = generate_report(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, portfolio!' and nothing else.",
        max_tokens=50
    )
    if 'error' not in result:
        print(f"  ✓ Response: {result['content'][:50]}...")
        print(f"    Model: {result['model']}, Time: {result['time_ms']}ms")
    else:
        print(f"  ✗ Error: {result['error']}")
    
    print("\nLLM utilities test complete.")


if __name__ == "__main__":
    test_llm()
