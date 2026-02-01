#!/usr/bin/env python3
"""
=============================================================================
LLM API UTILITIES
=============================================================================

PURPOSE:
Wrapper functions for LLM providers. Primary focus: Claude Opus 4.5.
Other providers (OpenAI, Google) are kept for legacy/comparison purposes.

USAGE:
    from utils.llm import generate_report, generate_parallel
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load credentials
sys.path.insert(0, '/Users/arjundivecha/python_utils')
try:
    from onepassword_credentials import load_credentials
    ONEPASS_AVAILABLE = True
except ImportError:
    ONEPASS_AVAILABLE = False

# Load from .env if onepassword not available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# LLM clients (lazy loaded)
_openai_client = None
_anthropic_client = None
_google_model = None

# Model configurations
# Primary: Claude Opus 4.5 for daily reports
MODELS = {
    'openai': {
        'daily': 'gpt-5.2',  # Legacy/comparison only
        'flash': 'gpt-4o-mini',
    },
    'anthropic': {
        'daily': 'claude-opus-4-5-20251101',  # PRIMARY: Claude Opus 4.5
        'flash': 'claude-haiku-4-5-20251001',
    },
    'google': {
        'daily': 'gemini-3-pro-preview',  # Legacy/comparison only
        'flash': 'gemini-2.0-flash',
    },
}


def _get_openai():
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except Exception as e:
            print(f"WARNING: Could not initialize OpenAI: {e}")
            return None
    return _openai_client


def _get_anthropic():
    """Lazy load Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            from anthropic import Anthropic
            _anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except Exception as e:
            print(f"WARNING: Could not initialize Anthropic: {e}")
            return None
    return _anthropic_client


def _get_google(model_name: str):
    """Lazy load Google Generative AI model."""
    global _google_model
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"WARNING: Could not initialize Google: {e}")
        return None


def generate_openai(system_prompt: str, user_prompt: str, 
                    model: str = 'gpt-4o', max_tokens: int = 4000) -> Dict:
    """
    Generate text using OpenAI.
    
    Returns:
        Dict with 'content', 'model', 'tokens_input', 'tokens_output', 'time_ms'
    """
    client = _get_openai()
    if client is None:
        return {'error': 'OpenAI client not available', 'model': model}
    
    start = time.time()
    try:
        # GPT-5.2 requires max_completion_tokens instead of max_tokens
        params = {
            'model': model,
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            'temperature': 0.7,
        }
        
        if model.startswith('gpt-5'):
            params['max_completion_tokens'] = max_tokens
        else:
            params['max_tokens'] = max_tokens
        
        response = client.chat.completions.create(**params)
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        return {
            'content': response.choices[0].message.content,
            'model': model,
            'provider': 'openai',
            'tokens_input': response.usage.prompt_tokens,
            'tokens_output': response.usage.completion_tokens,
            'time_ms': elapsed_ms,
        }
        
    except Exception as e:
        return {'error': str(e), 'model': model, 'provider': 'openai'}


def generate_anthropic(system_prompt: str, user_prompt: str,
                       model: str = 'claude-sonnet-4-5-20250929', 
                       max_tokens: int = 4000) -> Dict:
    """
    Generate text using Anthropic.
    
    Returns:
        Dict with 'content', 'model', 'tokens_input', 'tokens_output', 'time_ms'
    """
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


def generate_google(system_prompt: str, user_prompt: str,
                    model: str = 'gemini-2.5-pro-preview-05-06',
                    max_tokens: int = 4000) -> Dict:
    """
    Generate text using Google Gemini.
    
    Returns:
        Dict with 'content', 'model', 'time_ms'
    """
    model_instance = _get_google(model)
    if model_instance is None:
        return {'error': 'Google client not available', 'model': model}
    
    start = time.time()
    try:
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = model_instance.generate_content(
            full_prompt,
            generation_config={
                'max_output_tokens': max_tokens,
                'temperature': 0.7,
            }
        )
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        return {
            'content': response.text,
            'model': model,
            'provider': 'google',
            'tokens_input': None,  # Gemini doesn't easily expose this
            'tokens_output': None,
            'time_ms': elapsed_ms,
        }
        
    except Exception as e:
        return {'error': str(e), 'model': model, 'provider': 'google'}


def generate_report(system_prompt: str, user_prompt: str, 
                   provider: str = 'anthropic',  # Opus 4.5
                   report_type: str = 'daily',
                   max_tokens: int = 4000) -> Dict:
    """
    Generate a report using the specified provider.
    
    Args:
        system_prompt: System instructions
        user_prompt: User prompt with data
        provider: 'anthropic' (Opus 4.5) - other providers kept for legacy
        report_type: 'daily' or 'flash' (determines model selection)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dict with generation result
    """
    model = MODELS.get(provider, {}).get(report_type)
    if model is None:
        return {'error': f'Unknown provider/report_type: {provider}/{report_type}'}
    
    if provider == 'openai':
        return generate_openai(system_prompt, user_prompt, model, max_tokens)
    elif provider == 'anthropic':
        return generate_anthropic(system_prompt, user_prompt, model, max_tokens)
    elif provider == 'google':
        return generate_google(system_prompt, user_prompt, model, max_tokens)
    else:
        return {'error': f'Unknown provider: {provider}'}


def generate_parallel(system_prompt: str, user_prompt: str,
                      providers: List[str] = ['anthropic'],  # Opus only by default
                      report_type: str = 'daily',
                      max_tokens: int = 4000) -> Dict[str, Dict]:
    """
    Generate reports from multiple providers in parallel.
    Default: Opus 4.5 only. Other providers kept for legacy/comparison.
    
    Args:
        system_prompt: System instructions
        user_prompt: User prompt with data
        providers: List of providers to use
        report_type: 'daily' or 'flash'
        max_tokens: Maximum tokens
        
    Returns:
        Dict mapping provider name to result dict
    """
    results = {}
    
    def _generate(provider):
        return provider, generate_report(
            system_prompt, user_prompt, provider, report_type, max_tokens
        )
    
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {executor.submit(_generate, p): p for p in providers}
        
        for future in as_completed(futures):
            try:
                provider, result = future.result()
                results[provider] = result
            except Exception as e:
                provider = futures[future]
                results[provider] = {'error': str(e), 'provider': provider}
    
    return results


def test_providers() -> Dict[str, bool]:
    """
    Test that all LLM providers are working.
    
    Returns:
        Dict mapping provider name to success boolean
    """
    print("\nTesting LLM Providers...")
    print("-" * 40)
    
    test_system = "You are a helpful assistant."
    test_user = "Say 'Hello World' and nothing else."
    
    results = {}
    
    for provider in ['openai', 'anthropic', 'google']:
        print(f"\n[{provider.upper()}]")
        try:
            result = generate_report(test_system, test_user, provider, 'flash', 50)
            if 'error' in result:
                print(f"  FAILED: {result['error']}")
                results[provider] = False
            else:
                print(f"  PASSED: '{result['content'][:50]}...'")
                print(f"  Time: {result.get('time_ms', 'N/A')}ms")
                results[provider] = True
        except Exception as e:
            print(f"  FAILED: {e}")
            results[provider] = False
    
    return results
