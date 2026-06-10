#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: llm.py
=============================================================================

INPUT FILES:
    - report/prompts/system.md   (the system prompt)
    - .env                       (ANTHROPIC_API_KEY)

OUTPUT FILES:
    (none - returns report text + usage metadata to main.py)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Single LLM call for the unified report, using Claude Opus 4.6 with
    extended thinking. Bounded retries with exponential backoff; fails
    loudly after the last attempt (no silent fallback models).

DEPENDENCIES:
    - anthropic

USAGE:
    from llm import generate_report
=============================================================================
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PATHS, SETTINGS


def load_system_prompt() -> str:
    path = PATHS["system_prompt"]
    if not path.exists():
        raise FileNotFoundError(f"System prompt missing: {path}")
    return path.read_text()


def generate_report(data_package: str) -> dict:
    """
    Call the LLM once with the full data package.

    Returns:
        dict(text, model, tokens_input, tokens_output, elapsed_ms)
    """
    import anthropic

    client = anthropic.Anthropic()   # key from env
    model = SETTINGS["model"]
    system_prompt = load_system_prompt()

    last_err = None
    for attempt in range(1, SETTINGS["llm_retries"] + 1):
        t0 = time.time()
        try:
            print(f"  LLM call attempt {attempt}/{SETTINGS['llm_retries']} "
                  f"(model={model}, thinking={SETTINGS['thinking_budget']} tokens)...")
            response = client.messages.create(
                model=model,
                max_tokens=SETTINGS["max_tokens"],
                thinking={"type": "enabled",
                          "budget_tokens": SETTINGS["thinking_budget"]},
                system=system_prompt,
                messages=[{"role": "user", "content": data_package}],
                timeout=SETTINGS["llm_timeout_s"],
            )
            text_blocks = [b.text for b in response.content
                           if getattr(b, "type", "") == "text"]
            text = "\n".join(text_blocks).strip()
            if not text:
                raise RuntimeError("LLM returned an empty report")
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f"  LLM done in {elapsed_ms/1000:.0f}s "
                  f"({response.usage.input_tokens} in / "
                  f"{response.usage.output_tokens} out tokens)")
            return {
                "text": text,
                "model": model,
                "tokens_input": response.usage.input_tokens,
                "tokens_output": response.usage.output_tokens,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as e:
            last_err = e
            wait = 15 * attempt
            print(f"  !! LLM attempt {attempt} failed: {e}")
            if attempt < SETTINGS["llm_retries"]:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"LLM failed after {SETTINGS['llm_retries']} attempts: {last_err}")


def extract_executive_summary(report_md: str) -> str:
    """Pull the Executive Summary section text out of the report markdown."""
    lines = report_md.splitlines()
    capture = False
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("## executive summary"):
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture:
            out.append(line)
    summary = "\n".join(out).strip()
    return summary if summary else report_md[:800]
