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
    Single LLM call for the unified report, using Claude Opus 4.8 with
    adaptive thinking, via STREAMING. Bounded retries with backoff; fails
    loudly after the last attempt (no silent fallback models).

    Streaming is required, not optional: max_tokens is large (max-effort
    thinking plus the full report) and the generation runs several minutes,
    which would trip the SDK's non-streaming timeout guard / idle-connection
    drop. A max_tokens stop is treated as a NON-retryable hard failure
    (FAIL IS FAIL) - a truncated report is never saved or rendered.

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


class ReportTruncatedError(RuntimeError):
    """The model hit max_tokens and the report is incomplete.

    Non-retryable on purpose: retrying the same request with the same budget
    reproduces the truncation. Raising loudly (rather than saving a partial
    report) is the FAIL-IS-FAIL behavior - the fix is to raise
    SETTINGS['max_tokens'] or lower thinking_effort, then rerun.
    """


def generate_report(data_package: str) -> dict:
    """
    Call the LLM once with the full data package, via streaming.

    Returns:
        dict(text, model, tokens_input, tokens_output, elapsed_ms, stop_reason)

    Raises:
        ReportTruncatedError if the model hit max_tokens (report incomplete).
        RuntimeError if all retryable attempts fail.
    """
    import anthropic

    # with_options(timeout=) is the documented way to set a per-request timeout
    # on the streaming helper; the long generation lives entirely inside it.
    client = anthropic.Anthropic().with_options(
        timeout=SETTINGS["llm_timeout_s"])   # key from env
    model = SETTINGS["model"]
    effort = SETTINGS["thinking_effort"]
    system_prompt = load_system_prompt()

    last_err = None
    for attempt in range(1, SETTINGS["llm_retries"] + 1):
        t0 = time.time()
        try:
            print(f"  LLM call attempt {attempt}/{SETTINGS['llm_retries']} "
                  f"(model={model}, thinking=adaptive/{effort}, "
                  f"max_tokens={SETTINGS['max_tokens']}, streaming)...")
            with client.messages.stream(
                model=model,
                max_tokens=SETTINGS["max_tokens"],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                system=system_prompt,
                messages=[{"role": "user", "content": data_package}],
            ) as stream:
                response = stream.get_final_message()

            elapsed_ms = int((time.time() - t0) * 1000)
            stop_reason = getattr(response, "stop_reason", None)
            usage = response.usage
            block_types = [getattr(b, "type", "?") for b in response.content]

            # FAIL IS FAIL: a max_tokens stop means the report is truncated
            # (later sections silently dropped). Do NOT save/render a partial
            # report; fail loudly and non-retryably.
            if stop_reason == "max_tokens":
                raise ReportTruncatedError(
                    f"Report TRUNCATED: hit max_tokens="
                    f"{SETTINGS['max_tokens']} (output={usage.output_tokens} "
                    f"tokens, thinking_effort={effort}). Sections are missing. "
                    f"Raise SETTINGS['max_tokens'] or lower thinking_effort in "
                    f"report/config.py, then rerun.")

            text_blocks = [b.text for b in response.content
                           if getattr(b, "type", "") == "text"]
            text = "\n".join(text_blocks).strip()
            if not text:
                raise RuntimeError(
                    f"LLM returned an empty report "
                    f"(stop_reason={stop_reason}, blocks={block_types}, "
                    f"output_tokens={getattr(usage, 'output_tokens', '?')})")

            print(f"  LLM done in {elapsed_ms/1000:.0f}s "
                  f"({usage.input_tokens} in / {usage.output_tokens} out "
                  f"tokens, stop_reason={stop_reason})")
            return {
                "text": text,
                "model": model,
                "tokens_input": usage.input_tokens,
                "tokens_output": usage.output_tokens,
                "elapsed_ms": elapsed_ms,
                "stop_reason": stop_reason,
            }
        except ReportTruncatedError:
            # Deterministic failure - retrying truncates again. Surface it.
            raise
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
