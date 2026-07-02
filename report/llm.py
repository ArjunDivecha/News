#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: llm.py
=============================================================================

INPUT FILES:
    - report/prompts/system.md   (the system prompt)
    - Claude CLI subscription auth (primary)
    - .env                       (ANTHROPIC_API_KEY, direct API fallback only)

OUTPUT FILES:
    (none - returns report text + usage metadata to main.py)

VERSION: 1.0
LAST UPDATED: 2026-06-09
AUTHOR: Arjun Divecha

DESCRIPTION:
    Single LLM call for the unified report, using Claude Fable 5 (Anthropic's
    most capable model) via the Claude CLI subscription path first. The direct
    Anthropic SDK streaming path is a fallback only (with a server-side
    refusal-fallback to Opus 4.8 on Fable). Bounded retries with backoff;
    fails loudly after the last attempt.

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

import json
import os
import shutil
import subprocess
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


def _project_anthropic_api_key() -> str | None:
    """Prefer this repo's .env key over any stale shell-level key."""
    from dotenv import dotenv_values

    env_path = Path(__file__).resolve().parents[1] / ".env"
    values = dotenv_values(env_path)
    key = (values.get("ANTHROPIC_API_KEY") or "").strip()
    return key or None


def _generate_report_claude_cli(data_package: str, model: str,
                                effort: str, system_prompt: str) -> dict:
    """Generate via Claude CLI so subscription auth is primary."""
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("Claude CLI not found on PATH")

    cli_model = SETTINGS.get("cli_model") or model
    env = os.environ.copy()
    # Force Claude Code's subscription/keychain auth path. If this remains set,
    # Claude CLI may use direct API-key billing instead of the user's plan.
    env.pop("ANTHROPIC_API_KEY", None)

    cmd = [
        claude,
        "-p",
        "--model", cli_model,
        "--effort", effort,
        "--output-format", "json",
        "--no-session-persistence",
        "--tools", "",
        "--system-prompt", system_prompt,
    ]
    t0 = time.time()
    print(f"  LLM call via Claude CLI "
          f"(model={cli_model}, effort={effort}, API key stripped)...")
    proc = subprocess.run(
        cmd,
        input=data_package,
        text=True,
        capture_output=True,
        timeout=SETTINGS["llm_timeout_s"],
        env=env,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Claude CLI failed with exit code {proc.returncode}: "
            f"{detail[-2000:]}")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Claude CLI returned non-JSON output: {proc.stdout[:2000]}") from e

    if payload.get("is_error"):
        raise RuntimeError(
            f"Claude CLI error: {payload.get('api_error_status') or payload}")

    text = str(payload.get("result") or "").strip()
    stop_reason = payload.get("stop_reason") or payload.get("terminal_reason")
    usage = payload.get("usage") or {}
    tokens_input = int(usage.get("input_tokens") or 0)
    tokens_output = int(usage.get("output_tokens") or 0)

    if stop_reason == "max_tokens":
        raise ReportTruncatedError(
            f"Report TRUNCATED: Claude CLI hit max_tokens "
            f"(output={tokens_output} tokens, effort={effort}).")
    if not text:
        raise RuntimeError(
            f"Claude CLI returned an empty report "
            f"(stop_reason={stop_reason}, output_tokens={tokens_output})")

    print(f"  LLM done via Claude CLI in {elapsed_ms/1000:.0f}s "
          f"({tokens_input} in / {tokens_output} out tokens, "
          f"stop_reason={stop_reason})")
    return {
        "text": text,
        "model": payload.get("model") or SETTINGS.get("model", model),
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "elapsed_ms": elapsed_ms,
        "stop_reason": stop_reason,
    }


def _generate_report_anthropic_api(data_package: str, model: str,
                                   effort: str, system_prompt: str) -> dict:
    """Direct Anthropic streaming fallback. This may incur API charges.

    On Claude Fable 5 this opts into the server-side refusal fallback: if a
    safety classifier declines the request (finance content — very unlikely,
    but possible as a false positive), the API transparently re-serves it on
    Opus 4.8 inside the same call, so the daily report never dies on a
    classifier hiccup. Fable also requires: no thinking config (always on —
    adaptive is the only accepted explicit value) and 30-day data retention.
    """
    import anthropic

    api_key = _project_anthropic_api_key()
    client_kwargs = {"api_key": api_key} if api_key else {}
    client = anthropic.Anthropic(**client_kwargs).with_options(
        timeout=SETTINGS["llm_timeout_s"])

    kwargs = dict(
        model=model,
        max_tokens=SETTINGS["max_tokens"],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=system_prompt,
        messages=[{"role": "user", "content": data_package}],
    )
    is_fable = model.startswith("claude-fable") or model.startswith("claude-mythos")
    if is_fable:
        kwargs["betas"] = ["server-side-fallback-2026-06-01"]
        kwargs["fallbacks"] = [{"model": "claude-opus-4-8"}]
        stream_fn = client.beta.messages.stream
    else:
        stream_fn = client.messages.stream

    print(f"  LLM API fallback "
          f"(model={model}, thinking=adaptive/{effort}, "
          f"max_tokens={SETTINGS['max_tokens']}, streaming"
          f"{', refusal-fallback=opus-4-8' if is_fable else ''})...")
    with stream_fn(**kwargs) as stream:
        response = stream.get_final_message()

    stop_reason = getattr(response, "stop_reason", None)

    if stop_reason == "refusal":
        # With the server-side fallback, a final refusal means the whole chain
        # (Fable AND Opus 4.8) declined — surface it loudly, never save a
        # partial. stop_details carries the classifier category when present.
        details = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"LLM REFUSED the request (stop_reason=refusal, "
            f"category={getattr(details, 'category', None)}, "
            f"explanation={getattr(details, 'explanation', None)})")

    served_by = getattr(response, "model", model)
    if is_fable and served_by and not served_by.startswith("claude-fable"):
        print(f"  !! Fable declined; report served by fallback model "
              f"{served_by}")
    usage = response.usage
    block_types = [getattr(b, "type", "?") for b in response.content]

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

    return {
        "text": text,
        "model": served_by,          # the model that actually wrote the report
        "tokens_input": usage.input_tokens,
        "tokens_output": usage.output_tokens,
        "elapsed_ms": 0,
        "stop_reason": stop_reason,
    }


def generate_report(data_package: str) -> dict:
    """
    Call the LLM once with the full data package, via streaming.

    Returns:
        dict(text, model, tokens_input, tokens_output, elapsed_ms, stop_reason)

    Raises:
        ReportTruncatedError if the model hit max_tokens (report incomplete).
        RuntimeError if all retryable attempts fail.
    """
    model = SETTINGS["model"]
    effort = SETTINGS["thinking_effort"]
    system_prompt = load_system_prompt()
    backend = SETTINGS.get("llm_backend", "claude_cli")

    last_err = None
    for attempt in range(1, SETTINGS["llm_retries"] + 1):
        t0 = time.time()
        try:
            print(f"  LLM call attempt {attempt}/{SETTINGS['llm_retries']} "
                  f"(backend={backend})")
            if backend == "claude_cli":
                return _generate_report_claude_cli(
                    data_package, model, effort, system_prompt)
            result = _generate_report_anthropic_api(
                data_package, model, effort, system_prompt)
            elapsed_ms = int((time.time() - t0) * 1000)
            print(f"  LLM done in {elapsed_ms/1000:.0f}s "
                  f"({result['tokens_input']} in / "
                  f"{result['tokens_output']} out tokens, "
                  f"stop_reason={result['stop_reason']})")
            result["elapsed_ms"] = elapsed_ms
            return result
        except ReportTruncatedError:
            # Deterministic failure - retrying truncates again. Surface it.
            raise
        except Exception as e:
            last_err = e
            if (backend == "claude_cli"
                    and SETTINGS.get("llm_api_fallback", True)
                    and attempt == SETTINGS["llm_retries"]):
                try:
                    print(f"  !! Claude CLI failed: {e}")
                    print("  Falling back to direct Anthropic streaming API...")
                    result = _generate_report_anthropic_api(
                        data_package, model, effort, system_prompt)
                    result["elapsed_ms"] = int((time.time() - t0) * 1000)
                    return result
                except ReportTruncatedError:
                    raise
                except Exception as fallback_err:
                    last_err = fallback_err
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
