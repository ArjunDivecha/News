#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: fable_desk.py
=============================================================================

INPUT FILES:
    - report/prompts/fable_desk.md   (the Desk system prompt)
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/ASADO/Data/asado.duckdb
      (read-only, via asado.py)
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/report.db
      (open desk ideas for continuity/scorecard, via db.py)
    - Claude CLI subscription auth (PRIMARY backend — free under the plan)
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/.env
      (ANTHROPIC_API_KEY — the direct-API FALLBACK, used only when the CLI
       fails, e.g. the Fable plan quota is exhausted. This path costs money.)

OUTPUT FILES:
    - /Users/arjundivecha/Dropbox/AAA Backup/A Working/News/data/report.db
      (desk_ideas table: new ideas + grades applied to open ones)
    (the rendered section markdown is returned to main.py, which appends it
     to the daily report before PDF render)

VERSION: 2.0
LAST UPDATED: 2026-07-12
AUTHOR: Arjun Divecha

DESCRIPTION:
    Fable's Desk - the judgment-based second pass of the daily report.
    While the main report is deterministic (numbers-only, no news, no
    recommendations outside data-triggered Action Box items), the Desk is
    Fable with its brain switched on: live web search for world context,
    the ASADO 34-country signal snapshot, and an explicit license to
    propose idiosyncratic ideas - each with conviction, horizon and a
    falsifiable invalidation. Ideas are persisted to report.db and graded
    by the Desk itself on subsequent days (the scorecard), so the judgment
    stays accountable.

    Backends (v2): the Claude CLI (subscription auth, free) is tried FIRST.
    If it fails for ANY reason — most importantly an exhausted Fable plan
    quota, which returns HTTP 429 — the Desk automatically falls back to the
    direct Anthropic API with the SAME model and web search (Anthropic's
    native web_search server tool), so the section ships instead of being
    dropped. The API path costs money and says so loudly in the log.

    Additive-only: if BOTH backends fail, that is logged loudly and the daily
    report ships without the section. The Desk never sinks the report.

DEPENDENCIES:
    - claude CLI on PATH (subscription auth)
    - anthropic (API fallback)
    - duckdb (via asado.py; degrades gracefully if missing)

USAGE:
    from fable_desk import run_fable_desk
    section_md = run_fable_desk(asof, report_md, data_package)  # None on failure

    Standalone test (uses an existing report + package from outputs/):
    python3 report/fable_desk.py --date 2026-07-02
=============================================================================
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402  (reuses the project's .env API-key resolution)
from config import PATHS, SETTINGS

DESK_HEADING = "## Fable's Desk"
# Anthropic's native web_search SERVER tool — the API-fallback equivalent of
# the CLI's WebSearch. The _20260209 variant (dynamic filtering) is the current
# one; SETTINGS may pin the basic variant if a model doesn't support it.
DESK_WEB_SEARCH_TOOL = SETTINGS.get("desk_web_search_tool", "web_search_20260209")
_TRAILER_RE = re.compile(
    r"```(?:desk-json|json)\s*\n(\{.*?\})\s*\n```\s*$", re.DOTALL)


def load_desk_prompt() -> str:
    path = PATHS["desk_prompt"]
    if not path.exists():
        raise FileNotFoundError(f"Desk prompt missing: {path}")
    return path.read_text()


def build_desk_input(asof: str, report_md: str, data_package: str,
                     asado_snapshot: str | None,
                     open_ideas: list[dict]) -> str:
    """Assemble the user message for the Desk pass."""
    parts = [f"# FABLE'S DESK INPUT — {asof}\n"]

    parts.append("## OPEN DESK IDEAS (grade each in your scorecard "
                 "and the machine trailer)")
    if open_ideas:
        for i in open_ideas:
            parts.append(
                f"- id={i['id']} ({i['date']}): **{i['title']}** — "
                f"{i['action']} · conviction {i['conviction']} · "
                f"horizon {i['horizon']} · invalidation: {i['invalidation']}")
    else:
        parts.append("(none — this is the Desk's first edition or all prior "
                     "ideas are resolved; the scorecard can say so in one "
                     "line)")

    if asado_snapshot:
        parts.append("\n## ASADO WAREHOUSE SNAPSHOT")
        parts.append(asado_snapshot)
    else:
        parts.append("\n## ASADO WAREHOUSE SNAPSHOT\n(unavailable this run — "
                     "say so if it matters to an idea you would otherwise "
                     "have grounded in it)")

    parts.append("\n## TODAY'S DETERMINISTIC REPORT (verbatim)")
    parts.append(report_md)

    parts.append("\n## TODAY'S DATA PACKAGE (verbatim — the only source for "
                 "portfolio/market numbers)")
    parts.append(data_package)

    parts.append("\n---\nWrite today's Fable's Desk section now, per your "
                 "instructions. Use web search for live world context "
                 "before writing. End with the desk-json trailer.")
    return "\n".join(parts)


def parse_desk_output(text: str) -> tuple[str, list[dict], list[dict]]:
    """Split the Desk output into (rendered_md, new_ideas, grades).

    The machine trailer is REQUIRED; a missing/unparseable trailer raises
    (FAIL IS FAIL - we never silently lose idea tracking).
    """
    m = _TRAILER_RE.search(text.strip())
    if not m:
        raise ValueError(
            "Desk output missing the required desk-json trailer - refusing "
            "to accept an untrackable Desk section.")
    payload = json.loads(m.group(1))
    ideas = payload.get("ideas") or []
    grades = payload.get("grades") or []
    for i in ideas:
        missing = [k for k in ("title", "action", "conviction", "horizon",
                               "invalidation") if not i.get(k)]
        if missing:
            raise ValueError(f"Desk idea missing fields {missing}: {i}")
    clean_md = text[:m.start()].strip()
    if not clean_md:
        raise ValueError("Desk output contained only the trailer, no prose.")
    return clean_md, ideas, grades


def _generate_desk_claude_cli(desk_input: str, system_prompt: str) -> str:
    """One Desk generation via Claude CLI with web tools enabled."""
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("Claude CLI not found on PATH")

    cli_model = SETTINGS.get("cli_model") or SETTINGS["model"]
    effort = SETTINGS.get("desk_effort", "high")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)   # subscription auth path

    cmd = [
        claude,
        "-p",
        "--model", cli_model,
        "--effort", effort,
        "--output-format", "json",
        "--no-session-persistence",
        # Web tools ONLY: the Desk may read the world, never touch the disk.
        "--tools", "WebSearch,WebFetch",
        "--system-prompt", system_prompt,
    ]
    t0 = time.time()
    print(f"  Desk call via Claude CLI (model={cli_model}, effort={effort}, "
          f"tools=WebSearch,WebFetch)...")
    proc = subprocess.run(
        cmd, input=desk_input, text=True, capture_output=True,
        timeout=SETTINGS.get("desk_timeout_s", 1500), env=env)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Desk CLI failed (exit {proc.returncode}): "
                           f"{detail[-2000:]}")
    payload = json.loads(proc.stdout)
    if payload.get("is_error"):
        raise RuntimeError(f"Desk CLI error: "
                           f"{payload.get('api_error_status') or payload}")
    text = str(payload.get("result") or "").strip()
    stop_reason = payload.get("stop_reason") or payload.get("terminal_reason")
    if stop_reason == "max_tokens":
        raise RuntimeError("Desk output TRUNCATED (max_tokens)")
    if not text:
        raise RuntimeError(f"Desk returned empty output "
                           f"(stop_reason={stop_reason})")
    usage = payload.get("usage") or {}
    print(f"  Desk done in {elapsed:.0f}s "
          f"({usage.get('input_tokens', 0)} in / "
          f"{usage.get('output_tokens', 0)} out tokens)")
    return text


def _is_quota_error(exc: Exception) -> bool:
    """Did the CLI fail because the Claude plan quota is exhausted (429)?

    Used for logging only — the fallback fires on ANY CLI failure, because a
    dropped Desk section is worse than an API charge either way.
    """
    text = str(exc).lower()
    return ("429" in text
            or "reached your" in text and "limit" in text
            or "usage limit" in text
            or "rate limit" in text
            or "quota" in text)


def _generate_desk_anthropic_api(desk_input: str, system_prompt: str) -> str:
    """Direct Anthropic API fallback for the Desk. THIS COSTS MONEY.

    Mirrors llm.py's API fallback: same model, streaming (the Desk runs for
    minutes), and — on Fable — the server-side refusal fallback to Opus 4.8 so
    a safety-classifier false positive on finance content can't kill the Desk.

    Web access is preserved with Anthropic's native web_search SERVER tool
    (executed on Anthropic's side; no client tool loop needed), which is what
    makes this a real substitute for the CLI's WebSearch/WebFetch.
    """
    import anthropic

    api_key = llm._project_anthropic_api_key()
    client_kwargs = {"api_key": api_key} if api_key else {}
    client = anthropic.Anthropic(**client_kwargs).with_options(
        timeout=SETTINGS.get("desk_timeout_s", 1500))

    model = SETTINGS["model"]
    effort = SETTINGS.get("desk_effort", "high")
    max_tokens = SETTINGS.get("desk_max_tokens") or SETTINGS["max_tokens"]
    is_fable = model.startswith("claude-fable") or model.startswith("claude-mythos")

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        system=system_prompt,
        messages=[{"role": "user", "content": desk_input}],
        tools=[{"type": DESK_WEB_SEARCH_TOOL, "name": "web_search",
                "max_uses": SETTINGS.get("desk_max_searches", 8)}],
    )
    if is_fable:
        # `fallbacks` is a top-level request param that the installed SDK
        # (0.83) does not yet type — pass it via extra_body, which is the
        # documented pre-GA-SDK path. Drop this once the SDK is upgraded.
        kwargs["betas"] = ["server-side-fallback-2026-06-01"]
        kwargs["extra_body"] = {"fallbacks": [{"model": "claude-opus-4-8"}]}
        stream_fn = client.beta.messages.stream
    else:
        stream_fn = client.messages.stream

    t0 = time.time()
    print(f"  !! Desk API FALLBACK — THIS COSTS MONEY "
          f"(model={model}, effort={effort}, tool={DESK_WEB_SEARCH_TOOL}"
          f"{', refusal-fallback=opus-4-8' if is_fable else ''})...")
    with stream_fn(**kwargs) as stream:
        response = stream.get_final_message()
    elapsed = time.time() - t0

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"Desk REFUSED by the model chain (stop_reason=refusal, "
            f"category={getattr(details, 'category', None)})")
    if stop_reason == "max_tokens":
        raise RuntimeError(
            f"Desk output TRUNCATED (max_tokens={max_tokens}) - refusing a "
            f"partial Desk section.")

    text = "\n".join(b.text for b in response.content
                     if getattr(b, "type", "") == "text").strip()
    if not text:
        raise RuntimeError(f"Desk API returned empty output "
                           f"(stop_reason={stop_reason})")
    usage = response.usage
    served_by = getattr(response, "model", model)
    print(f"  Desk done via API in {elapsed:.0f}s "
          f"({usage.input_tokens} in / {usage.output_tokens} out tokens, "
          f"served_by={served_by})")
    return text


def _generate_desk(desk_input: str, system_prompt: str) -> str:
    """CLI first (free, subscription auth); direct API on any CLI failure.

    The quota case is the one this exists for: when the Claude plan's Fable
    limit is hit the CLI returns 429, and before v2 the Desk section was simply
    dropped from the daily report. Now it switches to the API.
    """
    try:
        return _generate_desk_claude_cli(desk_input, system_prompt)
    except Exception as cli_err:
        if not SETTINGS.get("desk_api_fallback", True):
            raise
        reason = ("Claude CLI is OUT OF QUOTA" if _is_quota_error(cli_err)
                  else "Claude CLI failed")
        print(f"  !! {reason}: {str(cli_err)[:300]}")
        return _generate_desk_anthropic_api(desk_input, system_prompt)


def run_fable_desk(asof: str, report_md: str, data_package: str) -> str | None:
    """Full Desk pass. Returns the section markdown (WITH its heading) ready
    to append to the report, or None on any failure (additive-only)."""
    try:
        import asado
        import db

        system_prompt = load_desk_prompt()
        open_ideas = db.get_open_desk_ideas()
        snapshot = asado.build_asado_snapshot()
        desk_input = build_desk_input(asof, report_md, data_package,
                                      snapshot, open_ideas)

        raw = _generate_desk(desk_input, system_prompt)
        clean_md, ideas, grades = parse_desk_output(raw)

        max_ideas = SETTINGS.get("desk_max_ideas", 4)
        if len(ideas) > max_ideas:
            raise ValueError(f"Desk proposed {len(ideas)} ideas "
                             f"(contract max {max_ideas}) - rejecting.")

        # Pre-validate the section with the SAME table check the PDF renderer
        # applies to the full report, so a malformed Desk table rejects only
        # the Desk (additive-only), never the whole report at render time.
        import pdf as pdf_mod
        pdf_mod._validate_report_tables(clean_md)

        db.apply_desk_grades(grades)
        db.save_desk_ideas(asof, ideas)
        print(f"  Desk: {len(ideas)} new idea(s), {len(grades)} grade(s) "
              f"applied, ASADO snapshot "
              f"{'included' if snapshot else 'UNAVAILABLE'}")
        return f"{DESK_HEADING}\n\n{clean_md}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ⚠️  Fable's Desk SKIPPED (report ships without it): {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Desk run")
    parser.add_argument("--date", required=True,
                        help="Report date to run the Desk against "
                             "(expects existing outputs/unified files)")
    args = parser.parse_args()
    out_dir = PATHS["output_dir"]
    report_path = out_dir / f"Unified_Report_{args.date}.md"
    pkg_path = out_dir / f"Data_Package_{args.date}.md"
    if not report_path.exists() or not pkg_path.exists():
        sys.exit(f"Missing {report_path} or {pkg_path}")
    section = run_fable_desk(args.date, report_path.read_text(),
                             pkg_path.read_text())
    if section is None:
        sys.exit("Desk run FAILED")
    out_path = out_dir / f"Fable_Desk_{args.date}.md"
    out_path.write_text(section)
    print(f"\nDesk section written to {out_path}\n")
    print(section)
