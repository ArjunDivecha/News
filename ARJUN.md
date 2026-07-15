# ARJUN.md — What to do with the News repo

*From Fable 5, 2026-07-06. Read time ~4 min.*

## What this repo is worth

**Alive, and it's one of the few you actually use every day.** This is your
household daily report: live Schwab + IBKR + GMO + off-broker sleeves, ~800
ETFs from Yahoo, real analytics, a Fable-written narrative, emailed as a PDF at
1:05 PM. It is not superseded by anything in the ecosystem — it's the consumer,
not a duplicate. The engineering is genuinely good (pure-function math, 105
passing tests, fail-loud discipline). The weak spot is not the code that runs
when you're watching; it's what happens on the **unattended launchd run when
something quietly breaks** — and that just got more expensive, because Fable
moves to API pricing tomorrow.

The one thing to internalize: the daily pipeline has failed silently or
semi-silently several times this month (PATH, file-descriptor limit, coverage,
and on 2026-07-06 the Claude CLI wasn't found so the report ran on **metered
API** and skipped Fable's Desk — with no alert). Each was fixed, but there's
still no tripwire that tells you "today's report is degraded."

## Extensions ranked by value ÷ effort

1. **Close the silent billing/quality leak (do this before tomorrow).** Highest
   value right now. If the launchd Claude CLI resolution breaks again after
   Fable→API pricing, you'll be billed at API rates and lose the Desk with zero
   signal. *First step:* the 5-minute version — make `report/run_daily.sh`
   hard-fail (or email a one-line ALERT) when `command -v claude` is empty. The
   real version is FABLE.md's P1: a run manifest + `DEGRADED` prefix on the
   email subject. *Reuse:* `report/notify.py`.

2. **Backtest Fable's Desk ideas and the ASADO signals as a real strategy.**
   You already persist every Desk idea (`desk_ideas` table: conviction, horizon,
   invalidation) and `asado.py` reads 34-country value/momentum/FX/risk
   z-scores. You own the QuantConnect/LEAN `backtest` skill on *exactly* the
   34-country-ETF universe. Find out whether the judgment layer has edge before
   you act on it. *First step:* export `desk_ideas` → a signals frame → feed the
   backtest harness (equal-weight benchmark, Full/5y/3y/1y). *Reuse:* backtest
   skill, ASADO, `desk_ideas`.

3. **Desk hit-rate dashboard.** The Desk grades its own open ideas daily; that
   scorecard is buried in `report.db`. A one-page view (hit rate, P&L since
   idea, open vs graded) tells you whether to trust it — and gates whether #2 is
   worth the compute. *First step:* read `desk_ideas`, render a static page,
   deploy to Vercel. *Reuse:* Vercel (authenticated), `report/db.py`.

4. **Feed the daily insights into the personal-knowledge MCP.** Every day's
   Executive Summary + "Worth Knowing" bullets are institutional memory you
   currently discard after the email. Forward them so "what did I flag about EM
   in June" is queryable — and so future reports can pull your own prior takes.
   The continuity system already stores exec summaries in `report.db`; this just
   posts them onward. *First step:* after `db.save_report`, create a
   personal-knowledge entry. *Reuse:* personal-knowledge MCP.

5. **Monthly market-diary chapter via book-ghostwriter.** A year of daily
   Fable-written commentary is raw material for a first-person market diary in
   your voice. *First step:* monthly job that concatenates the month's exec
   summaries and runs the book-ghostwriter skill. *Reuse:* book-ghostwriter,
   `report.db`. (Wait until you have a few months of Fable output.)

6. **Decide who owns GDELT.** `asado.py` consumes a GDELT news pulse; you also
   have standalone `GDELT` and `T2 GDELT` repos. Either News reads their output
   or they're the canonical source — don't maintain two GDELT ingests. *First
   step:* diff the GDELT columns in `asado.duckdb` against the GDELT repos'
   output. *Reuse:* GDELT/T2 GDELT repos, ASADO.

## Quick wins (< 1 hour)

- **Hard-fail on missing Claude CLI in `run_daily.sh`** — one edit, closes the
  silent-billing hole today, ahead of the pricing change.
- **Fix stale docs:** `report/README.md` says "35 tests" / "25 tests" (actual
  105); `report/main.py:34` docstring still says "Claude Opus" (it's Fable now).
- **Move broker secrets out of the Dropbox-synced `.env`** (see below).

## What NOT to do

- **Don't re-add Bloomberg / a terminal dependency to the daily path.** The
  ETF-only Yahoo path is *why* the daily report is 2.5 minutes and runs
  unattended; the terminal dependency is exactly what made the old
  Phase-0/Step-4 chain fragile. Keep Bloomberg for universe construction only.
- **Don't add more report sections.** It's already six sections + tag views +
  scenarios + the Desk — dense. The marginal value of another section is below
  the marginal value of the reliability work above. Invest in depth and
  trustworthiness per section, not in more sections.

## One flag worth your attention

`.env` at the repo root holds your **Schwab username + password + TOTP secret**
and the **IBKR Flex token** in plaintext, inside a Dropbox-synced folder. It's
gitignored (good — not in git), but Dropbox sync means those broker credentials
sit on Dropbox's servers and every synced device, and the password + TOTP seed
*together* defeat your 2FA. Consider pulling at least `SCHWAB_PASSWORD`,
`SCHWAB_TOTP_SECRET`, and `IBKR_FLEX_TOKEN` from 1Password (you have the
`1password-credentials` skill) or macOS Keychain instead.
