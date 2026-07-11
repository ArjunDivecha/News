# llmchat.md - Project Context Log

This file is the shared memory between project sessions and agents.
It is append-only. Do not edit existing entries unless explicitly asked.
Each session appends a timestamped block at the bottom.

---
SESSION START: 2026-07-11 09:18 CDT | Agent: Claude Code
---

### Session Summary
Attempted to activate the IBKR Flex Web Service as the unattended IBKR
holdings path for the daily report; it triggered an IBKR account-level
lockout that blocked Arjun's Client Portal login. Flex is now REMOVED
from the pipeline entirely (commit 3aaef4d). IBKR holdings come from a
logged-in IB Gateway/TWS session only; Arjun will log in every ~24h.

### Decisions Made
- **Abandon IBKR Flex Web Service permanently** (Arjun's explicit call).
  Do not re-add it. Rationale below in Constraints & Gotchas.
- IBKR path is IB Gateway subprocess only (`report/ibkr_fetch.py` via
  `.venv-ibkr312`); when Gateway isn't logged in, the run uses the loudly
  flagged stale snapshot from `data/holdings.xlsx` — that is accepted
  behavior, not a bug.

### Architecture / Design
- Deleted `report/ibkr_flex.py`; stripped Flex branches/env keys from
  `report/holdings.py` and `report/config.py`; `.env` Flex vars removed
  (comment block documents why); `report/README.md` and repo `CLAUDE.md`
  now document the removal and the reason.
- Commit ddf48f6 (SendRequest retry-on-transient) is superseded by the
  removal in 3aaef4d but kept in history as reference.

### Constraints & Gotchas (the post-mortem — why Flex is banned)
- Flex error 1001 ("statement could not be generated, try again shortly")
  is documented-transient but in practice near-continuous for this account
  outside a narrow window right after a portal-side manual generation —
  the two API successes each came within ~1 min of a Client Portal "Run";
  everything else failed. Cached-statement serving, not a healthy path.
- Retrying 1001 (even modestly: ~10 requests in an hour, then 1 gentle
  probe per 30 min) trips error 1025 "Too many failed attempts" — an
  ACCOUNT-LEVEL lockout. Regenerating the Flex token does NOT clear it
  (new token also got 1025 → lockout is account-scoped, not token-scoped).
- The 1025 state coincided with Arjun being locked out of Client Portal
  login itself (restored via IBKR by phone: 1-877-442-2757). Whether the
  API activity caused the portal lock is unproven but the correlation was
  enough — the risk/benefit is terrible.
- IBKR rate-limit code is 1018 (1/sec, 10/min per token) — never hit it;
  1001/1025 are a different, generation-side protection.

### Context for Next Session
- Daily launchd run (1:05 PM PT) will show `IBKR Gateway not reachable`
  + stale-holdings fallback unless a Gateway/TWS session is logged in on
  this Mac at run time. Arjun intends to log in daily; if reports show
  week-old `as_of` stamps, remind him rather than resurrecting Flex.
- 105/105 tests pass post-removal; imports verified clean.

---
SESSION END: 2026-07-11 09:18 CDT | Agent: Claude Code
---
