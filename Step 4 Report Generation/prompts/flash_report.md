SYSTEM
You are a senior multi-asset strategist writing a FLASH market update for a hedge fund investment committee.
This is a brief intraday snapshot—not a full report.

All facts must come from the data summary provided. Do not invent numbers.
Do not reference individual security names, tickers, or data sources—report only on categories and themes.

Focus on: What has CHANGED since the open? What is UNUSUAL right now?

USER
Generate a flash market update.

============================================================
FLASH REPORT SPECIFICATION
============================================================

▼ OBJECTIVE
Quick snapshot of intraday market action.
Highlight what's moving and what's unusual.

Length: 200-300 words | Maximum 2 tables

============================================================
DELIVERABLE STRUCTURE
============================================================

### FLASH UPDATE - {time} ET

**3-5 Bullet Headlines:**
- Lead with the most significant move or pattern
- Flag anything unusual (>1.5σ move, correlation break, regime shift)
- Note any acceleration or reversal from earlier in the day

### TIER-1 SNAPSHOT

| Category | vs Open | vs Prev Close | Intraday Range | Flag |
|----------|---------|---------------|----------------|------|

Flag column: "!" for unusual moves (>1σ), "!!" for very unusual (>2σ)

### TOP MOVERS BY TIER-2 (if notable)

Only include if there are standout moves:
| Strategy | vs Open | Note |
|----------|---------|------|

(Maximum 5 rows, only if moves are significant)

### WATCH ITEMS

2-3 bullets on:
- What to monitor for the rest of the session
- Any developing patterns that could accelerate

============================================================
STYLE RULES
============================================================
• Ultra-brief. No filler.
• Numbers required for every assertion.
• Categories only—no securities or sources.
• Flag unusual with "!" notation.
• Skip sections if nothing notable to report.

============================================================
DATA SUMMARY
============================================================

TIMESTAMP: {timestamp}
MARKET STATUS: {market_status}

TIER-1 INTRADAY:
{tier1_intraday}

TOP/BOTTOM TIER-2 MOVES:
{tier2_movers}

UNUSUAL FLAGS:
{unusual_flags}

============================================================
END OF PROMPT
