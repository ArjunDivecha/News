# Step 2 Data Processing — DORMANT LEGACY

This directory belongs to the **old** News direction: a multi-phase pipeline that would
ingest Goldman Sachs baskets, Bloomberg ETFs, and indices, cluster them, build a FAISS
vector store, and publish a Next.js site with semantic search.

**It was largely never built.** The plan lived in a PRD with milestones running Oct–Dec 2025
and a Jan 2026 beta that did not happen; most components were only ever `[TO BUILD]`. The
parent repo's `CLAUDE.md` records the universe-construction chain as dormant, with the old
Phase 0/2/4 work in `archive/`.

## Read the parent instead

The live News system is a completely different thing — one command
(`python3 report/main.py`) that pulls Schwab and IBKR holdings, downloads ~800 ETF closes
from Yahoo, runs unit-tested analytics, has Fable write a unified market-and-portfolio
report, and emails a PDF daily via launchd. **No Goldman API, no Bloomberg terminal, no web
front end.**

Start at `../CLAUDE.md` and `openwiki/quickstart.md`. Do not follow the milestones,
architecture, or tech stack described in this directory's history — they describe a road not
taken, and treating them as current will send you building against APIs the live system
deliberately doesn't use.

If something here is genuinely worth reviving, raise it as a new decision rather than
resuming the old plan.

## If you must work in here

Ingestion was `gs_basket_data_with_headings.py` (Marquee API, ticker→AssetId resolution, 200
per request), superseding the legacy `gs_basket_data.py`. Goldman client credentials were
**hardcoded in that script** — if you touch it, move them to the environment rather than
leaving them in source. Any path referencing a `/Users/macbook2024/` home directory is stale;
that user does not exist on this machine.
