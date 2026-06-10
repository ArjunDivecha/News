# News - Unified Daily Market & Portfolio Report

One command pulls live Schwab + IBKR holdings, fetches ~800 ETF prices from
Yahoo Finance (no Bloomberg terminal), runs unit-tested analytics, and has
Claude Opus write a single unified market + portfolio report as a PDF:

```bash
python3 report/main.py
```

Full documentation: [`report/README.md`](report/README.md)

```bash
python3 report/main.py --no-llm           # data + analytics only (free, fast)
python3 report/main.py --non-interactive  # cron mode (stale fallback, no prompts)
python3 -m pytest tests/ -v               # 25 tests on the financial math
```

## Repository layout

```
News/
├── report/                          # THE daily report system (start here)
│   ├── main.py                      #   one-command pipeline
│   ├── prompts/system.md            #   LLM strategist prompt
│   └── README.md                    #   full docs
├── tests/                           # pytest suite for the financial math
├── data/                            # universe.xlsx, report.db, holdings.xlsx
├── outputs/unified/                 # generated reports (PDF/MD + data packages)
│
├── Step 1 Data Collection/          # Universe construction (run rarely)
├── Step 2 Data Processing - Final1000/  #   classification & selection
├── Step 3 Data Analysis/            #   analytics on the universe
├── fine tuning/                     #   ML classifier training
│
├── archive/                         # Legacy reporting chain (replaced by report/)
├── AGENTS.md                        # AI agent instructions
└── README.md                        # this file
```

## The two pipelines

### 1. Daily report (`report/`) - run every day

See `report/README.md`. Replaces the old Phase 0 -> Step 4 -> Phase 2 chain
(now in `archive/`). Universe: 763 unique ETFs in `data/universe.xlsx`,
priced via Yahoo Finance. History: single SQLite db at `data/report.db`.

### 2. Universe construction (Steps 1-3 + fine tuning) - run rarely

Builds the Final 1000 Asset Master List from Bloomberg indices, ETFs, and
Goldman baskets via LLM classification. Only needed to rebuild/refresh the
universe; after changing it, regenerate the report universe with:

```bash
python3 report/build_universe.py
```

| Stage | Purpose | Output |
|-------|---------|--------|
| Step 1 Data Collection | Raw data acquisition | Filtered datasets from 3 sources |
| Step 2 Data Processing | Classification & selection | Final 1000 Asset Master List |
| Step 3 Data Analysis | Performance analytics | Factor profiles & deduplication |
| Fine Tuning | ML model training | Fine-tuned Llama classifier |

## Requirements

- **Python** 3.14 (`yfinance`, `pandas`, `anthropic>=0.109`, `schwabdev`,
  `python-dotenv`, `markdown`, `pytest`)
- `.venv-ibkr312/` - Python 3.12 venv with `ib_insync` (IBKR API requirement)
- **PrinceXML** (`brew install prince`) for PDF rendering
- `.env` at repo root: `ANTHROPIC_API_KEY`, `SCHWAB_APP_KEY`, `SCHWAB_APP_SECRET`
- TWS / IB Gateway logged in (auto-launched if not running)

## Documentation

| File | Purpose |
|------|---------|
| `report/README.md` | Daily report system - full guide |
| `AGENTS.md` | AI coding agent instructions |
| `archive/README.md` | What the legacy code was and what replaced it |
| `Step 2 .../README.md` | Classification workflow |
| `fine tuning/README.md` | ML training guide |

---

**Last Updated**: 2026-06-10
**Version**: 3.0.0 (unified report rearchitecture)
