# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**News Project** is a multi-phase data pipeline and insights engine that transforms raw multi-asset market data into concise, evidence-backed "Asset-Price Insights."

- **Phase 1:** Automated ETL pipeline for Goldman Sachs baskets, Bloomberg ETFs, and indices with data enrichment, clustering, vector store creation, and AI-powered asset taxonomy
- **Phase 2:** Static website with semantic search for stakeholder consumption

See `PRD.md` for complete requirements, timeline (target beta: Jan 5, 2026), and tech stack details.

## Data Flow Architecture

The pipeline follows this sequence:

1. **Data Ingestion (1A-1C)**
   - Goldman Sachs baskets: `gs_basket_data_with_headings.py` → `GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx`
   - Bloomberg ETFs: BQL query for high-liquidity instruments (>$10M ADV)
   - Bloomberg indices: Static copy of master list

2. **Data Enhancement (2A-2F)**
   - Bloomberg DAPI formulas (BDH, BDP) enrich with returns (1D, 5D, MTD, YTD, 1Y, 3Y) and betas
   - Lookup betas from `Betas from Bloomberg.xlsx` file
   - Schema normalization via `standardize_schema.py` (to be built)
   - OpenAI classification assigns `Category_L1/L2/L3` taxonomy

3. **Data Curation (3A-3C)**
   - Hierarchical clustering on 1Y correlation > 0.9; keep centroids to reduce size ≤30%
   - Union master list across three sources, target ≤3000 tickers
   - Vector store: OpenAI `text-embedding-3-small` → FAISS index (`vectors/master.faiss` + metadata JSON)

4. **Insight Generation (4A-5B)**
   - LLM prompt templates in `prompts/market_brief.md` reference RAG-retrieved snippets
   - CLI tool: `python generate_report.py --date YYYY-MM-DD` → `reports/daily_<date>.md + .json`
   - Auto-iterate until insight quality score > threshold or max 3 loops

5. **Web Deployment (6A-6F)**
   - Next.js 14 static export to `web/out/`
   - Vercel Edge Function for semantic search API
   - GitHub Actions CI/CD auto-deploys on push to `main`
   - Data refresh SLA: live by 06:30 ET daily; deployment within 3 min

## Key Files & Scripts

### Ingestion Scripts

- **`gs_basket_data_with_headings.py`** (PRIMARY)
  - Fetches Goldman Sachs GSCB_FLAGSHIP baskets via Marquee API
  - Resolves ticker → AssetId mappings; queries API for full metadata
  - Outputs: ticker, index name, Bloomberg code, financial metrics (empty stubs for now)
  - Expected output: `GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx`
  - Note: Requires valid Goldman Sachs API credentials (client_id, client_secret in env or code)

- **`gs_basket_data.py`** (LEGACY)
  - Earlier version; preserved for reference
  - Use `gs_basket_data_with_headings.py` instead

### Data Files

- **`GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx`** – Goldman basket export (output of gs_basket_data_with_headings.py)
- **`Betas from Bloomberg.xlsx`** – Lookup table for 1Y & 3Y betas to merge into enriched data
- **`PRD.md`** – Full product requirements document with acceptance criteria, timeline, risks

### To Be Built

- `standardize_schema.py` – Normalize column order and data types across baskets, ETFs, indices
- `prompts/market_brief.md` – LLM system + user prompt templates with RAG integration
- `generate_report.py` – CLI report generator with auto-iterate loop
- `vectors/` directory – FAISS index + metadata for semantic search
- `web/` – Next.js 14 frontend with static export

## Common Development Commands

```bash
# Run Goldman Sachs basket ingestion (Phase 1A)
python gs_basket_data_with_headings.py

# (Phase 1B - not yet implemented) Query Bloomberg ETFs via BQL
# (Phase 1C - not yet implemented) Copy indices master

# (Phase 2 - not yet implemented) Run enhancement & curation
python standardize_schema.py          # normalize schema
python enrich_with_betas.py          # merge betas from Bloomberg file
python ai_taxonomy.py                # OpenAI classification
python build_vector_store.py         # create FAISS index

# (Phase 5 - not yet implemented) Generate daily report
python generate_report.py --date 2025-10-15

# (Phase 6 - not yet implemented) Build & deploy static site
cd web && npm run export             # build Next.js static export
npm run deploy                       # (CI/CD handles this via GitHub Actions)
```

## Critical Implementation Notes

### Authentication & Credentials

1. **Goldman Sachs API**
   - Client credentials (client_id, client_secret) currently hardcoded in `gs_basket_data_with_headings.py`
   - **Must move to environment variables or macOS Keychain before production**
   - Requires `read_product_data` scope

2. **OpenAI API**
   - For AI taxonomy (Phase 2F) and insight generation (Phase 5)
   - Store as GitHub Secret for CI/CD
   - Models: GPT-4o (insight generation), text-embedding-3-small (RAG embeddings)

3. **Bloomberg Data API (DAPI)**
   - Phase 2 enhancement formulas (BDH, BDP) require Bloomberg Terminal on Mac mini
   - Cache responses to avoid rate limits (batch ≤300 tickers per call)

### Data Quality & Validation

- Goldman API batch size: max 200 per request (already implemented in `chunks()` helper)
- Missing descriptions and unresolved tickers logged as warnings
- Phase 2 must validate completeness across all three data sources (baskets, ETFs, indices)
- Betas merge via ticker lookup from `Betas from Bloomberg.xlsx`; handle mismatches gracefully

### Performance & Infrastructure

- **Clustering target:** Reduce each source to ≤30% of original size via 1Y correlation threshold > 0.9
- **Vector store:** FAISS index must support <50 ms query latency on Mac M2
- **Web search API:** <500 ms p95 latency for semantic search via Vercel Edge Functions
- **Deployment:** GitHub Actions cron @ 04:00 ET for data refresh; web live by 06:30 ET

### Web Deployment & CI/CD

- Static site: Next.js 14 with `npm run export` to `web/out/`
- Hosting: GitHub Pages (free) or Vercel (for edge functions)
- GitHub Actions workflow (`deploy.yml`): auto-deploy any push to `main` in <3 min
- Lighthouse performance budget: mobile score ≥90, TTFB <250 ms US East, JS ≤150 kB
- Secrets: Bloomberg creds in macOS Keychain, OpenAI & Vercel tokens as GitHub Secrets

### Compliance & Security

- Strip raw price columns before publishing to web (only derived stats & narratives)
- Retain proprietary Bloomberg time-series locally; do not expose via web
- Optional: Toggle `PRIVATE_BETA=true` env var to enable Cloudflare Access login gate
- Docker images for Python env (`ml-pipeline`) and Node env (`news-web`) for reproducibility

## Project Structure

```
/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/News/
├── PRD.md                                    # Product requirements & acceptance criteria
├── gs_basket_data_with_headings.py          # Goldman Sachs basket ingestion (Phase 1A)
├── gs_basket_data.py                        # Legacy version
├── GSCB_FLAGSHIP_coverage_with_desc_ALL.xlsx # Output of basket ingestion
├── Betas from Bloomberg.xlsx                 # Beta lookup table (input to Phase 2D)
├── CLAUDE.md                                 # This file
│
├── [TO BUILD] src/
│   ├── standardize_schema.py                # Phase 2E: normalize schema
│   ├── ai_taxonomy.py                       # Phase 2F: OpenAI classification
│   ├── build_vector_store.py                # Phase 3: create FAISS index
│   ├── generate_report.py                   # Phase 5: daily insight generation
│   └── utils.py                             # shared helpers (API batching, caching, etc)
│
├── [TO BUILD] prompts/
│   └── market_brief.md                      # Phase 4: LLM prompt templates with RAG
│
├── [TO BUILD] vectors/
│   ├── master.faiss                         # FAISS index
│   └── metadata.json                        # ticker→embedding metadata
│
├── [TO BUILD] data/
│   ├── baskets_goldman.xlsx                 # Phase 1A output
│   ├── etf_liquid.xlsx                      # Phase 1B output
│   ├── bbg_indices_master.xlsx              # Phase 1C (static)
│   └── enriched_master.xlsx                 # Phase 2E output (normalized schema)
│
├── [TO BUILD] reports/
│   ├── daily_2025-10-15.md                  # Phase 5 output (markdown)
│   └── daily_2025-10-15.json                # Phase 5 output (JSON with metadata)
│
└── [TO BUILD] web/
    ├── package.json                         # Next.js 14 config
    ├── next.config.js                       # Static export settings
    ├── tailwind.config.js                   # Tailwind CSS config
    ├── pages/
    │   ├── index.tsx                        # Home (latest report)
    │   ├── archive.tsx                      # Calendar archive
    │   ├── [slug].tsx                       # Insight detail view
    │   └── api/search.ts                    # Semantic search endpoint (Vercel Edge)
    ├── components/
    │   ├── ReportView.tsx
    │   ├── SearchBar.tsx
    │   └── ThemeToggle.tsx                  # Dark/light mode
    ├── out/                                 # Static export output
    └── .github/workflows/
        └── deploy.yml                       # GitHub Actions CI/CD
```

## Milestone Checklist (PRD Reference)

- **M1 (Oct 31):** Data scripts 1A-1C live
- **M2 (Nov 7):** Enhancement formulas verified
- **M3 (Nov 14):** AI taxonomy v1 complete
- **M4 (Nov 21):** Curation + vector store
- **M5 (Dec 5):** Insight engine MVP
- **M6 (Dec 12):** Static site v0.1
- **M7 (Dec 19):** Search API + CI/CD live
- **M8 (Jan 5, 2026):** Beta launch (private)

## Open Questions from PRD

1. Confirm hosting choice (GitHub Pages vs Vercel) for search edge function requirement
2. External stakeholder SSO requirements, or invite-only Cloudflare Access?
3. Should markdown reports include YAML front-matter for meta-parsing?
4. Budget for custom domain & SSL?

## Reference

- **Tech Stack:** Python 3.11, pandas, openpyxl, scikit-learn, FAISS, OpenAI API, Next.js 14, Tailwind CSS, GitHub Actions
- **Key Dependencies:** gs_quant (Goldman Sachs), pandas, openai, faiss-cpu, next
- **Hosting:** GitHub Pages (free) or Vercel (with edge functions for search)
- **Data SLA:** Refresh by 06:30 ET daily; web deployment within 3 min via GitHub Actions
