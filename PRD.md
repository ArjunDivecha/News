# News Project – Product Requirements Document (PRD)

**Version:** 0.3 (Draft)  
**Author:** ChatGPT (o3) for Arjun Divecha  
**Last updated:** 13 Oct 2025

---

## 0. Executive Summary

The News Project transforms raw multi‑asset market data into concise, evidence‑backed "Asset‑Price Insights."
Phase 1 builds an automated data pipeline and insight engine.
Phase 2 publishes the daily output to a static website with semantic search so stakeholders can consume insights without installing any software.

---

## 1. Goals & Non‑Goals

| | **In scope (Phase 1 & 2)** | **Out of scope (Phase 1 & 2)** |
| ----------------- | -------------------------------------------------------------------------- | -------------------------------------------- |
| **Data coverage** | Goldman thematic baskets, Bloomberg ETFs (≥ $10 mm ADV), Bloomberg indices | Single‑stock fundamentals, options, alt‑data |
| **Analytics** | Multi‑period returns, betas, clustering, AI taxonomy, z‑score moves | Price forecasting, NLP news sentiment |
| **AI / LLM** | OpenAI classification, vector RAG, GPT‑4o insight generation | Automated trade recommendations |
| **Distribution** | Excel files, JSON APIs, **static web site** | Mobile app, email alerts |

---

## 2. Functional Requirements

### 2.1 Data Collection

| ID | Task | Detail | Acceptance Criteria |
| ------ | -------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| **1A** | Ingest Goldman baskets | Run `gs_basket_data_with_headings.py` (local path) nightly. | `data/baskets_goldman.xlsx` contains ≥ N baskets with `Ticker, Name, Basket_Type`. |
| **1B** | Ingest high‑liquidity ETFs | BQL: `FUND_FLOWS WHERE AVG_TRADING_VOL_USD_LAST_20D > 10e6`. | `data/etf_liquid.xlsx` has ≥ 1 000 rows with `Ticker, Description, AvgVolUSD`. |
| **1C** | Load Bloomberg indices | Copy static source `data/bbg_indices_master.xlsx`. | File checksum matches source. |

Execution: GitHub Actions cron @ 04:00 ET on Bloomberg‑enabled Mac mini.

---

### 2.2 Data Enhancement (Excel DAPI)

| ID | Item | Bloomberg Formula | Output Columns |
| ------ | ---------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| **2A** | Returns for baskets | `BDH(<ticker>, "TOT_RETURN_INDEX_NET", dates)` | `Ret_1D, Ret_5D, Ret_MTD, Ret_YTD, Ret_1Y, Ret_3Y` |
| **2B** | ETF descriptions & returns | `BDP(<ticker>, "CIE_DES")` + `BDH` | `Description`, returns |
| **2C** | Index descriptions & returns | `BDP(<ticker>, "DS366")` + `BDH` | `Description`, returns |
| **2D** | Merge betas | Lookup from `Betas from Bloomberg.xlsx`. | `Beta_1Y, Beta_3Y` |
| **2E** | Normalize schema | Python script `standardize_schema.py`. | All three sheets share identical column order & dtypes. |
| **2F** | AI asset taxonomy | OpenAI call assigns `Category_L1/L2/L3`. | ≥ 95 % rows categorized; fallback = "Other". |

---

### 2.3 Data Curation

| ID | Step | Method | Success Metric |
| ------ | --------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------- |
| **3A** | De‑duplicate & shrink | Hierarchical clustering on 1‑Y corr > 0.9, keep centroid. | Each source list ≤ 30 % of original size. |
| **3B** | Master list union | `pandas.concat`, drop dups on `Ticker`. | Master ≤ 3 000 tickers. |
| **3C** | Vector store | `text-embedding-3-small`, store in FAISS index `vectors/master.faiss` with metadata JSON. | Query latency < 50 ms on Mac M2. |

---

### 2.4 LLM Prompt Library

| ID | Component | Requirement |
| ------ | ----------------- | ------------------------------------------------------------------------------------- |
| **4A** | Prompt templates | `prompts/market_brief.md` contains system + user prompts that reference RAG snippets. |
| **4B** | Evaluation rubric | YAML section defines scoring: `sigma_strength`, `precedent_similarity`, `confidence`. |

---

### 2.5 Report Generation (CLI)

| ID | Task | Detail |
| ------ | --------------------- | -------------------------------------------------------------------------------------------------------- |
| **5A** | Generate daily report | `python generate_report.py --date YYYY-MM-DD` → `reports/daily_<date>.md` + `reports/daily_<date>.json`. |
| **5B** | Auto‑iterate | Loop prompt until avg insight score > threshold or max 3 loops. |

---

### 2.6 Web Deployment & Distribution

| ID | Requirement | Acceptance Criteria |
| ------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **6A** | Static‑site generator | **Next.js 14** `npm run export` builds into `web/out/`. |
| **6B** | Site features | Pages: Home (latest report), Archive (calendar), Insight detail, Search. Responsive Tailwind UI, dark/light mode. |
| **6C** | Semantic search API | Vercel Edge Function loads FAISS index and returns top‑k JSON. Latency < 500 ms p95. |
| **6D** | CI/CD pipeline | GitHub Actions `deploy.yml`: build → upload to GitHub Pages (default) **or** Vercel via token. Any push to `main` auto‑deploys in < 3 min. |
| **6E** | Privacy / auth (opt) | Toggle `PRIVATE_BETA=true` env var to enable Cloudflare Access login gate. |
| **6F** | Performance budget | Lighthouse mobile score ≥ 90; TTFB < 250 ms (US East); JS ≤ 150 kB. |

---

## 3. Non‑Functional Requirements

* **Data refresh SLA:** New web content live by 06:30 ET daily.
* **Observability:** CloudWatch (or Vercel Analytics) alerts on build failure & search latency.
* **Reproducibility:** Docker images for Python env (`ml-pipeline`) and Node env (`news-web`).
* **Security:** Bloomberg creds in macOS Keychain; OpenAI & Vercel tokens as GitHub Secrets; no secrets in static export.
* **Compliance:** Retain raw Bloomberg data locally; do not expose proprietary time‑series via web.

---

## 4. Tech Stack

| Layer | Technology | Notes |
| ------------ | --------------------------------------------------------- | ------------------------------- |
| ETL | Python 3.11, pandas, openpyxl | Excel manipulation |
| Data science | scikit‑learn, faiss‑cpu | Clustering + vector store |
| AI APIs | OpenAI GPT‑4o, text‑embedding‑3‑small | Classification & RAG embeddings |
| Front‑end | Next.js 14 (static export), React 18, Tailwind CSS, MDX | No runtime server required |
| CI/CD | GitHub Actions, Vercel CLI or gh‑pages | One‑click deploy |
| Hosting | GitHub Pages (free) or Vercel (edge functions for search) | Custom domain `news-project.ai` |

---

## 5. Milestones & Timeline

| # | Milestone | Owner | Target Date |
| ------ | ----------------------------- | ---------- | ----------- |
| **M1** | Data scripts 1A‑1C live | Data Eng | 31 Oct 2025 |
| **M2** | Enhancement formulas verified | Analyst | 07 Nov 2025 |
| **M3** | AI taxonomy v1 complete | ML Eng | 14 Nov 2025 |
| **M4** | Curation + vector store | Data Eng | 21 Nov 2025 |
| **M5** | Insight engine MVP | LLM Eng | 05 Dec 2025 |
| **M6** | Static site v0.1 (HTML only) | Front‑End | 12 Dec 2025 |
| **M7** | Search API + CI/CD live | Full‑Stack | 19 Dec 2025 |
| **M8** | Beta launch (private) | PM | 05 Jan 2026 |

---

## 6. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
| -------------------------------- | ----------------- | ---------- | ------------------------------------------------------------------------------ |
| Bloomberg rate‑limits BDH | Data gap | Medium | Batch <= 300 tickers per call, local cache, exponential back‑off. |
| Exposing proprietary data on web | Compliance breach | Low‑Medium | Strip raw price columns before publish; share only derived stats & narratives. |
| Vercel edge cold‑start | Slow search | Medium | Keep warm via scheduled ping; use `--edge-config` pre‑fetch. |
| AI taxonomy mis‑labels | Insight quality | Medium | Human spot checks; fine‑tune classification with feedback loop. |

---

## 7. Open Questions

1. Confirm hosting choice (GitHub Pages vs Vercel) considering search edge function requirement.
2. Will external stakeholders require SSO or can we rely on invite‑only Cloudflare Access?
3. Should markdown reports include YAML front‑matter for meta‑parsing?
4. Budget for custom domain & SSL?

---

*End of draft PRD v0.3*

