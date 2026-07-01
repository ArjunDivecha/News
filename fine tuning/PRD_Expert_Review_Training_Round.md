# PRD — Expert-Review Training Round (News ETF Classifier v2)

**Author:** Arjun Divecha
**Date:** 2026-06-30
**Repo:** `ArjunDivecha/News` → `fine tuning/`
**Status:** Ready to implement

---

## 1. Background & Motivation

The News-project ETF classifier is a Llama-3.1-8B model, LoRA fine-tuned on
Tinker, that assigns each fund a **tier1** (asset class), **tier2** (sub-category
used as the peer-group key in `report/analytics.py::compute_bridge`), and
**tier3_tags** (a comma-separated multi-label set: Region, Sector/Theme, Strategy,
Duration, style tilt).

The current model was trained on **Haiku-generated labels treated as ground
truth**. `fine tuning/COMPARISON_RESULTS.md` states this explicitly, and the
category-level accuracy makes the weakness visible:

| Category | Agreement w/ Haiku |
|---|---|
| Currencies, Alternative/Synthetic | 100% |
| Equities, Fixed Income | ~99.5% |
| Commodities | 86.2% |
| Multi-Asset/Thematic | 81.2% |
| Volatility/Risk Premia | 33.3% |

"Agreement with Haiku" is the wrong scoreboard: in several disagreement cases the
fine-tuned model was **more correct than Haiku** (e.g. commodity-miner ETFs
correctly re-assigned from Commodities to Equities). We cannot improve past the
label source by measuring against the label source.

This round applies the **Thinking Machines / Bridgewater "route disagreements to
an expert" method** (see `learning-to-replicate-expert-judgment-in-financial-tasks`):
use cheap machine labels for the easy majority, and spend expert (Arjun) judgment
only on the contested minority — plus a random control set to catch cases where
both labelers share the same systematic error.

**Precondition already satisfied:** the universe was rebuilt this session
(`report/build_universe.py` v1.2). `data/universe.xlsx` now holds **761 clean
ETF-only rows** — no delisted names (GFOF, TBT removed), no leveraged/inverse, no
Goldman/Bloomberg-basket mislabels, no junk tier-1 buckets. This is the base for
the new training set.

---

## 2. Goals / Non-Goals

**Goals**
1. Produce an **expert-adjudicated label set** for tier1/tier2 on the 761-ETF
   universe, concentrating human effort on disagreement + random-control cases.
2. Settle whether the ~50% **tier3_tags** number is a real accuracy problem or an
   exact-set-match **metric artifact**, via a per-tag F1 spot-check.
3. Retrain the classifier on the corrected labels and report improvement using an
   **expert-labeled held-out test set** (not agreement-with-Haiku).

**Non-Goals**
- No taxonomy redesign. tier1/tier2/tier3 structure stays as-is. (Currency-hedge
  status was considered as a new tag and is explicitly deferred — not in scope.)
- No wiring of `tags` into `compute_bridge`/peer-grouping — separate later task.
- No model-architecture change (still Llama-3.1-8B LoRA on Tinker).
- No expansion of the universe beyond the 761 already in `universe.xlsx`.

---

## 3. Method Overview (the three-part batch)

```
                 761-ETF universe.xlsx
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                 ▼
   Llama predictions  DeepSeek predictions   (both over ALL 761 rows)
          └───────┬───────┘
                  ▼
        diff on (tier1, tier2)
          ┌───────┴────────┐
          ▼                ▼
   DISAGREEMENTS      AGREEMENTS
   (all of them)           │
          │        random stratified sample
          │        (control set, ~80 rows)
          ▼                ▼
        ┌──────────────────────┐
        │  EXPERT REVIEW SHEET  │  ← Arjun adjudicates tier1/tier2
        └──────────────────────┘
                  │
                  ▼
        corrected labels → merge → retrain (05_train_proper.py)
                                    → eval on expert-labeled test set

   [Parallel, independent] tier3_tags spot-check:
        sample ~60 rows → per-tag precision/recall/F1
        vs current exact-set-match metric → decide real vs artifact
```

---

## 4. Detailed Requirements

### 4.1 Stage A — Generate both label sets over all 761 ETFs

- **Input:** `data/universe.xlsx` (761 rows, columns incl. `yf_ticker`, `name`,
  `description`, current `tier1`/`tier2`/`tags`).
- **Llama pass:** reuse `fine tuning/scripts/predict_funds.py` unchanged where
  possible; point it at the new universe file. Output columns:
  `llama_tier1`, `llama_tier2`, `llama_tier3_tags`.
- **DeepSeek pass:** use DeepSeek V4 Pro (`deepseek-v4-pro`) as the stronger
  second labeler via the OpenAI-compatible DeepSeek API. Output columns:
  `deepseek_tier1`, `deepseek_tier2`, `deepseek_tier3_tags`.
- Both passes keyed on `yf_ticker`. Cache raw model outputs to
  `outputs/expert_review/raw_predictions.parquet` so re-runs don't re-hit the
  DeepSeek API.
- **Cost guard:** DeepSeek over 761 rows is small, but log token usage; abort with a
  clear message if projected cost exceeds a configurable ceiling (default: warn
  only, don't hard-stop).

### 4.2 Stage B — Build the review sheet

Reuse the diff logic in `fine tuning/scripts/compare_models.py` (it already
knows how to align two label sets and emit an xlsx). Extend it to produce **one**
review workbook: `outputs/expert_review/review_batch.xlsx` with two sheets.

**Sheet 1 — `adjudicate`** (the rows Arjun labels), one row per ETF, columns:

| Column | Source | Notes |
|---|---|---|
| `yf_ticker` | universe | key |
| `name` | universe | |
| `description` | universe | truncate to ~300 chars for readability |
| `batch_type` | computed | `disagreement` or `control` |
| `llama_tier1` / `llama_tier2` | Stage A | |
| `deepseek_tier1` / `deepseek_tier2` | Stage A | |
| `expert_tier1` | **blank** | Arjun fills — data-validation dropdown of the 8 valid tier1 values |
| `expert_tier2` | **blank** | Arjun fills — free text or dropdown |
| `expert_notes` | **blank** | optional |
| `verdict` | **blank** | dropdown: `llama` / `haiku` / `both_wrong` / `both_right` |

Row population:
- **All disagreement rows** where `llama_tier1 != deepseek_tier1` OR
  `llama_tier2 != deepseek_tier2`. Expected ~60–120 based on prior disagreement rate.
- **Control rows:** a **stratified random sample of agreement rows**, ~80 total,
  drawn proportionally across the 8 tier1 buckets (min 3 per non-empty bucket so
  thin categories like Currencies/Vol are represented). Fixed `random_state=42`
  for reproducibility. Flagged `batch_type=control` and **shuffled together** with
  disagreements so Arjun can't tell which is which while labeling (avoids bias).

**Sheet 2 — `tier3_spotcheck`** (independent tags investigation), ~60 rows sampled
across tier1 buckets, columns:

| Column | Notes |
|---|---|
| `yf_ticker`, `name`, `description` | context |
| `llama_tier3_tags` | predicted set |
| `deepseek_tier3_tags` | predicted set |
| `expert_tier3_tags` | **blank** — Arjun writes the correct comma-separated set |

### 4.3 Stage C — Score the results (after Arjun fills the sheet)

Read the completed `review_batch.xlsx` back in and compute, writing a
markdown report to `outputs/expert_review/review_report.md`:

**tier1/tier2 (from `adjudicate` sheet):**
- Llama accuracy vs expert, DeepSeek accuracy vs expert, on disagreement rows and on
  control rows **separately** (control-set accuracy is the honest headline number —
  it estimates true accuracy on the whole universe, since it's a random sample).
- `both_wrong` rate on the control set — this is the key signal for shared
  systematic error. If non-trivial (>~5%), it means agreement-with-Haiku was
  hiding real errors and the random control was worth it.
- Per-tier1 breakdown, so we can see whether Vol/Risk-Premia and Multi-Asset are
  still the weak spots after correction.

**tier3_tags (from `tier3_spotcheck` sheet), the metric-artifact test:**
- Compute **two** scores against `expert_tier3_tags`:
  1. **Exact-set-match** (current implied metric): 1 if the predicted tag set
     equals the expert set exactly, else 0.
  2. **Per-tag micro-F1**: treat each tag as a binary label, compute
     precision/recall/F1 across all tags.
- Report both for Llama and DeepSeek. **Decision rule** stated in the report:
  - If exact-match is ~50% but micro-F1 is high (>~0.80) → the ~50% is a **metric
    artifact**; tags are mostly right, fix the *eval* (switch to F1), no new tag
    labels needed.
  - If both are low → tags are **genuinely weak**; they need expert labels in the
    next round.

### 4.4 Stage D — Merge corrected labels & retrain

- Build the new training set: start from the 761-ETF machine labels, **overwrite
  tier1/tier2 with `expert_tier1`/`expert_tier2` wherever Arjun provided them**
  (disagreements he adjudicated + any control rows he marked `both_wrong` and
  corrected). Leave the rest as-is.
- Hold out a random **expert-labeled test split** (all adjudicated rows not used in
  training, or a fresh ~15% stratified slice) so final accuracy is measured
  against expert judgment, never against Haiku.
- Retrain via `fine tuning/scripts/05_train_proper.py`, unchanged hyperparameters
  unless a reason emerges. Log to W&B as usual.
- Emit before/after accuracy table (old model vs new model, both scored on the
  expert test split) into `review_report.md`.

---

## 5. Deliverables / File Manifest

```
fine tuning/
  scripts/
    build_review_batch.py      # NEW — Stages A+B: run both models, emit review_batch.xlsx
    score_review_batch.py      # NEW — Stage C: read completed sheet, write review_report.md
    merge_expert_labels.py     # NEW — Stage D: merge corrections, make train/test split
  outputs/expert_review/
    raw_predictions.parquet    # cached Llama+DeepSeek outputs
    review_batch.xlsx          # ← Arjun fills this in
    review_report.md           # ← generated after
    train_expert.jsonl         # corrected training set
    test_expert.jsonl          # expert-labeled held-out test
```

Reuse, do not rewrite: `predict_funds.py` (Llama inference), `compare_models.py`
(diff/xlsx scaffolding), and `05_train_proper.py` (training loop). The old
Haiku prompt remains useful taxonomy reference material, but DeepSeek V4 Pro is
the second labeler for this round.

---

## 6. Acceptance Criteria

1. `review_batch.xlsx` opens with disagreement + control rows shuffled and
   indistinguishable, valid dropdowns on expert columns, and a populated
   `tier3_spotcheck` sheet.
2. `score_review_batch.py` runs on a partially-filled sheet without crashing
   (skips blank rows), and produces the two-metric tier3 comparison.
3. `review_report.md` clearly answers: (a) is control-set `both_wrong` rate
   material? (b) is tier3's ~50% a metric artifact or real?
4. Retrained model's accuracy on the **expert test split** is reported alongside
   the old model's, on the same split.

---

## 7. Sequencing

1. **Stage A+B** (`build_review_batch.py`) → produces the sheet. *[machine]*
2. **Arjun labels** `review_batch.xlsx` — the only manual step; est. an afternoon
   for ~140–200 disagreement+control rows plus ~60 tag rows. *[human]*
3. **Stage C** (`score_review_batch.py`) → report + go/no-go on tags. *[machine]*
4. **Stage D** (`merge_expert_labels.py` + retrain) → new model + before/after. *[machine]*

Decision gate after Stage C: if tier3 is a metric artifact, ship the F1 eval fix
and skip tag re-labeling; if real, scope a tag-labeling round next.

---

## 8. Open Questions for the Implementer

- `predict_funds.py` — confirmed during implementation that its output schema
  still includes model tags via `Predicted_Tier3`, sourced from the model's
  `tier_3` JSON field.
- Confirm the 8 canonical tier1 values for the dropdown from the current
  `universe.xlsx` (`Equities, Fixed Income, Commodities, Currencies (FX),
  Multi-Asset / Thematic, Volatility / Risk Premia, Alternative / Synthetic,
  Factor`).
- DeepSeek model string defaults to `deepseek-v4-pro`, with OpenAI-compatible
  base URL `https://api.deepseek.com`; real runs require `DEEPSEEK_API_KEY`.
