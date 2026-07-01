# Tier-3 Report Enhancements — Research Synthesis & Plan (v2)

**Author:** Arjun Divecha (with Claude Code)
**Date:** 2026-06-30
**Status:** Design / ideation — builds on `report/PRD_Tier3_Report_Enhancements.md`
**Inputs:** the v1 PRD + a data audit of `universe.xlsx`/holdings + three deep-research streams
(market-day characterization, ETF cross-section quant, tag-based portfolio attribution).

---

## 0. TL;DR — what this changes vs the v1 PRD

The v1 PRD is directionally right (orthogonal tags, "what kind of day," pure-addition, don't-sum-tilts,
gate behind a flag). Six substantive changes come out of the audit + research:

1. **The tag foundation is smaller than feared AND bigger than the PRD said.**
   *Smaller:* the canonical tag vocabulary and a DeepSeek classifier **already exist** in
   `fine tuning/scripts/expert_review_common.py` (`ALLOWED_TIER3_TAGS`, `TAG_ALIASES`, `tags_to_string`,
   `SYSTEM_PROMPT`, `build_user_prompt`, `parse_model_json`) and `build_review_batch.py`
   (`run_deepseek_predictions` → `OpenAI(base_url="https://api.deepseek.com")`). We **reuse** it, not rebuild.
   *Bigger:* the portfolio-side tag views are **blocked today** — only 22 of 67 holdings are in the universe
   and 20 are tagged — so a **dynamic holding tagger** (check→cache→DeepSeek) is a hard prerequisite the v1
   PRD didn't call out.

2. **Three high-value MARKET signals the PRD missed outrank some PRD items** (all confirmed by research
   as top signal-per-effort): **cross-sectional dispersion** (macro day vs stock-picker's day),
   a **volatility/noise gate** (VIX/16 — the single biggest *credibility* win), and an
   **average-pairwise-correlation / absorption-ratio** regime gauge.

3. **Tilts must be demeaned against the universe first** (subtract the equal-weight universe return before
   any tag mean). This removes the market — the biggest shared component — and is what makes correlated
   multi-label tilts safe to read side by side.

4. **The market lead becomes a "day-type" classifier**, not just a tilt table: paired spreads
   (Value−Growth, Cyclical−Defensive, EM−DM, Small−Large, High-beta−Low-vol, Quality−Junk); the 2-3
   largest name the day, and the report says *which axis dominated*.

5. **The portfolio "bridge" (with-vs-against today's tape) is the single highest-value portfolio add** —
   but it sits on top of Workstream 0 (tagged holdings + tagged universe).

6. **Two features are demoted to "honest v2 or skip":** tag **risk contribution** (needs a covariance
   model; a daily no-covariance version violates FAIL-IS-FAIL unless labeled an approximation) and the
   **crowding proxy** (weak without valuation spreads).

---

## 1. Data reality (audit of the real files)

- **Universe tags:** 755/761 funds tagged, avg **4.94 tags/fund**, 152 distinct raw tags. Axes are present
  but **not axis-prefixed** and **inconsistent**: some rows use `Region: Global` / `Strategy: Passive`,
  most use bare `Global`/`Passive`; dupes like `Large-cap`/`Large-Cap`, `Metals`/`Precious Metals`,
  `Commodity`/`Commodities`. The canonical set (`ALLOWED_TIER3_TAGS`, ~55 tags) is the *target*; the current
  `universe.xlsx` is the *messy* set. `tags_to_string()` already maps→canonical and drops non-canonical.
- **Style-spread viability (non-factor funds):** Value 70 / Growth 75 / Quality 43 / Momentum 22 /
  Low Volatility 29 / Dividend 88 / Defensive 90 — all well above the n≥3 floor. Spreads are viable now.
- **Portfolio coverage:** of 67 held names, **22 in universe, 20 tagged**; the other **45 are untagged**
  (single-country ETFs — Australia/Canada/Poland/Mexico/South Africa/Indonesia/Chile… — plus single names
  like Intel and a couple of CUSIPs). These are easy to tag from the resolved name; but until they are,
  every portfolio-side tag view is empty.

---

## 2. Workstream 0 — the tag foundation (prerequisite for everything)

Everything below reads clean, canonical tags on both the universe and the holdings. Build this first,
behind the existing `SETTINGS["enable_tag_views"]` flag.

**0a. Canonical vocabulary + normalize — reuse, don't invent.** Route every tag (universe and holdings)
through `expert_review_common.tags_to_string()` so the whole report speaks one vocabulary
(`ALLOWED_TIER3_TAGS` + `TAG_ALIASES`). This fixes `Metals`→dropped/`Materials`, `Commodities`→`Commodity`,
prefixed/dupe noise. (The expert-review round is separately re-tagging the universe to this set; normalizing
at read-time means we're correct even before that lands.)

**0b. tag→axis map.** A ~55-entry dict mapping each canonical tag to one axis:
`AssetClass | Region | Sector | Style | Strategy | Duration`. Lets the report say *which axis* dominated,
and keeps per-axis decompositions self-contained (never mix axes in one table). Coverage on real data ≈ 96%.

**0c. Dynamic holding tagger — reuse the existing DeepSeek classifier.** New `report/tags.py`, mirroring
the proven `report/names.py` cache pattern:
`resolve_tags(symbols) -> {symbol: [canonical tags]}` with a 3-step fallback per holding —
(1) in universe → its tags; (2) in `report.db` `security_tags` cache → cached; (3) neither → **DeepSeek**:
`OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")` with
`expert_review_common.SYSTEM_PROMPT` + `build_user_prompt` (fed the ticker + the **already-resolved full name**
from `names.py` + a short yfinance description), then `parse_model_json` → `tags_to_string` → cache.
~45 one-time calls now, near-zero ongoing (only new holdings). Cheap, cached, canonical.

**0d. `explode_tags(asset_table)` helper** (PRD §3): one row per (ticker, tag) with returns carried, factor
proxies dropped, missing-data excluded (never zero-filled). Shared by all market-side tag functions.

---

## 3. MARKET-SIDE features (ranked by signal-per-effort × fit for this reader)

> All are pure additions to `compute_market`'s return dict; none touch the P&L/alpha math.
> Every aggregate honors n≥3 (grid n≥5). No cell ever renders "n/a" (project rule): a stale value shows
> as `value*`, a genuinely-undefined cell as `—`.

**M1 — Day-type classifier / paired-spread lead** *(upgrades PRD §4.1 + §4.2).*
Compute, from **universe-demeaned** tag means, the paired spreads Value−Growth, Cyclical−Defensive,
EM−DM, Small−Large, High-beta−Low-vol, Quality−Junk, plus per-axis tilts. The 2-3 largest |spread| **name
the day** and identify the dominant **axis** (style day? regional day? sector day?). Add a rolling z-score
(63d/252d) + percentile so "extreme" is defined, not vibes. Maps 1:1 onto the tags. *(Research: MSCI factor
work; equal-weight within tag; z to make axes comparable.)*

**M2 — Cross-sectional dispersion + percentile** *(NEW — #1 signal/effort).*
`disp = ret.std(axis=1)`; bucket by trailing percentile (low 0-25 / normal / high 75-100). High = a
**stock-picker's day** (selection mattered); low = a **beta/macro day** (holding *which* fund barely
mattered). Compute on **beta-adjusted** returns so a big macro down-day doesn't fake high dispersion.
This is the frame that makes every tilt interpretable.

**M3 — Volatility / noise gate** *(NEW — biggest credibility win).*
Expected daily move ≈ **VIX/16** (Rule of 16). Classify the day sub-1σ / 1-2σ / >2σ, and when a day is
sub-1σ **and** breadth mixed **and** the cross-asset scorecard incoherent **and** dispersion below its
median, the report says plainly: *"today was noise/beta."* Turns the existing "it's OK to say it was noise"
ethos into a quantitative gate. *(Needs a VIX level — see §6.)*

**M4 — Two-dimensional breadth + outlier check** *(upgrades PRD §4.4).*
Report **% of tag-groups positive** (broad vs narrow) as an axis *independent* of dispersion; per top tilt,
share of members moving the tilt's way and the **mean−median gap / trimmed mean** ("Energy +1.5%, 90% of
energy funds up" = macro move vs "+1.5% but half red" = one outlier). Cheap, and it confidence-qualifies
the narrative.

**M5 — Style-factor spread monitor** *(PRD §4.2).*
The paired/excess style spreads as a small table with today's value + **rolling percentile**; persist the
spread series to `report.db` so percentiles have history. A poor-man's factor monitor at zero data cost and
an independent cross-check on the 15-factor complex.

**M6 — EM dispersion decomposition** *(NEW — tailored to this reader).*
Split cross-sectional dispersion into country vs sector vs style vs specific (ANOVA-style between/within on
the Region/Sector/Style tags). MSCI: **country drives >50% of EM dispersion >70% of the time** — "today's EM
move was country-driven, not style-driven" is a differentiated read this reader specifically will value.

**M7 — Region × Sector grid** *(PRD §4.3).*
Mean 1d for each (Region, Sector) cell with **n≥5** (EM country×sector cells are thin — enforce the floor,
show names when a cell is below it). "European Financials +2%, US Tech −1%" — richer than tier2's single slot.

**M8 — Regime gauge: average pairwise correlation (+ optional absorption ratio)** *(NEW — medium effort).*
Trailing avg off-diagonal correlation (calm ≈0.4-0.5, crisis ≈0.9) as a one-number risk-on/off/fragility
read. Optional absorption ratio (Kritzman et al.) computed on **~40-60 tag-level baskets** (not the raw 760,
which is rank-deficient at any sane window). Defer to a later phase; M2+M8-corr give ~80% of the regime read.

**M9 — Crowding proxy** — *low confidence, optional/last.* Return-only proxy (within-tag correlation up +
dispersion compression + momentum) lacks the key valuation-spread input; label "directional, low confidence"
and never as a crash timer. Consider skipping.

---

## 4. PORTFOLIO-SIDE features (ranked) — all require Workstream 0

**P1 — Tag active tilts.** Book tag-weight − universe tag-weight, **per axis** (never one mixed table). Show
top ~3 over/under-weights per axis + a per-axis "tag active share" (½Σ|active weight|). Flag any tilt driven
>X% by a single position (idiosyncratic, not a style view). Framing: active-share / factor-tilt literature.

**P2 — The tag BRIDGE (with-vs-against the tape)** *(highest-value portfolio add).* Overlay the book's tilt
sign on today's dominant market tilts (M1): overweight a tag that **led** = WITH; overweight a tag that
**lagged** = AGAINST. Separates **deliberate contrarian** tilts (consistent with thesis) from **accidental
offside** exposure. Present as a directional read + rough offside-P&L, not an additive sum (correlated axes
double-count). This is the tag-native successor to the current tier2 "bridge."

**P3 — Tag P&L attribution, per axis.** Sum each position's contribution (bps) into its tag bucket **within
one axis** (Region tags partition the book → sums to 100% exactly; repeat for Style/Sector). Three small
tables, each 100% of the day, with "do not add across tables." Optional Brinson allocation-vs-selection for
the 2-3 biggest axes ("made money because EM was strong, or because my EM picks were?").

**P4 — Concentration / effective bets by tag.** 1/HHI on gross tag weights → effective # of tags per axis +
effective # of positions; largest single position and tag as % gross. Cheap, robust, quant-friendly; footnote
that plain HHI ignores correlation (overstates diversification).

**P5 — Tag-exposure vs realized-beta reconciliation.** Side-by-side: characteristic exposure (tag weights)
vs returns-based beta; the **gap is the signal** ("40% Value by tag but trades like Growth" = hedged-out or
mislabeled). Don't collapse to one number.

**P6 — Risk contribution by tag** — *v2 only.* Correct version needs a covariance matrix (Euler/PCR). A
daily no-covariance version understates concentration when tags co-move and would be a fabricated number →
**skip in v1** (or ship a clearly-labeled beta-approximation only). Upgrade path: a lightweight 60-day EWMA
covariance on tag-basket returns.

---

## 5. Explicitly rejected (v1 PRD's list + research additions)

- **Summing tag tilts** or a **joint multi-tag OLS** as the primary attribution (multicollinearity → unstable,
  sign-flipping, double-counting). Demean-vs-universe per-tag is the right default; orthogonalization/ANOVA
  is a deliberate, order-disclosed *periodic* deep-dive, never the daily default.
- **Any tag-predicts-returns / tag-momentum** feature (tags are descriptive, not predictive).
- **Active-vs-passive as a daily signal** (noise; least-reliable tag).
- **Crowding as a crash timer** (compressed spreads predict *reduced premia*, not crashes).

---

## 6. Locked decisions (2026-07-01)

1. **Portfolio benchmark = 60% ACWI + 40% TLT** (a fixed blended benchmark), used for BOTH:
   - **Return/alpha:** benchmark daily return = `0.6·ACWI_ret + 0.4·TLT_ret`; portfolio beta, Expected,
     and Alpha are measured against this blend. **This replaces the single-factor SPY beta/alpha** the
     current report uses (and repeatedly flagged as misleading) — a strict upgrade, independently valuable.
   - **Tag tilts (P1/P2):** book tag-weights minus the benchmark's tag-weights, using **ETF-level tags on
     both sides** (ACWI → its tags at 60% weight, TLT → its tags at 40%; no holdings look-through — a
     bigger data lift, out of scope). ACWI and TLT must themselves be tagged (they are ETFs → resolve via
     Workstream 0).
   - **Data:** add `ACWI`, `TLT` to the daily price fetch (and to `resolve_tags`).
   - *Distinct from* the **market-side** demeaning (M1/M2/M5), which stays the **universe cross-sectional
     mean** — that answers "which tags beat the average fund today," a different reference from the book's
     benchmark. Do not conflate the two.
2. **Portfolio tilts/concentration: gross-led**, net noted.
3. **Noise gate (M3): add `^VIX`** to the daily price fetch (Rule-of-16 expected move).
4. **Absorption ratio (M8): DEFERRED** — dispersion (M2) + average pairwise correlation give the regime read
   cheaply; AR needs basket-level PCA + shrinkage. Revisit as a later phase.
5. **Risk-by-tag (P6): DEFERRED** per FAIL-IS-FAIL — no honest daily version without a covariance model.

**New tickers to add to the fetch:** `ACWI`, `TLT`, `^VIX`. (Consider whether to fold ACWI/TLT into the
universe or keep as report-only benchmark rows.)

---

## 7. Sequencing

- **Phase 0 — tag foundation:** Workstream 0 (normalize via `tags_to_string`, tag→axis map, `resolve_tags`
  DeepSeek dynamic tagger + `security_tags` cache, `explode_tags`). Flag **off**. Prove byte-identical output.
- **Phase 1 — market lead:** M1 day-type + M2 dispersion + M3 noise gate + M4 breadth. Wire the "what kind of
  day" section in `prompt.py`, flag off.
- **Phase 2 — market depth:** M5 spreads (+persist history) + M6 EM decomposition + M7 grid.
- **Phase 3 — portfolio core:** P1 tilts + P2 bridge + P4 concentration.
- **Phase 4 — portfolio depth + regime:** P3 attribution + P5 reconciliation + M8 regime gauge. (P6/M9 deferred.)
- **Gate:** flip `enable_tag_views=True` only after (a) the expert-review round lands the corrected universe
  **and** (b) a spot-check of `resolve_tags` on the real holdings + a few generated reports for tag sanity.

---

## 8. Acceptance criteria (v1 PRD's + additions)

1. All new functions pure (DataFrame/dict in→out), unit-tested in isolation, missing-data excluded (never
   zero-filled).
2. `enable_tag_views=False` → report byte-identical to today (zero P&L-pipeline regression).
3. Tilts are **demeaned vs the universe**; tilts are **never summed**; no joint multi-tag OLS in v1.
4. Every aggregate uses n≥3 (grid n≥5); thin tags dropped, not shown as a 1-fund "mean".
5. **No "n/a" anywhere** (project rule): stale→`value*` + footnote, undefined→`—`.
6. Dynamic holding tagger: reuses the existing DeepSeek classifier + canonical vocab, caches per-ticker in
   `report.db`, outputs only canonical tags; unresolvable IDs keep their ticker.
7. Flag-on report leads its market section with a correct "what-kind-of-day" summary (dominant axis + top
   2-3 tilts) and honestly labels noise/beta days.

---

## Appendix — key sources
- MSCI — dispersion in equity markets & EM country decomposition; "Is the market ripe for stock picking?"
- Kritzman, Li, Page, Rigobon (2010) — *Principal Components as a Measure of Systemic Risk* (absorption ratio).
- Menchero — *Beyond Brinson* (factor↔sector attribution; orthogonalization).
- Cremers & Petajisto — *How Active Is Your Fund Manager?* (active share / tilts).
- Man Group / BlackRock / Morgan Stanley — factor taxonomy, cyclical-vs-defensive, dispersion-vs-correlation.
- Existing repo assets: `fine tuning/scripts/expert_review_common.py` (vocab + prompt + parser),
  `build_review_batch.py` (`run_deepseek_predictions`, DeepSeek client), `report/names.py` (cache pattern to mirror).
</content>
