# PRD — Tier-3 Tag Enhancements for the Daily Market Report

**Author:** Arjun Divecha
**Date:** 2026-06-30
**Repo:** `ArjunDivecha/News` → `report/`
**Status:** Design — ready to implement (see dependency note in §1)

---

## 1. Motivation & the core insight

The report currently narrates the market almost entirely through **tier1**
(asset class) and **tier2** (the single peer-group key). But every fund also
carries **tier3 `tags`** — a comma-separated, *multi-label* set spanning Region,
Sector/Theme, Style (Value/Growth/Momentum/Quality/Low-Vol), Strategy
(Active/Passive/Options-Based), and Duration.

The key property tier1/tier2 can't express: **tags are orthogonal.** A fund is
*simultaneously* Region=Europe, Sector=Financials, Style=Value, Strategy=Passive.
tier2 forces one bucket; tags let the same fund contribute to several
cross-cutting views at once. That orthogonality is exactly what lets the report
answer **"what *kind* of day was it"** — a style day? a regional day? a sector
day? — instead of only "which tier2 bucket moved."

**Implementation is low-risk.** `tags` is already joined onto `asset_table` in
`compute_market` (the `meta` join). Every function below is a **pure addition**
that reads `asset_table` and returns new dicts/DataFrames — it touches none of the
existing return/contribution/alpha math and cannot regress the P&L pipeline.

**Dependency (sequencing, not blocking):** these views are only as trustworthy as
the tags. Today's tags are Haiku-bootstrapped and being corrected by the
expert-review training round (`PRD_Expert_Review_Training_Round.md`). Build and
wire these functions now, but treat their narrative output as provisional until
the corrected universe lands — a mis-tagged options-overlay fund wrongly tagged
`Value` would poison the Value-vs-Growth spread. **Recommend gating the tag
narrative behind a config flag** (`SETTINGS["enable_tag_views"]`, default off)
until the label round completes, so the plumbing ships without emitting
confident-but-wrong prose.

---

## 2. Goals / Non-Goals

**Goals**
1. Add tag-based analytics to `analytics.py` as pure functions, consumed by
   `prompt.py`, that surface the day's dominant cross-cutting tilts.
2. Give the narrative a "what kind of day was it" lead built from tags.
3. Keep every existing metric, signature, and convention untouched.

**Non-Goals**
- No predictive/forecasting use of tags. Tags describe what a fund *is*; these are
  **decomposition/attribution** tools for explaining *today*, never a forecast.
  (Prediction lives in T2/ASADO, not here.)
- No change to tier1/tier2 aggregation or the peer-group bridge.
- No new data feed — everything derives from `asset_table` already in memory.
- No orthogonalized multi-tag regression in v1 (see §4.1 caveat).

---

## 3. Tag parsing (shared helper, build once)

Add one helper that all tag functions use:

```
explode_tags(asset_table) -> long DataFrame
    one row per (yf_ticker, tag), with return_1d/1w/1m/ytd carried along.
    - split `tags` on comma, strip whitespace, drop blanks
    - drop factor-proxy rows (is_factor==1) so benchmark ETFs don't
      double-count inside their own theme
    - a fund with 5 tags produces 5 rows; a fund with no tags produces 0
```

This mirrors the module's existing discipline: assets without data are simply
absent, never zero-filled. All aggregates below are means over funds **with**
`return_1d`, exactly like `tier1_summary`.

---

## 4. Features (ranked by narrative value)

### 4.1 Tag-tilt decomposition — the headline feature

**Function:** `compute_tag_tilts(asset_table) -> pd.DataFrame`

For each tag: mean `return_1d` of all funds carrying it, **minus the universe
mean** (the tilt). Also carry `n` (funds in the tag) and the 1w tilt for context.
Return sorted by `abs(tilt_1d)` descending, filtered to tags with `n >= 3` (same
peer-floor the bridge already uses).

Output feeds a narrative lead like:
> "A value-over-growth day: Value +1.2% vs universe, Growth −0.8%; Energy the
> strongest sector tilt (+1.5%); EM lagged (−0.9%)."

Group tags by **axis** (Region / Sector / Style / Strategy / Duration) in the
output so `prompt.py` can say which *axis* dominated, not just which tag.

**Critical caveat for the implementer — do NOT sum tilts.** Tags are multi-label
and correlated (Value funds skew US, etc.), so tilts are **not additive** and must
**not** be dropped into a single cross-sectional regression in v1 — that invites
collinearity and false attribution. Each tilt is an independent clean mean vs the
universe. (An orthogonalized contribution decomposition is a deliberate,
clearly-labeled v2, not the default.)

### 4.2 Style-factor spreads — a free factor monitor

**Function:** `compute_style_spreads(asset_table) -> dict`

From the Style tags, construct daily long-minus-short spreads purely from the ETF
cross-section — **no separate factor-return feed**:
- Value − Growth, Momentum − (universe), Quality − (universe), LowVol − (universe)

Each spread = mean return of funds tagged X minus the comparison mean. Report
today's value and — if a short history table exists — its **percentile** within a
trailing window (reuse the existing `return_percentile` pattern; a spread history
can be persisted to `report.db` the same way other series are). Flag extremes.

Value: a poor-man's factor monitor at zero data cost, and an **independent
cross-check** against the T2 factor-timing work (different data source entirely).

### 4.3 Region × Sector cross-tab

**Function:** `compute_region_sector_grid(asset_table) -> pd.DataFrame`

Because Region and Sector are separate tags, build the grid tier2's single slot
can't: mean `return_1d` for each (Region, Sector) cell with `n >= 3`. Yields
"European Financials +2%, US Tech −1%, EM Materials +1.5%" — far richer than
"Sector Indices +0.4%". Ships as a ranked text table; natural as a heatmap if viz
is ever added (a `tufte-viz` small-multiple).

### 4.4 Breadth within the dominant tags

**Function:** extend §4.1 output with, per top tilt-tag: share of member funds
moving the tilt's direction, and cross-sectional dispersion (std of member
returns). Turns "Energy +1.5%" into "Energy +1.5%, 90% of energy funds positive"
(broad macro move) vs "+1.5% but half negative" (one outlier dragging the mean).
Reuses the breadth idiom already in `compute_bridge`. Confidence-qualifies the
narrative, which is what makes it trustworthy.

---

## 5. Explicitly rejected (with reasons)

- **Active-vs-passive dispersion as a daily signal.** The Strategy tag supports it,
  but daily active/passive divergence across ETFs is mostly noise, and the
  Active/Passive tag is among the *least* reliable labels. Low signal, easy to
  over-read. Skip.
- **Any tag-momentum / tag-predicts-returns feature.** Same trap as the
  CNN-regime idea: tags are descriptive, not predictive. Out of scope by design.

---

## 6. Wiring & file manifest

```
report/
  analytics.py   # ADD: explode_tags, compute_tag_tilts, compute_style_spreads,
                 #      compute_region_sector_grid  (pure functions, new dict keys)
  prompt.py      # ADD: a "what kind of day" section consuming tag_tilts +
                 #      style_spreads + region_sector_grid, gated on
                 #      SETTINGS["enable_tag_views"]
  config.py      # ADD: SETTINGS["enable_tag_views"] = False  (flip on post-labeling)
  main.py        # ADD: call the new functions, persist spread history to report.db
```

`compute_market`'s return dict gains keys: `tag_tilts`, `style_spreads`,
`region_sector_grid`. Nothing existing is renamed or removed.

---

## 7. Acceptance criteria

1. All new functions are pure (DataFrame in, DataFrame/dict out), unit-testable in
   isolation, and honor the "missing data excluded, never zero-filled" rule.
2. With `enable_tag_views=False`, report output is **byte-identical** to today's —
   proves zero regression to the P&L pipeline.
3. With the flag on, the report leads its market section with a correct
   "what kind of day" summary naming the dominant axis and top 2-3 tilts.
4. `compute_tag_tilts` never sums tilts and never regresses correlated tags
   against each other in v1.
5. Every aggregate uses the `n >= 3` floor; thin tags are dropped, not shown with
   a one-fund "mean".

---

## 8. Sequencing

1. Build §3 helper + §4.1 `compute_tag_tilts` (the headline) behind the flag.
2. Add §4.2 spreads + §4.3 grid + §4.4 breadth.
3. Wire `prompt.py` narrative section, flag still off.
4. **Gate:** once the expert-review training round lands the corrected universe,
   flip `enable_tag_views=True` and review the first few reports for tag-quality
   sanity before trusting the prose.
