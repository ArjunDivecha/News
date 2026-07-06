# PRD — Action-Oriented Report + "Fable's Desk" Intelligence Section

**Date:** 2026-07-05
**Status:** DECIDED & PARTIALLY IMPLEMENTED 2026-07-06. The principal's
decisions: (1) the no-recommendations directive is removed ("outdated"),
(2) the policy is outcome-based — **real return > 5%/yr at vol ≤ 60/40** —
NOT allocation bands (§R2's band sketch is superseded), (3) Fable's Desk is
approved and built. Shipped: system.md recommendation contract + Action Box,
`POLICY` config + `compute_policy_check` + POLICY CHECK package section,
`asado.py` + `fable_desk.py` + `prompts/fable_desk.md` + `desk_ideas` table.
Still open from this PRD: R3 (thesis watch), R4 (60/40 alpha), R5 (spread
history percentiles), R6 (serialization tweaks), R7 (IBKR gateway, Muni YTD).
**Prepared from:** deep dive of `report/` internals, the actual household portfolio
(2026-07-02 snapshot, $99.0M), web research on investment-committee reporting
practice, live world-state research (July 5, 2026), and the ASADO warehouse
(`/Users/arjundivecha/Dropbox/AAA Backup/A Working/ASADO/Data/asado.duckdb`).

---

## 1. Diagnosis — where the report stands today

The daily report is a **world-class descriptive engine that is deliberately
forbidden from being action-oriented**. Two contract clauses do it:

1. `report/prompts/system.md:24` — "Ideas are welcome; recommendations are
   not... never as trades to execute." Reiterated at line 94: "Never recommend
   trades. He makes the decisions."
2. `report/prompts/system.md:37` — "you have no news feed... must not assert
   specific news events occurred."

Everything else about the pipeline is consistent with that choice: one
single-shot LLM call with **all tools disabled** (`report/llm.py:100`,
`--tools ""`), a data package that is 100% internal price/holdings analytics,
and continuity limited to 5 days of executive-summary prose
(`report/db.py:327-335`).

Consequences, concretely, as of the 2026-07-02 report:

- **The report has never mentioned**: the Strait of Hormuz closure (since
  Feb 28) with Brent ~$105 while the household holds zero energy; the yen at a
  40-year low (162.8) with the USD-spike scenario being the only one in its own
  stress table **with no cushion at all**; the **Vietnam FTSE EM upgrade
  effective 2026-09-21** — a hard, dated catalyst for the household's largest
  live-book position (Vietnam Enterprise Investments, 17.7% of gross, $5.6M);
  the Anthropic $965B re-rating and possible October IPO (a held private
  stake); the Nov 3 midterms. None of this *can* appear — the writer isn't
  allowed to know it.
- **No quantified "extreme" context**: spreads/tilts are shown raw with no
  historical percentile or z-score (PRD_v2's M5 was never built — no
  spread-history table exists in `report/db.py`), so the report cannot say
  "today's Value-vs-Growth spread is a 97th-percentile event," which is
  exactly what an action trigger needs.
- **No scorecard**: signposts and watchlist items are set daily but nothing
  tracks whether they resolved. The Watchlist is the report's most
  action-adjacent artifact (system.md:76) and it evaporates every 24 hours.
- **The headline alpha is still the one the PRD called "misleading"**:
  PRD_Tier3_Enhancements_v2 locked decision #1 replaced SPY-only alpha with
  the 60/40 ACWI/TLT blend; `report/analytics.py:420-434` still computes alpha
  vs SPY only, and the system prompt still carries the caveat machinery
  (system.md:34, 64).

### The portfolio the report serves (2026-07-02)

$99.0M household: GMO sleeve $43.1M (44% of household with one manager —
Equity Dislocation alone is $22.4M / 22.6%), Baupost LP $13.8M, live
Schwab+IBKR book $39.9M incl. $13.6M cash. Live book: 58 positions but 12.9
effective; 97% of gross is equity; top position Vietnam Enterprise 17.7% of
gross; −8% Russell 1000 Growth short; frontier/commodity EM complex (South
Africa 6.4%, Brazil 6.3%, Turkey 4.7%, Mexico 4.1%) with realized EM beta
0.43. Household cash 18.6%. Kill shot: US −40% → −$20.2M with a $1.0M
cushion. Only scenario with zero cushion: USD +10%.

---

## 2. What the IC-reporting research says (condensed)

Full research report with 23 sources lives in the session log; the synthesis:

**The features that make reports drive decisions** (from PE/IC memo practice,
OCIO/endowment governance, BBH rebalancing policy, GMO letters, sell-side
top-trades schemas):

| # | Feature | In today's report? |
|---|---|---|
| 1 | Explicit recommendation with a verb (Add/Trim/Hold/Watch) | ✗ forbidden |
| 2 | Stated conviction level | ✗ |
| 3 | Time horizon per view | ✗ |
| 4 | Pre-committed invalidation ("what would change my mind") | ~ (signposts exist, but attach to observations, not positions) |
| 5 | Link to policy targets/bands (why *now*, mechanically) | ✗ no IPS/targets exist in the system |
| 6 | Exception-based filtering (default = silence) | ~ (Worth Knowing is exception-based; rest is full recitation) |
| 7 | Claim-style headlines (conclusion, not category) | ✗ fixed section names |
| 8 | Downside-first, quantified risks with trip wires | ✓ Scenario Risk + Watchlist — the report's best feature |
| 9 | Ranked "ask" — the 1-3 decisions needed today, on top | ✗ |
| 10 | Scenario/base-rate anchoring | ✓ episode-calibrated scenarios |
| 11 | Standing theses updated, not re-invented (+ decision journal) | ✗ only 5 days of prose continuity |
| 12 | Valuation anchors over news anchors | ~ (percentiles/YTD, but no valuation layer — ASADO has it) |

**Cadence discipline** (the anti-recency-bias defense, and the most important
design constraint for a *daily* product): daily = exceptions/triggers only,
with "no action required" as the default and a feature; weekly = thesis review
and conviction re-rating; quarterly = full allocation review and
decision-journal retrospective. A daily report that manufactures a fresh trade
every day is the recency-bias trap the literature warns about — the Fable
section below is explicitly allowed to say "nothing today."

---

## 3. Recommendations — the deterministic report

Ordered by impact. R1–R3 change the product; R4–R6 fix known debts; R7 is ops.

### R1. Add an **Action Box** at the top; renegotiate the no-recommendations clause ⚖️
New first section, before the Executive Summary. 0–3 items, each:

> **Verb + instrument + size** · conviction (H/M/L) · horizon · the trigger
> that fired (mechanical: band breach, percentile extreme, streak, scenario
> change) · invalidation.
> Empty state: **"No action required — all bands hold."** (printed proudly,
> most days.)

This requires consciously amending system.md's "never recommend trades." The
research-backed middle ground, fitting how you already operate: recommendations
are **proposals tied to mechanical triggers** (band breach, percentile
extreme), never free-floating opinions — the deterministic report proposes only
what a rule fired; the judgment-based ideas live in Fable's Desk (§4), clearly
labeled. You still make every decision; the report stops hiding its conclusion.

### R2. Define policy targets + bands; report **breaches only** ⚖️
There is currently no IPS in the system — nothing to be "action-oriented"
*against*. Add to `report/config.py`:

```python
POLICY = {  # target %, band ± pp (absolute at bucket level)
    "Equities": (40, 5), "Bonds": (14, 3), "Alternatives": (28, 5),
    "Cash": (18, 5),
    # sub-bands, relative ±25% of target (BBH convention):
    "Equities/EM": (24, 6), "Equities/US": (7, 2), ...
    # concentration & risk lines:
    "max_single_position_pct_gross": 15,   # Vietnam is 17.7% today → would flag
    "max_single_manager_pct_household": 40,  # GMO is 44% today → would flag
    "min_two_day_liquidity_pct": 30,
}
```
⚖️ **The numbers above are illustrative placeholders — they need your actual
targets.** Note both concentration lines would flag *on day one* (Vietnam
17.7% of gross; GMO-managed 44% of household): decide whether those are
intentional standing positions (then set the band around them) or genuine
flags. New "Policy" table in the report: only rows in breach, each with a
so-what and a proposed action (feeds R1). Tax-aware ordering note (use
inflows/IRA first) per BBH practice.

### R3. **Thesis Watch + scorecard** — give the report a memory
- New db table `theses`: 3–6 standing theses (e.g. "growth short pays in the
  AI unwind," "frontier/commodity EM ≠ EM factor," "metals handoff from
  energy"), each with stance, conviction, invalidation, opened date. The
  report updates each thesis in one line a day ("no change" most days —
  that's the anti-recency defense), and owns re-rates.
- New db table `signposts`: every Watchlist item and forward-looking claim
  gets stored with its falsifier; each report grades yesterday's open items
  (confirmed / killed / open). Monthly: hit-rate table — the decision journal
  the IC literature keeps finding improves process quality.
- This also upgrades continuity from "5 days of prose" to structured state.

### R4. Ship the shelved PRD decision: benchmark alpha vs 60/40 ACWI/TLT
Locked in PRD_v2:191-204, still not implemented (`analytics.py:420-434`).
The blend already exists in config (`BENCHMARK`, config.py:123) and the tag
layer already uses it. Replace the headline SPY-only alpha; keep SPY beta as a
secondary line. Kills the daily alpha-caveat boilerplate.

### R5. Build spread/tilt **history percentiles** (PRD_v2 M5)
Persist daily style/region spreads and tag tilts to report.db; render
percentile/z vs trailing 1y next to each. "Value−Growth +1.2pp (96th pctile)"
is a trigger; "+1.2pp" is trivia. This is the data substrate R1's mechanical
triggers stand on.

### R6. Small data-package upgrades (cheap, already computed)
- Long/short/gross/net **trend** and position-count trend in PORTFOLIO
  HISTORY (columns exist in `portfolio_summary`, not serialized —
  `prompt.py:584-588`).
- Top-5 *most* correlated factor pairs alongside the 5 least
  (`prompt.py:397-410`) — crowding context.
- Tier-2 1m returns (`analytics.py:221-226` computes 1d/1w/YTD only).
- Recent-trades diff: the 6/29→7/2 snapshot diff shows 10 exits (~$1.6M,
  QQQ/VNM/EWL/...) — the report never mentions your own trades. Add a
  "changes since last report" line derived from `portfolio_snapshots`.

### R7. Ops debts the deep-dive surfaced
- **IBKR Gateway has been unreachable for weeks** (`daily_run.log`;
  holdings stale-by-policy on most runs). The Flex Web Service path exists
  (`ibkr_flex.py`) — worth a diagnosis pass.
- **Muni ($5.0M) and Dancing Elephant show +0.00% YTD** — likely a
  missing-return artifact rendering as a real zero; per house rules that
  should be a `*` stale value or an em dash, not a fake 0.00.
- Private marks: add `mark_date` + source to MANUAL_HOLDINGS and print it
  ("marked 2026-07 @ Series H $965B post") so staleness is auditable in the
  report itself.

---

## 4. "Fable's Desk" — the intelligence-based section

**What it is:** a second, clearly-labeled section written by Fable with its
full world knowledge switched on — live web news + the ASADO warehouse + the
portfolio — producing **idiosyncratic, non-deterministic ideas**: things no
rule would generate. Explicitly *not* rule-based: no checklist, no fixed
rubric; the mandate is "what would you tell him at the morning meeting if you
were the smartest macro PM he knows."

### Architecture

```
main.py (unchanged, deterministic report)
   └── fable_desk.py (new, second pass, ~5-10 min budget)
        inputs:  today's data package + rendered report
                 + open theses/ideas from db (continuity + scorecard)
        tools:   WebSearch / WebFetch          (live world)
                 asado_query (read-only DuckDB) (34-country signals, GDELT
                 tone/attention, factor returns, commodities, predmkt)
        output:  "## Fable's Desk" markdown appended to the report
                 + ideas persisted to db table `desk_ideas`
```

- **Invocation:** Claude CLI with tools enabled (today's call passes
  `--tools ""` — llm.py:100; the Desk pass instead allows WebSearch and a
  small MCP/CLI helper exposing `asado_query(sql)` read-only), or the Agent
  SDK. Model `claude-fable-5`, effort high/xhigh. Keep it a **separate call**
  so the deterministic report's no-fabrication guarantees stay intact.
- **Feature flag** `SETTINGS["enable_fable_desk"]`, same pattern as
  `enable_tag_views` (config.py:283-285) — staged rollout, easy kill switch.
- **ASADO freshness contract:** verified live 2026-07-05 — `t2_factors_daily`
  through 2026-07-03 (34 countries × 107 vars), `gdelt_factors_daily` through
  2026-07-02, `factor_returns_daily` through 2026-07-03. The Desk prompt gets
  the DB map (`ASADO-gmd-ingest/ASADO_DATABASE_MAP.md`) distilled: the
  `_CS`/`_TS` convention, sign convention (higher = more attractive), exact
  country strings, and the "returns are the source of truth" doctrine.

### Output contract (the section's rules)

1. **0–4 ideas, most days 1–2, some days zero.** "Nothing worth your
   attention today; here's what I checked" is a first-class output.
2. Every idea carries: **thesis (2-4 sentences) · the trade expression ·
   conviction (H/M/L) · horizon · invalidation · what I'd watch.** The
   sell-side top-trades schema, minus the sell-side.
3. **Provenance discipline:** portfolio numbers only from the package; ASADO
   numbers only from queries actually run; world claims only from fetched
   sources, cited inline. Judgment is welcome; it must be *labeled* judgment.
4. **Anti-recency guard:** yesterday's ideas are in context; repeating one
   requires new information, and every open idea gets scored over time
   (`desk_ideas` table: date, idea, expression, conviction, invalidation,
   status, outcome). The monthly hit-rate table keeps the Desk honest —
   intelligence with accountability.
5. **Tone:** first person, opinionated, allowed to disagree with the
   deterministic report above it and say so.

### Why this split is right
The deterministic report stays a trustable instrument (numbers-only, no news,
falsifiable). The Desk is where non-determinism lives — same day, same data
could legitimately produce different ideas, which is the point: it's judgment,
tracked like judgment. Cost: roughly one extra Fable call with tool use per
day (~5-10 min wall clock, well inside the nightly window).

---

## 5. Worked example — what Fable's Desk would say **today** (2026-07-05)

*Written with live web research (July 5) + the actual 2026-07-02 portfolio +
ASADO queries run 2026-07-05. This is the demo of §4's output contract.*

---

### Fable's Desk — Sunday, July 5, 2026

The tape's two-day momentum unwind is the noise; three dated catalysts your
book already straddles are the signal. ASADO's July 3 snapshot says your book
is, country by country, long the cheap side of the world and short the
expensive side — Brazil +1.3σ, Turkey +1.0σ, South Africa +0.7σ on
cross-sectional value vs NASDAQ −2.3σ, Taiwan −1.9σ, U.S. −1.3σ — so the
structure needs no fixing. These are the places where I'd act or pre-commit:

**1. Vietnam: pre-write the trim plan for September 21. — HIGH conviction,
3-month horizon.** Vietnam Enterprise Investments is 17.7% of live-book gross
($5.6M), your largest position, top casualty in three of six stress scenarios
— and it now has a hard, dated catalyst: FTSE Russell confirmed the EM
upgrade effective 2026-09-21, with the World Bank estimating $3–5B of inflows.
Closed-end-fund discounts are exactly what index-inclusion flows close.
ASADO flags Vietnam's currency signal at +4.6σ — the standout FX reading in
the universe. The asymmetry: flows are front-run-able, and you own the
pre-flow position; but at 17.7% you're also one visa-free-China headline away
from giving it back. The action isn't "sell" — it's writing the exit ladder
*now* (e.g., trim to 12% into inclusion-week strength, hold the rest), so
September doesn't become an emotional decision. Invalidation: FTSE delays or
waters down the phasing.

**2. The dollar is the one fire you're not insured against — and it's
burning. — HIGH conviction, 1-3 months.** Your own scenario table says USD
+10% is the *only* scenario with no cushion (−$4.1M), and it's the scenario
currently happening: yen at a 40-year low (162.8), intervention jitters, the
dollar bid on Iran risk and a Fed that markets briefly priced for a *hike*
after JOLTS. Your EM complex is unhedged ZAR/BRL/TRY/MXN/VND. Two cheap
expressions: (a) long yen — asymmetric at a 40-year extreme with the MoF's
finger on the trigger; Thursday's soft NFP already knocked hike odds down, so
the carry against you is peaking; (b) buy a 3-6 month USD-call/EM-basket-put
overlay sized to neutralize a quarter of the −$4.1M. This is insurance while
implied FX vol is still yawning at the story. Invalidation for (a): clean
break above 165 with Tokyo silent.

**3. Close the biotech short; it's fighting your own tape. — MEDIUM
conviction, this week.** Healthcare is the strongest theme in your universe
(+4.7% 1w excess, an 11-day up streak in FBT) and the market's "Great
Rotation" destination — Lilly crossed $1T, biotech M&A is running. Your only
healthcare expression is a −0.13% SPDR Biotech short: immaterial as P&L
(−$6.2K open), but it's short the single strongest board on the wall while
your book's whole thesis is "own what momentum is rotating *into*." Cover it,
and decide separately whether healthcare-value (not biotech-momentum)
deserves 2-3% of the book — it's the one big defensive-value sector your
value tilt somehow doesn't own. Invalidation of the add-idea: the sector's 1w
excess turns negative.

**4. Hormuz is closed, Brent is $105, and you have zero energy — decide if
that's a view or an accident. — MEDIUM conviction, standing.** The Strait
has been shut for 3+ months, 11M b/d shut in, Qatari LNG damaged — and yet
energy equities are *rolling over* (OIH 5-day down streak), which is the
market pricing resolution or demand destruction. Your book profits from
resolution (EM/value rallies, oil falls) and is naked to escalation — which
would also hit your frontier-EM oil importers (Turkey, Philippines) *and*
re-light the inflation scenario (−$5.9M). A 1-2% position in energy majors or
an oil call spread is not a bet on oil — it's a hedge on the war getting
worse, bought while the market is leaning the other way. I'd note ASADO's
GDELT layer has Hong Kong local tone at −2.2σ (worst in the book's regions)
— the China-adjacent news pulse is deteriorating even while H-share RSI runs
hot; that's the same escalation complex.
Invalidation: verified reopening of the Strait.

**5. For the watchlist, not the blotter:** (a) **Indonesia** — ASADO's
cheapest country (+1.3σ value) with −2.2σ momentum and a +3.1σ currency
signal; you own 1.9% and it's −38% YTD; when its momentum z crosses back
above −1, that's historically the frontier-value setup this book exists for.
(b) **Korea** — still +3.6σ momentum-extended *after* a 16% semi crash; your
1.25% long is small, but Beyond China's 2.13 beta makes Korea your real
transmission line; the AI-capex credit story (BIS warning, ~$600B
capex-revenue gap, $1.5T of coming tech debt issuance) says the growth
short's next leg gets confirmed or killed in *credit*, so add IG tech spreads
to the daily watch. (c) **Anthropic** — $965B Series H, IPO chatter for
October: your $330K mark is fresh, but start the tax/lockup planning now; an
IPO converts a footnote into a position. (d) **Midterms (Nov 3)** — markets
have Democrats ~78% to take the House; gridlock historically compresses
policy tail risk — mild positive for your tariff-sensitive EM complex, worth
nothing today, worth re-checking at Labor Day.

**Scorecard note:** first edition — nothing to grade yet. Ideas 1-4 logged
with invalidations; grades start tomorrow.

*Sources: FTSE/World Bank via vietnam-briefing.com & worldbank.org; EIA STEO
(Hormuz, Brent $105); fxleaders.com & forex.com (USD/JPY 162.8, NFP);
CNBC/Bloomberg (Kospi, Samsung/SK Hynix, July 2); Fortune/BIS annual report &
JPMorgan midyear (AI capex debate); TechCrunch/CNBC (Anthropic Series H);
Kalshi/Polymarket via 270towin (midterms); GMO letters for the idiom. ASADO
queries: t2_factors_daily 2026-07-03 (value/momentum/FX/risk _CS composites),
gdelt_factors_daily 2026-07-02 (HK local_tone_CS −2.15).*

---

## 6. Implementation plan

| Phase | Scope | Size |
|---|---|---|
| 0 | ⚖️ Decisions: amend system.md no-recommendation clause (R1); set real policy targets/bands (R2); confirm Desk scope/tools | discussion |
| 1 | R4 (60/40 alpha — code exists, swap the benchmark) + R6 serialization tweaks + R7 Muni-YTD artifact | small, 1 session |
| 2 | R5 spread-history table + percentiles (schema + backfill from price history) | medium |
| 3 | R2 policy/breach engine + R1 Action Box (new prompt section, mechanical triggers from R5/R2) | medium |
| 4 | R3 theses/signposts tables + scorecard rendering | medium |
| 5 | **Fable's Desk**: `fable_desk.py`, asado_query helper, `desk_ideas` table, prompt (§4 contract), `enable_fable_desk` flag | medium-large, the fun one |
| 6 | R7 IBKR gateway diagnosis (independent, any time) | ops |

Tests follow the house pattern: each phase adds to `tests/` (policy-breach
math, spread-percentile correctness, desk-idea persistence, prompt-contract
golden files).

---

*Prepared by Claude (Fable 5). All portfolio figures from the 2026-07-02 data
package and report.db; all world facts from July 5, 2026 web research; all
country signals from asado.duckdb read-only queries run 2026-07-05.*
