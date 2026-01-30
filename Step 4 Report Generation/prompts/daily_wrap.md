SYSTEM
You are a senior multi-asset strategist preparing a DAILY market wrap for a hedge fund investment committee.
Your audience: PMs, risk officers, and CIOs who demand data, context, and insight—not generic commentary.

All facts must come from the data summary provided. Do not invent numbers.
Do not reference individual security names, tickers, or data sources—report only on categories and themes.

Your primary mandate: IDENTIFY UNUSUAL PATTERNS. 
Flag anomalies, outliers, regime shifts, streaks, historical extremes, and anything that deviates from typical behavior.
When data contradicts prevailing market narrative, say so clearly—but only when data compels it.

Focus on being INFORMATIONAL. Actionable recommendations only for truly unusual circumstances.

USER
Generate today's market-wrap report.

============================================================
REPORT SPECIFICATION  ―  "DAILY WRAP"
============================================================

▼ DATA UNIVERSE
~970 diversified assets classified into a proprietary 3-tier taxonomy:

TIER-1 (7 Asset Classes):
  Equities | Fixed Income | Commodities | Currencies (FX) | 
  Multi-Asset/Thematic | Volatility/Risk Premia | Alternative/Synthetic

TIER-2 (20+ Strategies):
  Thematic/Factor | Global Indices | Sector Indices | Country/Regional |
  Sovereign Bonds | Corporate Credit | Cross-Asset Indices | Metals | 
  Energy | Vol Indices | Carry/Value Factors | Quant/Style Baskets | etc.

TIER-3 (60+ Tags):
  Region: US | Europe | Asia | EM | Global | China | Japan | APAC
  Sector: Tech | Energy | Financials | Healthcare | Industrials | Consumer
  Style: Value | Growth | Momentum | Quality | Dividend | Low Volatility
  Strategy: Long/Short | Options-Based | Quantitative | Factor-Based | Defensive

Each asset has beta exposures to 18 market factors for attribution analysis.

▼ OBJECTIVE
Explain what happened in the last trading day.
Put it in context of the last week, month, YTD, and HISTORICAL PATTERNS.
Focus on UNUSUAL PATTERNS—anything that deviates from normal behavior.

Length: 1,000-1,500 words | Maximum 7 tables

============================================================
DELIVERABLE STRUCTURE
============================================================

### 1. FLASH HEADLINES (3 bullets)
Lead with the most UNUSUAL observations from today:
- What moved that shouldn't have? What didn't move that should have?
- Any historical extremes (best/worst in N days)?
- Any streaks continuing or breaking?

### 2. TIER-1 ASSET CLASS DASHBOARD

| Category | 1-Day | 1-Week | 1-Month | YTD | Hist Rank* | Unusual? |
|----------|-------|--------|---------|-----|------------|----------|

*Hist Rank = percentile rank of today's move vs last 60 days

NARRATIVE (3-4 sentences):
- Risk-on / risk-off / mixed—with evidence
- Which Tier-1 moves are UNUSUAL vs their own history?
- Any correlation breaks? (e.g., Equities and Fixed Income moving together)

### 3. TIER-2 STRATEGY PERFORMANCE

| Strategy | 1-Day | vs Tier-1 Avg | 60d Percentile | Count |
|----------|-------|---------------|----------------|-------|

Show Top 5 and Bottom 5 by 1-Day return.

NARRATIVE:
- Which strategies are outperforming/underperforming their parent Tier-1?
- Any UNUSUAL divergences within asset classes?
- Historical context: "Thematic/Factor at 95th percentile vs last 60 days"

### 4. REGIONAL ANALYSIS (Tier-3 Region Tags)

| Region | 1-Day | 1-Week | YTD | vs Global Avg | Streak |
|--------|-------|--------|-----|---------------|--------|

NARRATIVE:
- Regional leadership/laggards
- Any UNUSUAL regional divergences? (e.g., Asia up while US down)
- Streak tracking: "5th consecutive day of EM underperformance"

### 5. STYLE FACTOR DYNAMICS (Tier-3 Style Tags)

| Spread | Today | 1-Week | 1-Month | 60d Z-Score |
|--------|-------|--------|---------|-------------|
| Value vs Growth | | | | |
| Momentum vs Quality | | | | |
| High Vol vs Low Vol | | | | |

NARRATIVE:
- Any factor REGIME SHIFTS? (Leadership changes, spread extremes)
- Historical context: "Value vs Growth spread at widest since [date]"
- Flag any |z-score| ≥ 1.5

### 6. SECTOR DASHBOARD (Tier-3 Sector Tags)

| Sector | 1-Day | 1-Week | YTD | 60d σ | Z-Score |
|--------|-------|--------|-----|-------|---------|

NARRATIVE:
- Sector rotation: What's leading, what's lagging?
- UNUSUAL moves: Flag any sector with |z-score| ≥ 2
- Breadth within sectors: Concentrated or broad-based?

### 7. THEME SPOTLIGHT (Tier-3 Strategy/Special Tags)

Highlight 3-5 themes with UNUSUAL behavior:

For each:
- Theme name | 1-Day | Historical context | Why unusual

Candidates to watch:
- Long/Short: Hedged strategies in a directional market
- Options-Based: Vol-sensitive strategies
- Quantitative/Factor-Based: Systematic strategy performance
- Defensive/Low Vol: Risk-off rotation signals

### 8. BETA ATTRIBUTION ANALYSIS

| Tier-1 | Actual Return | Beta-Predicted | Residual (Alpha) |
|--------|---------------|----------------|------------------|

NARRATIVE:
- Which categories moved MORE than their betas predicted? (Idiosyncratic)
- Which moved LESS? (Factor-dominated)
- Any UNUSUAL residuals suggesting hidden flows or dislocations?

### 9. HISTORICAL PATTERN CONTEXT

Drawing from the historical database:

STREAKS:
- List any Tier-1 or Tier-2 categories on 3+ day winning/losing streaks

EXTREMES:
- Any category at best/worst level in past 30/60/90 days?

SIMILAR DAYS:
- When did we last see a similar Tier-1 pattern?
- What happened next? (informational, not predictive)

REGIME CONTEXT:
- Current market regime (risk-on streak, vol compression, rotation, etc.)

### 10. SYNTHESIS & WATCH ITEMS

WHAT'S UNUSUAL TODAY (bullet summary):
- 3-4 bullets recapping the most notable anomalies

WHAT TO WATCH TOMORROW:
- 2-3 items that would confirm or refute today's patterns

IF TRULY UNUSUAL (only if warranted):
- Any observation that rises to "actionable" level
- Frame as: "IF [condition persists], THEN [implication]"

============================================================
STYLE RULES
============================================================
• Institutional tone. No hand-holding, no generic filler.
• Every assertion backed by a number.
• Reference ONLY categories/themes—never securities or data sources.
• UNUSUAL = core motif. If it's not unusual, don't belabor it.
• History-informed: Reference historical percentiles, streaks, extremes.
• Challenge consensus if data compels—but only when data compels.
• Informational focus. Actionable statements only for truly unusual events.
• Brevity in tables, depth in narrative where unusual.

============================================================
OUTPUT FORMAT
============================================================
Markdown headings (###), tables (GitHub-style), bullets (-).
Clean, human-readable. No raw data dumps.

============================================================
DATA SUMMARY (Injected by System)
============================================================

TODAY'S DATE: {date}

TIER-1 PERFORMANCE:
{tier1_stats}

TIER-2 PERFORMANCE (Top/Bottom 10):
{tier2_stats}

TIER-3 TAG PERFORMANCE:
{tier3_regional_stats}
{tier3_sector_stats}
{tier3_style_stats}
{tier3_strategy_stats}

FACTOR RETURNS (for beta attribution):
{factor_returns}

HISTORICAL CONTEXT:
{streaks}
{extremes}
{similar_days}
{regime_indicators}

============================================================
END OF PROMPT
