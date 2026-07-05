# Market, Portfolio, and Bridge Analytics

`report/analytics.py` contains the core financial math for the daily report. This is the main place to study when you need to understand how the report measures returns, exposures, alpha, bridge attribution, or sub-portfolio performance.

## What it computes

### Market analytics
`compute_market()` builds the market-side picture from the universe and price matrix. It returns:

- a per-asset table with 1d, 1w, 1m, YTD, volatility, percentile, beta, and metadata,
- tier-1 and tier-2 summaries,
- factor tables and factor correlations,
- movers and streaks,
- basic market data quality metrics.

Important behavior:

- The market view only includes universe tickers.
- SPY must be present; if not, the function fails loudly.
- The function uses equal-weighted means for tier summaries.
- Streaks are only emitted for meaningful runs of consecutive up/down days.

### Portfolio analytics
`compute_portfolio()` is the main book-level calculator.

It:

- aggregates raw broker rows to one row per symbol,
- separates cash from priced positions,
- marks positions to market when a price exists,
- recomputes open P&L where the broker did not supply it,
- calculates signed gross/net exposures,
- derives current weights versus gross,
- computes beginning-of-day weights for exact contribution tie-out,
- computes 1d contribution in basis points,
- computes YTD on a current-weights proxy basis,
- estimates portfolio beta, expected return, and alpha versus SPY,
- builds a factor-exposure table,
- records data-quality metadata for unpriced positions.

The implementation is explicit about a few important conventions:

- shorts keep their sign naturally,
- contributions are in basis points,
- returns are in percent,
- unpriced positions are included in exposure but excluded from realized return calculations,
- YTD is a proxy, not a realized full-year number.

### Sub-portfolio analytics
`compute_subportfolios()` summarizes each broker/account sleeve, plus GMO and manual off-broker sleeves when present.

This matters because the report's household table includes more than the live Schwab + IBKR book. The function:

- preserves account labels from `config.ACCOUNT_NAMES`,
- uses last-available daily returns so stale NAV funds do not disappear,
- uses the correct beginning-of-day base for stale holdings,
- supports extra holdings such as Baupost,
- can override a manual sleeve's 1d/YTD return with a synthesized look-through proxy.

### The bridge
`compute_bridge()` connects the market and the book.

It provides:

- per-position attribution versus tier-2 peers,
- a list of unheld themes that moved the most,
- portfolio breadth statistics.

The bridge logic is intentionally honest about universe coverage:

- If a held symbol is priced but not in the universe, it is labeled `Portfolio-Specific`.
- If a universe label exists but the symbol is not priced today, the tier-2 backfill still allows peer comparison.
- Peer returns are only computed when the peer group has enough members.

## Practical conventions

These conventions are used throughout the report and enforced by tests:

- **Weights** are signed and measured against gross exposure.
- **1d contributions** use beginning-of-day weights so they reconcile to dollar P&L.
- **YTD** is a current-weights proxy.
- **Missing data** is not hidden; the code prefers explicit `NaN` handling and separate data-quality reporting.
- **Stale data** is last-available data with a stale marker, not a fabricated zero.

## Change guidance

If you are changing the math, check these functions first:

- `aggregate_holdings()`
- `compute_portfolio()`
- `compute_subportfolios()`
- `compute_bridge()`
- helper functions for return windows, betas, percentiles, and return normalization

Tests that matter most:

- `tests/test_analytics.py`
- `tests/test_report_pipeline.py`

Watch out for regressions around:

- short positions changing sign,
- stale NAV or holiday handling,
- YTD dilution from missing data,
- contribution scaling in bps,
- bridge labels for off-universe holdings,
- factor exposure labels and names.

## Source references

- `report/analytics.py`
- `tests/test_analytics.py`
- `tests/test_report_pipeline.py`
