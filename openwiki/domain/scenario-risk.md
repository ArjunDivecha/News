# Scenario Risk

`report/scenarios.py` is the standing stress engine for the report. It answers the question: what happens to the household if a named historical or hypothetical shock hits the current look-through book?

## What it computes

The scenario engine returns three things:

1. a scenario table with estimated household impact,
2. a crash-beta estimate for the current-weight book,
3. a liquidity ladder from cash through illiquid sleeves.

## Scenario design

The current scenario set includes six episode-calibrated shocks:

- US equities -40% / 2008-09 GFC
- Asia / EM crisis / 1997-98 Asia-LTCM
- China / Taiwan event
- Inflation / rates +300bp / 2022 hiking cycle
- USD +10% spike
- Tech / growth crash / 2000-02 dot-com

Each scenario has:

- a name,
- an anchor describing the historical episode or calibration source,
- a class/region shock matrix,
- optional per-symbol overrides for concentrated or special names.

The engine applies shocks to the household's actual look-through slices, not to an oversimplified top-level bucket view. That means multi-asset funds are decomposed before the shock hits, and shorts sign correctly.

## Structural risk measures

### Crash beta
`compute_crash_beta()` compares the current-weights book's beta to SPY in two regimes:

- the full sample,
- the worst-decile SPY days over the lookback window.

The point is not prediction; it is to reveal whether the book behaves differently in drawdowns than its headline beta suggests.

### Liquidity ladder
`compute_liquidity_ladder()` buckets the household into:

- cash,
- ETFs,
- daily-NAV mutual funds,
- closed-end funds,
- LP lockup,
- unpriced lines.

This is useful because the owner's real question is not just downside magnitude but how quickly the household can convert to cash under stress.

## Reporting contract

The data package and system prompt treat Scenario Risk as a required section whenever the package contains it. The report must show:

- the table of scenario impacts,
- the assumption string for each shock,
- the structural panel,
- a short interpretation of the true kill shot and the scenarios the book is better protected against.

## Change guidance

If you are changing scenario logic, inspect:

- `report/scenarios.py` for the actual assumptions and math,
- `report/prompt.py` for the rendered package contract,
- `report/config.py` for look-through data and manual holdings,
- `tests/test_scenarios.py` for the regression coverage.

Watch out for:

- changing assumptions without updating the prompt text or audit trail,
- breaking the shared look-through decomposition used by allocation,
- misclassifying off-broker sleeves or unpriced holdings,
- accidentally removing the auditability of scenario shocks.

## Source references

- `report/scenarios.py`
- `report/prompt.py`
- `report/config.py`
- `tests/test_scenarios.py`
