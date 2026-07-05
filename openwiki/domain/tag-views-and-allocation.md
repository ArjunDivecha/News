# Tag Views and Household Allocation

This repository's tag layer adds a second lens on the book: instead of only market beta and peer groups, it classifies holdings across canonical axes and turns those tags into table-first views in the report.

## Canonical tagging model

`report/tags.py` is the tag-resolution layer. It combines several sources in priority order:

1. Manual overrides for holdings the owner knows better than the classifier.
2. Universe tags for clean universe rows, except where forced corrections are needed.
3. Cached tag values from `report.db`.
4. Fresh classification using real facts gathered from Yahoo / other lookup paths and then passed into the classifier.

The cache lives in the SQLite database so repeated runs are fast and stable.

Important behavior:

- Holdings are force-reclassified fresh when needed, because some universe tags are known to be contaminated.
- The code prefers canonical vocabulary rather than free-form labels.
- The tag cache is persisted even for override-sourced entries so a no-fetch pass still has usable data.

## Tier-3 tag analytics

`report/tag_analytics.py` builds the report's tier-3 section. It is additive and can be disabled through settings, but when enabled it provides:

- market day-type analysis,
- tag leadership tables,
- style and region spreads,
- region × sector grids,
- portfolio tag tilts versus a 60/40 ACWI/TLT benchmark,
- the bridge between portfolio positioning and today's tape,
- per-axis tag P&L attribution,
- exposure-vs-beta reconciliation,
- concentration metrics.

The model is deliberately multi-label and axis-aware. That means:

- tags are not summed across axes,
- correlations between tags are acknowledged rather than ignored,
- benchmark tilts are calculated against pinned benchmark tags, not a loose approximation.

## Household asset allocation

The same tag layer also supports the household allocation table.

`compute_asset_allocation()` in `report/tag_analytics.py` turns the live book plus GMO and manual sleeves into one hierarchical allocation table with:

- Equities
  - US
  - International
  - EM
- Bonds
  - US
  - EM
- Alternatives
- Cash

Important rules:

- Multi-asset and global funds are looked through using `config.FUND_LOOKTHROUGH`.
- A fund without look-through data is not invented into a fake bucket; it can remain unclassified.
- Off-broker sleeves such as Baupost are included via `MANUAL_HOLDINGS`. Baupost receives a synthesized look-through proxy return when it has no daily mark; private company stakes (e.g. Anthropic, Perplexity) do not — they render as em dashes until `market_value` is updated in config with a new mark.
- Bucket returns are P&L over gross exposure so shorts behave correctly.
- Parent rows are supported by child region rows.

## Why this exists

This layer makes the report more decision-useful. The owner wants to know not just what moved, but whether the book was positioned in the right style, region, or factor regime, and whether hidden concentration or beta is building up.

The tag work also makes the allocation and scenario sections more faithful, because those sections use the same look-through decomposition and canonical tags.

## Change guidance

If you are changing positioning, allocation, or tag interpretation, start here:

- `report/tags.py` for canonical label and cache behavior
- `report/tag_analytics.py` for the actual views and allocation math
- `report/config.py` for benchmark composition, manual holdings, and look-through tables
- `report/prompt.py` for how those tables are rendered into the report package
- `tests/test_tag_analytics.py` for the contract that must stay true

Watch out for:

- breaking the NO-NA rule,
- fabricating a `Global` bucket when look-through data is missing,
- summing correlated tag tilts across axes,
- forgetting that the benchmark tag map is pinned,
- changing allocation behavior without updating scenario risk, which uses the same look-through logic.

## Source references

- `report/tags.py`
- `report/tag_analytics.py`
- `report/config.py`
- `report/prompt.py`
- `tests/test_tag_analytics.py`
