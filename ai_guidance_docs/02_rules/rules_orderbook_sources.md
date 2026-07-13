---
doc_status: active
default_read: routed
owner: user+agent
purpose: Orderbook readiness, timestamp safety, and usage rules.
do_not_use_for: News/context readiness.
last_rebuilt: 2026-07-13
---


# Rules - Orderbook Sources

## Maintained-fork exception

The current checkout contains a user-approved, temporarily retained historical Bybit integration under `freqtrade/**`. It provides the custom archive download/conversion commands used by `user_data/Custom_Launcher/launcher_v2/services/orderbook_service.py`.

1. This is a tolerated existing exception to the upstream-core protection rule, not permission to add more orderbook functionality to Freqtrade core.
2. Do not modify or expand the core integration without explicit user approval naming the files and behaviour.
3. Before an upstream Freqtrade update, test the integration in an isolated branch/worktree.
4. If it conflicts with upstream or complicates the update, extract the downloader/converter and launcher entrypoint into a standalone `user_data/**` tool, then remove the core integration.
5. Any extracted implementation must preserve the `D:` raw archive locations and the gap/timestamp rules in this document.

## Valid uses

Orderbook may be used for:

1. confirmation,
2. contradiction,
3. risk/invalidation,
4. target-zone evidence,
5. liquidity-vacuum/path-through evidence,
6. crash/fragility acceleration,
7. venue agreement/divergence.

## User terminology for orderbook sources

If the user says `live orderbook` or `Binance orderbook`, use the streamed Binance orderbook data and its live-source readiness masks.

If the user says `historic orderbook`, `historical orderbook`, `Bybit`, or `archive orderbook`, use the historical Bybit archive data and the `D:` raw archive locations below.

If the user says `live plus orderbook`, clarify whether they mean live Binance orderbook or historical Bybit archive orderbook before running tests.


## Raw archive locations

Historical Bybit raw ZIP archives are intentionally stored on `D:`, not under `C:\FreqTradeStuff\user_data`:

1. Root: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit`
2. Spot raw: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\spot_raw`
3. Linear raw: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`
4. Inverse raw: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\inverse_raw`
5. Features/logs/manifests: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit`

Do not infer incomplete raw coverage from `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw` being absent or empty.

## Required source-detail naming

Name the exact venue/source blocks used:

1. `orderbook_spot`
2. `orderbook_bybit_linear`
3. `orderbook_bybit_inverse`
4. any derived family such as wall persistence, wall evaporation/removal, support/resistance rebuild, liquidity vacuum, pressure flip, spread/fragility, venue agreement/divergence.

## Anti-lookahead readiness masks

Orderbook availability must be enforced in the actual masks, not only in report text.

Minimum mask logic:

1. `present == true`
2. `low_coverage == false` unless the test is explicitly about low coverage.
3. `source_future_violation == false`.
4. If a source max timestamp column exists: `source_max_ts <= feature_hour`.
5. If a coverage-ratio column exists: require the chosen minimum coverage threshold and report the threshold.

A readable equivalent is:

`usable_orderbook_row = present & ~low_coverage & ~source_future_violation & (source_max_ts <= feature_hour when available)`

## Missing archive/gap rules

1. Missing orderbook is not zero pressure.
2. Missing orderbook must not become balanced/quiet/no-wall state.
3. State must not carry through raw archive boundaries.
4. Each archive/day should start from its own snapshot state where the raw format requires it.
5. Rolling, shift, duration, and persistence features must be gap-safe inside contiguous coverage segments.

## Required reporting columns

Reports should include, where available:

1. present rows,
2. usable rows,
3. missing rows,
4. low-coverage rows excluded,
5. future-timestamp rows excluded,
6. late/max-source timestamp rows excluded,
7. coverage threshold used,
8. venue/source groups required.

## Known historical interpretation

Previous orderbook work found promising risk/invalidation/failure-state evidence, but broad generic orderbook filters often did not improve results. Treat orderbook as targeted confluence/risk/target evidence until stronger proof exists.
