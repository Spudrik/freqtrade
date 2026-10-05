# Strategy Registry And Bucket Guide

This folder contains the long-term manual strategy review registry.

The registry is separate from Sieve result files. Sieve rows are run-level evidence. The registry is the current strategy-level judgement after reviewing one or more runs.

Primary file:

- `strategy_registry.csv`

## Registry Columns

- `strategy`: Strategy file/class stem, for example `sieve1_pivot_long_trend_pullback`.
- `side`: `long`, `short`, `both`, or blank if unknown.
- `core_behavior`: Short human label for what the strategy is testing.
- `current_bucket`: Manual bucket value from the list below.
- `current_status`: Current lifecycle value from the status list below.
- `last_reviewed_run`: Sieve or backtest run id used for the most recent judgement.
- `last_reviewed_at`: Review date in `YYYY-MM-DD` format.
- `last_result_rows`: Count of result rows inspected for this strategy in the reviewed run.
- `total_trades`: Total reviewed trades across the inspected rows, or the most relevant aggregate.
- `best_tp_sl`: Best observed TP/SL setup, for example `3/3`, `5/1`, or `5/3`.
- `best_winrate`: Best useful win rate observed. Avoid treating one-trade 100% rows as meaningful.
- `best_profit_abs`: Best useful absolute profit, if available.
- `best_drawdown_pct`: Best or most relevant drawdown percentage, if available.
- `profile_3_3`: Compact summary for TP 3% / SL 3%.
- `profile_5_1`: Compact summary for TP 5% / SL 1%.
- `profile_5_3`: Compact summary for TP 5% / SL 3%.
- `next_action`: Short action: rerun, loosen, add_guard, fix_infra, rework, retire, etc.
- `review_notes`: Short manual judgement explaining why the bucket was chosen.

Suggested profile format:

```text
wr=54.2%;tr=1444;p=268;dd=7.1%
```

## Bucket Values

- `unreviewed`: Strategy exists but has not had a manual bucket assigned yet.
- `best_all_round`: Strong under balanced TP/SL, strong when given room, and still positive under tight-stop asymmetric testing.
- `needs_room`: Strong under `3/3` and `5/3`, but weak under `5/1`. Entry may be valid but needs wider or adaptive exits.
- `overtrading_or_noise`: Acceptable or mediocre under `3/3`, then weak under wider or tighter scenarios. Often indicates noisy entries or mean-reversion churn.
- `asymmetric_candidate`: Positive under `5/1` despite low win rate. Useful when tight stops and larger winners produce acceptable expectancy.
- `guard_confluence_candidate`: High trade count, roughly break-even to modest win rate, and controlled drawdown. Candidate for extra guards or confluence filters.
- `too_few_trades`: Too few trades to judge. This overrides good-looking win rates, including 100% results from one trade.
- `infrastructure_error`: Result failed because of import/runtime/tooling/config issues. Fix infrastructure before judging signal quality.
- `rework_or_retire`: Weak across TP/SL profiles. Needs redesign or retirement.
- `retired`: Manually retired or archived. Not part of active testing.

## Status Values

- `active`: Still part of current testing.
- `watch`: Interesting but not yet trusted.
- `rework`: Needs strategy or indicator changes before the next serious run.
- `blocked`: Cannot be judged until infrastructure/data issues are fixed.
- `retired`: Removed from active testing.

## Review Rules

- Bucket the strategy once. Do not bucket each individual backtest row separately.
- Low trade count overrides good-looking results. If trade count is too low, use `too_few_trades`.
- Do not judge `infrastructure_error` rows as signal failures.
- Compare behavior across TP/SL profiles before assigning a final bucket.
- Use `last_reviewed_run` so later reviews can trace which evidence produced the current judgement.
- Keep notes short enough that the CSV stays readable.
- New strategies should start as `unreviewed` until manually classified.
- Retired strategies can be marked `retired` here and moved to an archive folder separately.

## Agent Guidance

- Do not overwrite manual bucket decisions automatically.
- When reviewing new results, propose registry updates first unless explicitly asked to edit the CSV.
- Preserve CSV validity. Quote fields that contain commas.
- Prefer root-cause fixes for `infrastructure_error` strategies rather than adding defensive workaround logic.
- If a run is contaminated by mid-run code or indicator changes, record that in `review_notes` and avoid promoting strategies from that run alone.
- Keep this registry summary-only. Do not embed full Freqtrade metrics, trades, orders, or hyperopt result objects here.
