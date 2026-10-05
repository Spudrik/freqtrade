---
doc_status: active
default_read: no
owner: user+agent
purpose: Evidence-backed handover for a related task planning the stage after the active Sieve3 V2 exit campaign.
do_not_use_for: Replacing live run ownership, silently changing strategies, or making final promotion decisions before the campaign completes.
snapshot_at: 2026-07-28T00:20:54+01:00
---

# Sieve3 V2 Results And Next-Step Planning Handover

## 1. Start Here

This document is for a related planning task. The original task still owns the live Sieve3 V2 batch run. Do not stop, restart, edit, or take ownership of that run unless the user explicitly asks.

Read only:

1. [Repo contract](../../AGENTS.md)
2. [Sieve3 V2 exit rules](../02_rules/rules_sieve3_exit_hyperopt.md)
3. [Exit/risk rules](../02_rules/rules_exit_and_risk_research.md)
4. [Hyperopt rules](../02_rules/rules_hyperopt_general.md)
5. [Sieve2 baseline lookup](../04_results/sieve3_sieve2_baseline_lookup.md) when comparing entry foundations

The older [recent results summary](../04_results/results_recent_summary.md) and [programme trace](../05_program_traceability/tracks/sieve_entry_exit.md) are historical checkpoints. Their `66`-batch count is stale. The current manifest is authoritative.

## 2. Current Campaign

The active surface contains `599` standalone `sieve3_V2_*.py` strategies from `134` fixed entry sources:

| Exit family | Planned files | Meaning |
|---|---:|---|
| `profit_level_full_or_ladder` | 134 | Full exit or compact arbitrary-profit ladder |
| `profit_ladder_three_stage_no_ratchet` | 134 | Two partials plus final exit; protective stop remains unchanged |
| `profit_ladder_three_stage_ratchet` | 134 | Two partials plus final exit; stop moves to entry, then target one |
| `entry_target_full_or_zone_reversal` | 100 | Entry-defined target with full or zone-reversal exit |
| `target_partial_invalidation_remainder` | 97 | Entry-defined target partial plus invalidation/remainder management |

Authoritative queue:

- [Current 32-batch manifest](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/v2/sieve3_v2_batches.json)
- `24` standard batches followed by `8` rare-pattern batches
- `599` queued references and `599` unique active files

Current result coverage at this snapshot:

- `370/599` unique strategies have a latest successful validation row (`61.8%`)
- Standard batches `001-020` are complete
- Standard batch `021/024` has `17/20` completed rows and is still running
- Standard batches `022-024` have not started
- Pattern batches `001-008` have not started
- `229` strategy results remain, including the three unfinished rows in batch `021`

Rare patterns are expected to have low trade counts. Do not classify low pattern trade count as a strategy defect.

## 3. Active Run Snapshot

Active job:

- Job: `20260727T201847_entry_sieve3_v2_refined_standard_batch_021`
- [Job configuration](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/jobs/20260727T201847_entry_sieve3_v2_refined_standard_batch_021.json)
- [Live status](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/status/20260727T201847_entry_sieve3_v2_refined_standard_batch_021.json)
- [Current JSONL](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/results/20260727T201847_entry_sieve3_v2_refined_standard_batch_021.jsonl)
- [Current summary](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/results/20260727T201847_entry_sieve3_v2_refined_standard_batch_021.summary.json)
- [Stdout log](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/logs/20260727T201847_entry_sieve3_v2_refined_standard_batch_021.stdout.log)
- [Stderr log](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/logs/20260727T201847_entry_sieve3_v2_refined_standard_batch_021.stderr.log)

Actual run contract:

| Setting | Value |
|---|---|
| Hyperopt space | `sell` |
| Sieve exit control | `control_entry_exits=false` |
| Loss | `MultiMetricHyperOptLoss` |
| Epochs | `300` |
| Seed | `42` |
| Training | `20200101-20221231` |
| Validation | `20240401-20260401` |
| Actual pairs | BTC, ETH, SOL perpetuals |
| Maximum cores | `18` |
| Current split while backlog exists | `17` Hyperopt workers plus `1` backtest lane |
| Pipeline | Split-venv enabled |

The job contains inherited `speed_pair_count=5` metadata, but `speed_run_mode=false`. Its `normal_run_pairs` and the executed commands use exactly BTC, ETH, and SOL.

At the snapshot, the active Hyperopt was:

`sieve3_V2_target_partial_invalidation_remainder_from_overtrade_supply_zone_breakout_long_vp_market_guard`

It was in a long epoch-straggler tail: most Joblib workers were waiting while a few expensive epochs continued to accumulate CPU. A stale status timestamp alone is not proof of failure. Check the exact process descendants and logs before intervening.

## 4. Raw Evidence Locations

| Evidence | Location |
|---|---|
| All current JSONL rows and summary JSON | [results](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/results/) |
| Freqtrade backtest ZIPs | [backtests](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/backtests/) |
| Hyperopt `.fthypt` files and per-run checkpoints | [hyperopt userdirs](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/hyperopt_userdirs/) |
| Archived selected parameter JSON | [params](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/params/) |
| Frozen per-run strategy/config snapshots | [runs](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/runs/) |
| Submitted job JSON | [jobs](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/jobs/) |
| Runtime statuses | [status](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/status/) |
| Runtime logs | [logs](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/logs/) |
| Current active pointer | [active.json](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/active.json) |
| Latest-result pointer | [latest.json](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/latest.json) |
| Completed-run state | [state.json](../../user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve_v2_speed_recovery/state.json) |
| Active strategy files | [strategies](../../user_data/strategies/) |

Do not manually load all raw files into chat. Use structured parsing:

1. Read strategy names from the current manifest.
2. Read JSONL files under the current runtime result directory.
3. Exclude rows whose strategy is not in the current manifest.
4. Group by strategy and select the latest successful row by `finished_at`.
5. Treat older errors and retries as history, not additional strategy evidence.

At this snapshot there are `385` manifest-matching raw rows but only `370` unique latest strategy rows. All `370` latest rows are successful. Historical errors in batches `005` and `011` were superseded by successful runs.

## 5. Runtime Incidents Already Resolved

1. Batch `005` initially resolved no active tunable entry parameters. The root cause was fixed and the complete batch reran successfully.
2. Batch `011` recorded one failed Hyperopt row, but the strategy later completed successfully.
3. Batches `017` and `018` exposed target levels equal to invalidation levels. This generated huge repeated callback errors. The target resolvers were corrected to exclude those collisions, and only the failed strategies were rerun. Their latest rows are clean.
4. Do not count failed code/config launches as strategy failures.

The active rule remains: a runtime failure blocks advancement; fix the root cause and rerun the same batch from completed results before moving forward.

## 6. Metric Meaning

- `profit_total_pct` is the percentage change in the configured validation wallet, not average profit per trade.
- `max_drawdown_pct * 100` is wallet drawdown in percentage points.
- `winrate` alone is not a quality verdict. Low-win-rate asymmetric strategies can be useful.
- Large drawdown is the clearest current weakness signal.
- Batch averages are analytically meaningless. Every row is a fixed entry plus one independently optimized exit solution.

## 7. Current Result Snapshot

These counts are descriptive only. Families do not all cover the same source set.

| Exit family | Rows | Profitable | DD `>10%` | DD `<=5%` |
|---|---:|---:|---:|---:|
| Target full/zone reversal | 59 | 31 | 6 | 44 |
| Target partial/invalidation | 53 | 29 | 3 | 38 |
| Profit full/ladder | 86 | 41 | 9 | 54 |
| Three-stage no ratchet | 86 | 35 | 8 | 54 |
| Three-stage ratchet | 86 | 45 | 7 | 62 |

Paired dominance means higher profit with no worse drawdown, or lower drawdown with no worse profit.

| Same locked entry comparison | A dominates | B dominates | Tradeoff |
|---|---:|---:|---:|
| Ratchet vs no ratchet | 43 | 15 | 28 |
| Target full vs target partial | 22 | 10 | 21 |
| Target full vs arbitrary profit full/ladder | 30 | 15 | 14 |
| Target partial vs arbitrary profit full/ladder | 23 | 15 | 15 |
| Target full vs ratcheted ladder | 22 | 20 | 17 |
| Target partial vs ratcheted ladder | 18 | 17 | 18 |

### 7.1 Critical attribution limitation

The ratchet and no-ratchet files Hyperopt independently. Their target levels, gaps, partial sizes, and hard stops often differ. Therefore `43 vs 15` means that **complete ratcheted solutions** were more often preferable; it does not isolate ratcheting as the cause.

Among the `86` paired sources:

- Only `11` selected exactly matching exit parameter vectors.
- Only `6` exact matches produced trades.
- Of those six, ratchet was better in `3`, no-ratchet in `1`, and `2` were tradeoffs/ties.
- `14` traded pairs shared the same target schedule while allowing partial sizes to differ: ratchet won `7`, no-ratchet won `1`, and `6` were tradeoffs.

Treat ratcheting as promising, not universally proven.

### 7.2 Entry-thesis interpretation

Do not explain results using long versus short direction. Ask:

> After this entry reaches its first favourable target, is a return toward entry normal price development or evidence that the entry thesis is failing?

Strong ratcheted solutions currently cluster around continuation/reclaim/breakout entries such as prior-high break plus BOS/retest, volume-supported breakout, prior-day-high break, BOS continuation, POC/midrange reclaim, and supply/resistance breakout. A plausible explanation is that target one confirms continuation, while a full return to entry signals failed follow-through.

Some weaker ratchet comparisons occur in range expansion, breakout-retest, rejection, liquidity/CHoCH, and shallow-target entries. Those entries may require normal retest room after the first partial. This is a hypothesis until post-partial trade paths are inspected.

### 7.3 Target and reversal observations

| Selected target action | Rows | Profitable | DD `>10%` |
|---|---:|---:|---:|
| `close_full` | 21 | 13 | 0 |
| `touch_full` | 2 | 1 | 0 |
| `zone_reversal` | 36 | 17 | 6 |
| `close_partial` | 24 | 15 | 0 |
| `touch_partial` | 13 | 7 | 1 |
| `reversal_1_partial` | 5 | 2 | 1 |
| `reversal_2_of_3_partial` | 11 | 5 | 1 |

Current evidence does not support a generic one-candle reversal. Two-of-three reversal remains unproven and has not yet beaten simply acting around a target close. Waiting for reversal can give back profit or expand drawdown, but event-level inspection is required before attributing cause.

### 7.4 Representative current leads

The table keeps one row per entry source, requires at least `10` trades and drawdown no greater than `5%`, then orders by profit. It is not a promotion list.

| Entry source | Exit family | Trades | Win rate | Profit | DD | PF |
|---|---|---:|---:|---:|---:|---:|
| `mtf_confluence_d1_vp_bos_4h_retest_short` | Ratchet ladder | 105 | 41.9% | +11.65% | 4.71% | 1.98 |
| `overtrade_bos_bull_continuation_long_1h_vp_market_guard` | Ratchet ladder | 40 | 52.5% | +10.97% | 1.82% | 3.17 |
| `mtf_std_daily_prior_high_breakout_long_1h` | Ratchet ladder | 46 | 52.2% | +9.15% | 3.15% | 1.85 |
| `multi2_tlv2_vp_res_break_vp_val_long_4h` | Profit full/ladder | 11 | 36.4% | +8.85% | 0.33% | 10.78 |
| `mtfx_d1_vp_poc_reclaim_long_4h_retest` | Target full/zone | 34 | 26.5% | +8.37% | 2.68% | 2.46 |
| `overtrade_mtf_h4_supply_breakout_long_1h_local_break_vp_market_guard` | Target full/zone | 96 | 21.9% | +7.85% | 3.82% | 1.41 |
| `novel_mtf_h4focus_prior_high_break_1h_retest_long` | Ratchet ladder | 95 | 16.8% | +6.90% | 3.43% | 1.55 |
| `novel_mtf_d1_midline_reject_1h_reject_short` | No-ratchet ladder | 13 | 46.2% | +6.80% | 1.89% | 4.50 |

The selected parameter checkpoint for every row is linked from its JSONL `params_file` field.

## 8. Correct Analysis Framework For The Branched Task

Analyse by entry thesis and expected post-entry path, not by batch or direction:

1. Identify what specifically caused entry.
2. Classify the mechanism: impulse continuation, breakout-retest, reclaim, rejection, sweep/CHoCH, range expansion/traverse, or rare pattern.
3. State what target one means for that entry: proof of thesis, ordinary noise, or merely the first obstacle.
4. State whether returning to entry is expected retesting or thesis failure.
5. Compare selected exit parameters before comparing outcomes.
6. Attribute performance to ratchet/full/partial/reversal only when other parameters converge or event-level trade evidence isolates the feature.
7. Otherwise describe each row as a complete alternative exit solution.
8. Collapse dominated exits within one entry source before presenting cross-entry leads.
9. Keep rare patterns separate.

## 9. Questions For The Next-Step Planning Task

The related task should plan, not implement, unless the user explicitly expands scope.

1. Which entry mechanisms need distinct exit architecture rather than one universal exit?
2. When does target one confirm enough of the entry thesis to move the stop to entry?
3. When should target interaction produce a full exit, a small partial, or remainder management?
4. What continuation/rejection evidence should select full versus partial without becoming generic confluence clutter?
5. Which exit families are genuinely distinct enough to survive into Sieve4?
6. Can Sieve4 combine retained target, invalidation, ladder, and stop-management reasons in parallel without recreating a huge hidden-gate search?
7. What event-level fields must be extracted from backtest ZIPs to identify exactly which trades changed because of ratcheting, partials, or delayed reversal?
8. Should architecture decisions wait for all `599` rows, or can clearly dominated ideas be provisionally parked while the campaign finishes?

Likely working hypothesis:

- Entry-defined targets and invalidations should remain first-class where coherent.
- Arbitrary staged ladders remain useful when the entry has no trustworthy target.
- Stop movement is an optional safety response tied to what target attainment means for that entry.
- Full versus partial should be selected from the entry thesis and current continuation/rejection evidence, not direction alone.
- Generic one-candle reversal should not be a default.

## 10. Worktree And Safety Warning

Last committed focused V2 checkpoint: `81b791882` (`feat(sieve3): checkpoint focused V2 exit research surface`).

The current working tree intentionally differs from that checkpoint:

- All `599` V2 strategy files are modified by the later standalone/callback-performance work and targeted runtime fixes.
- The batch manifest is modified by the current `32`-batch regrouping.
- Guidance also has unrelated existing changes.

Do not use `git checkout`, `git reset`, broad restore commands, or regenerate files mechanically. Do not assume the committed checkpoint is the current executable surface.

The branched planning task should be read-only unless the user explicitly approves named edits.
