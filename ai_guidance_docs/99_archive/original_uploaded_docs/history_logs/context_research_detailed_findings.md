# External Context FreqAI Research Findings

Last updated: 2026-05-27

This is the durable handoff ledger for external context/FreqAI research. Update it whenever an agent adds data, changes feature logic, runs FreqAI, validates timestamp alignment, or resolves a gap. The aim is to avoid repeating work after context compaction and to make clear when the project is ready for full confluence testing.

## Agent Update Rules

1. Add new FreqAI runs with timerange, model, rows, feature counts, headline metrics, and output paths.
2. Record data coverage separately from model results.
3. Keep orderbook, news/web, global context, GDELT/GKG, and confluence results separate until a deliberate combined test is run.
4. Move fixed flaws to the resolved log; do not silently remove them.
5. Treat direction accuracy, return correlation, and path metrics as research diagnostics, not trading profitability.
6. Do not move scraping/API fetching into Freqtrade strategy code.
7. Define the objective and pass/fail rule before running new research tests.
8. Pause relevant collectors and export parquet snapshots before tests that would otherwise touch live SQLite databases.

## Feature Design Priority

All new external context datasets should prioritize real-world behaviour logic: describe the market state or state transition a trader would care about, then encode it numerically.

Examples:

1. Orderbook: persistent sell walls should be treated as resistance zones, not just one-off prices. If those walls evaporate while bid pressure rises and spread expands, encode that as resistance removal / pressure shock / breakout confirmation. If support walls vanish while ask pressure rises, encode support removal / breakdown risk.
2. News/web/GDELT: treat first mentions, source confluence, severity, persistence, and market impact channel as separate facts. A single article count is weaker than "major macro topic first mentioned by several source families and still expanding after 24h."
3. Global/macro: encode surprise and persistence, not just current values. A rate/liquidity shock confirmed by DXY/equities/gold is different from a normal daily value.
4. Cross-source confluence: prefer compact features that describe agreement across families, such as "macro stress news + DXY move + crypto liquidity thinning", rather than adding all raw fields and hoping the model infers the story.

Avoid hardcoded opinion scoring as the main method. The dataset should state observable facts and behavioural states; direct tests and FreqAI decide whether they predict returns, drawdowns, breakouts, or risk avoidance.

## Current Pipeline State

| Area | Status | Store | Notes |
|---|---|---|---|
| News SQL | Paused for orderbook research | `C:\FreqTradeStuff\user_data\research_news_data` and context feature DB | Stop file written 2026-05-25. Converted into numeric 1h context features. Needs stronger topic/severity validation. |
| Web SQL | Paused for orderbook research | `C:\FreqTradeStuff\user_data\research_news_data` and context feature DB | Stop file written 2026-05-25. Low row count compared with news/GDELT. |
| Global context | Paused for orderbook research | `C:\FreqTradeStuff\user_data\research_news_data\context_features\context_features.sqlite` | Stop file written 2026-05-25. Needs source-by-source coverage table. |
| GDELT events | Paused for orderbook research | `context_features.sqlite` | Active backfill/check processes were stopped for the 2026-05-25 orderbook research pass. Historical gaps remain. |
| GKG | Paused for orderbook research | `context_features.sqlite` | Active backfill processes were force-stopped because no stop-file mechanism was available. Rows exist, but latest build loaded zero GKG rows for the current overlap window. |
| Orderbook live | Paused for orderbook research | `C:\FreqTradeStuff\user_data\orderbook_data\live\orderbook_events.sqlite` | Stop file written 2026-05-25. The live collector ignored the stop file and was force-stopped. Raw storage is large. |
| Bybit historical orderbook | Trial implemented | Same orderbook DB via historical importer | Free OB200 ZIPs imported at 10-second sampling for selected windows, then compacted to 1h. |
| FreqAI research strategy | Active | `C:\FreqTradeStuff\user_data\strategies\ContextFreqAIResearchStrategy.py` | Research-only strategy; existing trading strategies are not changed. |

## Latest Feature Store Snapshot

Context feature DB latest observed state from `run_state.json` on 2026-05-23:

| Field | Value |
|---|---:|
| Feature rows | `56040` |
| First feature hour | `2020-01-01T00:00:00+00:00` |
| Last feature hour | `2026-05-23T23:00:00+00:00` |
| Feature columns | `398` |
| Numeric export columns | `403` |
| Latest source rows found, news | `13069` |
| Latest source rows found, web | `428` |
| Latest source rows found, global | `9225` |
| Latest source rows found, GDELT | `54913` |
| Latest source rows found, GKG | `20315` |
| Latest overlap rows loaded, GDELT | `625` |
| Latest overlap rows loaded, GKG | `0` |

Orderbook feature snapshot after historical Bybit import and compaction:

| Field | Value |
|---|---:|
| Source rows found | `2660512` |
| Source rows loaded in rebuild | `2659008` |
| 1h feature rows written | `9308` |
| Feature column count | `485` |
| Latest compact export | `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_20260523_232420.parquet` |

## FreqAI Result Ledger

### Bybit Historical Orderbook Targeted Test

Purpose: test whether Bybit orderbook features add incremental signal over price/volume on a quiet control window and an event/stress window.

Data:

| Window | Dates | Prediction rows | Price path | Validation |
|---|---|---:|---|---|
| Quiet control | `2025-07-25` to `2025-08-11` | `408` | `+1.38%`, max drawdown `-6.10%`, max upside `+6.29%` | `0` lookahead, `0` tick mismatches |
| Event/stress | `2026-01-29` to `2026-02-16` | `432` | `-22.78%`, max drawdown `-29.43%`, max upside `+13.68%` | `0` lookahead, `0` tick mismatches |

Model setup:

| Model | Train/backtest | FreqAI model | Feature count | Train rows/window |
|---|---|---|---:|---:|
| Price-only baseline | `14d / 1d` | `SKLearnRidgeRegressorMultiTarget` | `76-80` | `198` |
| Price + Bybit OB | `14d / 1d` | `SKLearnRidgeRegressorMultiTarget` | `141-157` | `198` |

Headline results:

| Window | Target | Price corr | Bybit corr | Price dir acc | Bybit dir acc | Read |
|---|---|---:|---:|---:|---:|---|
| Quiet | `future_return_24h` | `-0.0548` | `-0.0149` | `55.39%` | `56.62%` | Slight improvement, weak. |
| Quiet | `future_max_upside_72h` | `-0.2178` | `-0.1917` | `78.68%` | `87.75%` | Better path direction, but quiet-window class balance may inflate accuracy. |
| Quiet | `shock_down_72h` | `-0.0976` | `-0.0938` | `81.62%` | `91.18%` | Accuracy likely helped by few shock labels; not strong proof. |
| Event | `future_return_1h` | `-0.0064` | `0.0430` | `49.77%` | `51.62%` | Small improvement. |
| Event | `future_return_24h` | `-0.3214` | `-0.2035` | `40.28%` | `39.35%` | Correlation less bad, direction not improved. |
| Event | `future_max_upside_72h` | `-0.2108` | `-0.1317` | `50.46%` | `56.25%` | Best event-window uplift, still weak. |
| Event | `future_max_drawdown_72h` | `-0.6110` | `-0.5582` | `65.28%` | `65.97%` | Slight uplift; dominated by crash regime. |
| Event | `shock_down_72h` | `-0.3209` | `-0.2875` | `28.47%` | `30.32%` | Still poor classification. |

Artifacts:

| Artifact | Path |
|---|---|
| Final comparison CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\bybit_targeted_final_comparison_20260523.csv` |
| Quiet price run | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\freqai_context_20260523_232648` |
| Quiet Bybit run | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\freqai_context_20260523_232852` |
| Event price run | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\freqai_context_20260523_233122` |
| Event Bybit run | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\freqai_context_20260523_233319` |

Conclusion: Bybit orderbook features show weak incremental signal in event/stress conditions, mainly in 1h return and 72h path metrics. This is not yet reliable. More historical windows and better labels are required before confluence testing.

### Orderbook Everything / Wall-Focused Test

Purpose: test whether broader wall, depth, pressure, imbalance, spread/fragility, and confluence features clarify the relationship between visible book structure and future BTC movement.

Direct bucket/event study:

| Item | Value |
|---|---:|
| Feature rows used | Compact orderbook 1h parquet export |
| Windows | Quiet control `2025-07-25` to `2025-08-11`; event/stress `2026-01-29` to `2026-02-16` |
| Feature families | ask walls, bid walls, wall delta, pressure, imbalance, liquidity/depth, spread/fragility, confluence |
| Compact features tested | `248` |
| Bucket report CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_everything_bucket_report_20260524.csv` |
| Bucket report text | `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_everything_bucket_report_20260524.txt` |

Bucket-study observations:

| Window | Stronger observed effects | Read |
|---|---|---|
| Quiet control | High total top-20 depth and low depth-thinness buckets had worse forward returns: 24h delta around `-1.75%`, 72h delta around `-2.71%`. | High visible depth did not imply bullish support in this quiet sample. |
| Quiet control | Close ask/bid wall distance buckets leaned negative over 24h/72h. | Proximity to walls may matter more than raw wall size. |
| Event/stress | High bid liquidity within 5 bps, high bid wall score/notional, high ask wall notional, and pressure flip counts clustered before worse 24h/72h outcomes. | During stress, big/near liquidity may mark distribution or liquidity that gets consumed rather than protection. |
| Event/stress | Low spread buckets showed negative forward returns in this sample. | In this crash window, tight spreads did not mean low risk; the market could still move hard after apparently orderly book states. |

FreqAI everything comparison:

| Window | Model | Feature count | Train rows/window | Prediction rows |
|---|---|---:|---:|---:|
| Quiet | Price-only | `78-80` | `198` | `408` |
| Quiet | Headline OB | `141-157` | `198` | `408` |
| Quiet | Full OB everything | `444-479` | `198` | `408` |
| Event | Price-only | `76-80` | `198` | `432` |
| Event | Headline OB | `141-153` | `198` | `432` |
| Event | Full OB everything | `445-475` | `198` | `432` |

Headline FreqAI results:

| Window | Target | Price corr | Headline corr | Full corr | Price dir acc | Headline dir acc | Full dir acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Quiet | `future_return_24h` | `-0.0548` | `-0.0149` | `-0.0080` | `55.39%` | `56.62%` | `56.86%` |
| Quiet | `future_max_upside_72h` | `-0.2178` | `-0.1917` | `-0.2053` | `78.68%` | `87.75%` | `87.25%` |
| Quiet | `shock_down_72h` | `-0.0976` | `-0.0938` | `-0.0424` | `81.62%` | `91.18%` | `95.34%` |
| Event | `future_return_6h` | `-0.0740` | `-0.0029` | `0.0860` | `45.14%` | `44.44%` | `48.84%` |
| Event | `future_return_24h` | `-0.3214` | `-0.2035` | `0.0811` | `40.28%` | `39.35%` | `38.89%` |
| Event | `future_max_upside_72h` | `-0.2108` | `-0.1317` | `0.0765` | `50.46%` | `56.25%` | `66.90%` |
| Event | `future_max_drawdown_72h` | `-0.6110` | `-0.5582` | `-0.0874` | `65.28%` | `65.97%` | `58.33%` |
| Event | `shock_down_72h` | `-0.3209` | `-0.2875` | `-0.1185` | `28.47%` | `30.32%` | `35.88%` |

Artifacts:

| Artifact | Path |
|---|---|
| Full comparison CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_everything_freqai_comparison_20260524.csv` |
| Quiet full OB run | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\freqai_context_20260524_003208` |
| Event full OB run | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\freqai_context_20260524_003402` |

Conclusion: the "everything" orderbook run improved several event-window correlations, especially `future_return_6h`, `future_return_24h`, `future_max_upside_72h`, and reduced the strongly wrong drawdown correlation. However, directional accuracy is mixed, and the model has far more features than training rows. Treat this as evidence that richer wall/depth/fragility features are worth refining, not as a reliable model.

Implementation note: the first full-feature FreqAI attempt failed because the live orderbook SQLite database was locked by collection/writes. A research-only parquet compact feature loader class was added so backtests can consume the exported 1h feature cache without contending with the live writer.

### Focused Orderbook Hypothesis Tests

Purpose: test pre-registered orderbook hypotheses directly before another FreqAI pass, using compact BTC 1h parquet features and explicit random/shuffled controls.

Run details:

| Field | Value |
|---|---:|
| Run date | `2026-05-24` |
| Pair | `BTC/USDT` |
| Windows | `quiet_control`, `volatile_control`, `event_stress` |
| Fresh compact export | `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_20260524_100947.parquet` |
| Compaction mode | `update`, `overlap-days=3` |
| Source rows found | `2660512` |
| Source rows loaded | `14348` |
| Feature rows written | `73` |
| Feature column count | `1523` |
| Test report rows | `252` |
| Candidate rows | `84` |
| Candidate passes | `52` |

Artifacts:

| Artifact | Path |
|---|---|
| Focused test report CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_focused_tests_report.csv` |
| Candidate summary CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_focused_tests_candidates.csv` |
| Summary JSON | `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_focused_tests_summary.json` |

Headline read:

1. The strongest stable candidates were Bybit spot wall-distance/proximity and spread-fragility features.
2. Top-decile Bybit spot wall-distance/proximity rows had coherent effects across all three windows for `future_return_24h`, `future_max_upside_72h`, and `future_max_drawdown_72h`.
3. Bybit spot spread p95/last also passed across all three windows, especially on 24h return and 72h path targets.
4. Pressure flip and liquidity-vacuum features produced some passing candidates, but with weaker and less clean interpretation.
5. Several wall-proximity columns produced identical-looking results, so the next refinement should collapse aliases/correlated duplicates before FreqAI modelling.

Conclusion: this is the strongest orderbook evidence so far, but it is direct bucket evidence rather than a full model result. Treat wall proximity and spread fragility as the next FreqAI feature focus, with duplicate reduction before modelling.

### Refined Orderbook FreqAI Runs

Purpose: test whether the direct bucket signal survives a smaller, model-ready FreqAI feature profile.

Implementation:

1. Fixed the parquet orderbook loader to filter `canonical_pair` before merging with BTC OHLCV, preventing duplicate multi-pair rows from being treated as BTC rows.
2. Added a `bybit_refined` compact profile to the research strategy.
3. Added historical rolling deltas and z-scores for selected wall, spread, liquidity, depth, imbalance, and pressure features in orderbook compaction.
4. Rebuilt BTC compact orderbook features with `full-rebuild`, then copied the output to the stable research cache path.

Refined feature/data artifacts:

| Artifact | Path |
|---|---|
| Full rebuilt BTC orderbook parquet | `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_20260524_104034.parquet` |
| Stable refined cache used by strategy | `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_refined_latest.parquet` |
| Refined feature manifest | `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_refined_feature_manifest_20260524.csv` |
| Raw-refined comparison CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_refined_freqai_comparison_20260524.csv` |
| Tail-aware refined comparison CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_refined_tail_freqai_comparison_20260524.csv` |

Tail-aware refined run setup:

| Window | Timerange | Price features | Refined features | Prediction rows |
|---|---|---:|---:|---:|
| Quiet | `20250725-20250811` | `78` | `148` | `408` |
| Volatile control | `20251015-20251031` | `80` | `150` | `384` |
| Event/stress | `20260129-20260216` | `78` | `148` | `432` |

Tail-aware refined headline results versus price-only:

| Window | Target | Price corr | Refined corr | Delta | Price dir acc | Refined dir acc | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| Event | `future_return_24h` | `-0.3214` | `-0.1034` | `+0.2181` | `40.28%` | `42.13%` | Better correlation, still weak direction. |
| Event | `future_max_drawdown_72h` | `-0.6110` | `-0.3234` | `+0.2876` | `65.28%` | `61.34%` | Much less wrong correlation, lower direction. |
| Event | `future_max_upside_72h` | `-0.2108` | `-0.0490` | `+0.1618` | `50.46%` | `49.31%` | Better correlation, direction flat. |
| Event | `shock_down_72h` | `-0.3209` | `-0.1225` | `+0.1984` | `28.47%` | `29.63%` | Improved but still poor classifier. |
| Quiet | `future_return_24h` | `-0.0548` | `0.0403` | `+0.0951` | `55.39%` | `59.31%` | Useful uplift in quiet window. |
| Quiet | `shock_down_72h` | `-0.0976` | `-0.0354` | `+0.0621` | `81.62%` | `90.44%` | Accuracy improved, but class balance likely helps. |
| Volatile control | `future_return_24h` | `0.1672` | `-0.0471` | `-0.2144` | `52.34%` | `52.86%` | Degraded correlation in this control window. |
| Volatile control | `future_max_drawdown_72h` | `0.3988` | `0.1755` | `-0.2233` | `90.36%` | `90.36%` | Degraded correlation; direction unchanged. |

Conclusion: the tail-aware refined profile is materially better than price-only in the event/stress window and improves some quiet-window targets, but it fails the stability requirement because the volatile control window degrades. The direct bucket signal is real enough to keep, but Ridge on these features is not yet a robust general model. Next modelling step should be explicit event-state features or tree-based models that can use threshold behaviour, while keeping the feature set compact.

### LightGBM Tree Refined Orderbook Runs

Purpose: test whether a tree model can use threshold-style orderbook behaviour better than Ridge, especially persistent walls, wall/spread fragility, sudden pressure, and risk-avoidance states.

Setup:

| Field | Value |
|---|---:|
| Model | `LightGBMRegressorMultiTarget` |
| Feature profile | `bybit_refined` tail-aware orderbook profile |
| Price-only features | `78-80` |
| Refined tree features | `148-150` |
| Windows | Quiet, volatile control, event/stress |
| Comparison CSV | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_tree_lgbm_freqai_comparison_20260524.csv` |

Headline results versus price-only LightGBM:

| Window | Target | Price tree corr | Refined tree corr | Delta | Price dir acc | Refined dir acc | Read |
|---|---|---:|---:|---:|---:|---:|---|
| Event | `shock_down_72h` | `-0.4457` | `-0.3737` | `+0.0720` | `29.40%` | `34.95%` | Risk signal improved but still weak. |
| Event | `future_max_drawdown_72h` | `-0.4580` | `-0.3890` | `+0.0690` | `92.59%` | `94.91%` | Direction improved, correlation still wrong-signed. |
| Event | `future_return_24h` | `-0.1793` | `-0.1522` | `+0.0271` | `43.75%` | `48.38%` | Modest improvement. |
| Quiet | `future_return_6h` | `-0.1108` | `-0.0723` | `+0.0385` | `48.77%` | `53.68%` | Small useful uplift. |
| Quiet | `future_return_24h` | `-0.0067` | `0.0111` | `+0.0178` | `57.60%` | `57.84%` | Mild positive alignment. |
| Volatile control | `future_return_24h` | `-0.1017` | `-0.2243` | `-0.1226` | `52.34%` | `40.89%` | Degraded badly. |
| Volatile control | `future_return_6h` | `-0.0741` | `-0.1381` | `-0.0639` | `48.18%` | `43.49%` | Degraded. |

Conclusion: LightGBM did not unlock the orderbook signal as a general trading model. It slightly improved some risk/stress targets, but not enough, and it degraded the volatile control window. This supports the hypothesis that the useful orderbook signal is conditional and event-state based, not a broad continuous predictor. Next step should be explicit compact state features such as persistent wall, wall evaporation, spread shock, pressure shock, and wall/spread/volume confluence before rerunning models.

### Orderbook Behaviour Dataset Setup

Purpose: reframe orderbook data around how a trader would analyse behaviour: persistent zones, wall removal, wall movement, pressure shocks, liquidity vacuums, spread shocks, and rebuild/acceptance states.

Status: dataset and FreqAI run setup are prepared, but FreqAI has not been run on this behaviour dataset yet.

Implementation:

1. Added behaviour features to orderbook compaction after hourly raw/rolling features are built.
2. Behaviour features use an active-data gate based on sampled book rows and valid book ratio, not the strict live 1-second `ready` threshold, so historical 10-second Bybit samples remain usable.
3. Added the `bybit_behaviour` FreqAI strategy profile.
4. Added setup-only run manifest generation in `orderbook_behaviour_freqai_setup.py`.
5. Added 3h return/path reporting support for orderbook research reports.

Behaviour feature examples:

| Feature type | Example columns | Intended read |
|---|---|---|
| Wall evaporation | `ob1h_bybit_spot_beh_ask_wall_evaporation_1h`, `ob1h_bybit_spot_beh_bid_wall_evaporation_1h` | Resistance/support disappearing quickly. |
| Wall movement | `ob1h_bybit_spot_beh_ask_wall_closer_1h`, `ob1h_bybit_spot_beh_bid_wall_closer_1h` | Resistance/support moving closer to current price. |
| Pressure shock | `ob1h_bybit_spot_beh_bullish_pressure_shift_1h`, `ob1h_bybit_spot_beh_bearish_pressure_shift_1h` | Sudden shift in book pressure. |
| Liquidity vacuum | `ob1h_bybit_spot_beh_liquidity_vacuum_up`, `ob1h_bybit_spot_beh_liquidity_vacuum_down` | Thin path above/below price. |
| Persistent zones | `ob1h_bybit_spot_beh_persistent_ask_zone_24h`, `ob1h_bybit_spot_beh_persistent_bid_zone_24h` | Long-lived resistance/support zones. |
| Confirmation/rebuild | `ob1h_bybit_spot_beh_resistance_removed_score`, `ob1h_bybit_spot_beh_post_breakout_support_rebuild` | Resistance removal and support rebuilding after a move. |
| Risk context | `ob1h_bybit_spot_beh_fragile_book_score`, `ob1h_bybit_spot_beh_false_breakout_risk` | Fragile execution/risk state and possible rejection risk. |

Dataset artifacts:

| Artifact | Path |
|---|---|
| Behaviour parquet cache | `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_behaviour_latest.parquet` |
| Source rebuild export | `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_20260524_191433.parquet` |
| FreqAI setup manifest | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_behaviour_setup_20260524_190449\orderbook_behaviour_freqai_manifest.json` |
| FreqAI run script | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_behaviour_setup_20260524_190449\run_orderbook_behaviour_freqai.ps1` |

Dataset sanity check:

| Field | Value |
|---|---:|
| Rows | `9330` |
| Total columns | `1684` |
| Behaviour columns | `100` |
| Bybit spot behaviour columns | `25` |
| FreqAI setup commands prepared | `9` |

Next step: run the prepared FreqAI commands only after confirming the objective/pass criteria for this behaviour dataset. The initial comparison should be price-tree versus behaviour-tree, with behaviour-ridge as a simpler baseline. Primary targets should be `1h`, `3h`, `6h`, and `24h`, with 72h treated as secondary for persistent zone behaviour only.

### News / Web / Global / GDELT / GKG Context

Current read:

1. The numeric context feature builder works and maintains a long 1h feature table from 2020 onward.
2. Prior context experiments did not produce convincing stable predictive results from simple counts/rolling features alone.
3. The likely missing piece is better event characterization: topic, severity, relevance, persistence, source confluence, and relation to price/volume regime.
4. GDELT and GKG are promising broad-coverage sources, but data gaps and sparse loaded overlap windows need to be resolved before conclusions are trustworthy.

Known limitation: this section needs a precise run ledger reconstructed from older `freqai_context_*` folders.

Suggested table format for future updates:

| Run date | Sources | Timerange | Model | Rows | Key metrics | Verdict | Output path |
|---|---|---|---|---:|---|---|---|
| TODO | News/Web/Global/GDELT/GKG | TODO | TODO | TODO | TODO | TODO | TODO |

## Data Gaps And Open Flaws

| ID | Area | Gap/flaw | Impact | Status |
|---|---|---|---|---|
| G1 | GKG | Latest overlap build found `20315` rows but loaded `0`. | GKG may not be contributing to current features. | Open |
| G2 | GDELT/GKG | Historical coverage gaps are not fully mapped here. | Cannot trust full-period confluence tests until coverage is audited. | Open |
| G3 | News/Web | Topic/severity scoring is still underdeveloped relative to known market-moving events. | Weak count features may miss the actual signal. | Open |
| G4 | Global context | Need source-by-source coverage and timestamp safety summary. | Unknown whether each global feature is usable for historical tests. | Open |
| G5 | Orderbook | Historical Bybit importer samples every 10 seconds for trial windows. | Good enough for hourly trials, not equivalent to live 1-second collector. | Open |
| G6 | Orderbook | Full compact feature table includes long empty gaps between imported historical windows. | Exact merge prevents stale fill, but FreqAI windows must include training data with actual coverage. | Open |
| G7 | Modelling | Current runs use Ridge with small training windows and limited rows. | Metrics are noisy and should not be over-interpreted. | Open |
| G8 | Confluence | No deliberate all-source confluence run yet. | Cannot assess combined signal until each family is individually stable enough. | Open |
| G9 | Storage | Live orderbook DB is very large. | Need retention/compaction policy before long-term unattended collection. | Open |
| G10 | Orderbook modelling | Full orderbook FreqAI used `444-479` features with only `198` training rows per fold. | High overfit risk; next pass should compress/select wall/depth event features before modelling. | Open |
| G11 | Orderbook coverage | Raw Bybit trader-state dataset only has four monthly windows with enough coverage for strict direct tests. | Positive results are promising but not general enough yet. | Open |
| G12 | Orderbook modelling | Trader-state Ridge improved correlation but produced poorly calibrated prediction magnitudes, especially for event/binary labels. | Use it as ranking evidence only; add scaled/narrow linear models or classifier/event models before treating event predictions seriously. | Open |
| G13 | Orderbook semantics | `trader_state_breakout_failure` passed strict tests with inverted practical meaning. | Rename/refine into "failure already resolved", "resistance cleared", and "pre-failure risk" sub-states before confluence use. | Open |
| G14 | Orderbook event modelling | Pre-event failure-risk states did not pass strict direct tests; resolved rejection/bounce states did. | Next tests should be event-window/filter tests around level touches, not broad every-hour prediction. | Open |

## Resolved / Validated Items

| Date | Item | Evidence |
|---|---|---|
| 2026-05-23 | Bybit historical OB200 ZIPs can be imported and compacted into the existing orderbook DB schema. | `historical_bybit_import.py`, target windows imported with no missing days. |
| 2026-05-23 | Targeted Bybit windows have no feature lookahead and no raw tick mismatches. | Quiet/event validation checks returned `0` lookahead and `0` mismatches. |
| 2026-05-23 | FreqAI can run price-only and price+Bybit OB comparisons. | Four corrected runs completed successfully. |
| 2026-05-24 | Full compact orderbook features can be tested without live SQLite lock contention. | `ContextOrderbookOnlyParquetFreqAIResearchStrategy` completed quiet/event full OB runs from parquet cache. |
| 2026-05-24 | Orderbook behaviour direct tests found a focused candidate family. | `zone_compression_score` passed pre-registered direct-test checks across quiet, volatile, and event windows; report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_behaviour_analysis_20260524.md`. |
| 2026-05-25 | Raw Bybit orderbook archive stale-gap leakage was fixed. | `orderbook_trader_state_features.py` now resets sample timing and book state at archive boundaries. Rebuilt parquet has `0` source-after-feature violations and real missing-archive gaps remain visible. |
| 2026-05-25 | Direct state tests now preserve sparse archive coverage. | `orderbook_state_experiments.py` builds a continuous OHLCV base, exact-merges sparse features, and filters tests with `feature_present` so missing archive gaps are not treated as consecutive future hours. |
| 2026-05-25 | Raw-L2 trader-state feature family produced the strongest orderbook evidence so far. | Direct tests produced `2` strict passes across four eligible monthly windows; FreqAI trader-state Ridge improved 24h return correlation by `+0.2136` versus price-only across `3/3` windows. |
| 2026-05-25 | Refined raw-L2 tree models improved failure-event prediction over price-only event controls. | Refined LightGBM improved `breakout_failure_next_6h` AUC by mean `+0.0547` and `breakdown_failure_next_6h` AUC by mean `+0.0520`, both across `3/3` windows. |

## Next Research Steps

1. Build event-window tests centered on resistance/support touches, because refined orderbook states look useful around level interactions rather than every hour.
2. Train classifier-style models for `breakout_failure_next_6h` and `breakdown_failure_next_6h`; regression multi-target output is useful for research but not ideal for event labels.
3. Test refined failure/bounce states as risk filters on existing strategy entries near support/resistance, without changing entry logic yet.
4. Add more historical Bybit raw archive windows before claiming generality. Four eligible windows are not enough for production confidence.
5. Use scaled/narrow linear models for continuous targets and classifier/event models for trader-event labels; do not over-read uncalibrated Ridge event-label magnitudes.
6. Build a source coverage report that outputs per-source first timestamp, last timestamp, row count, usable timestamp field, missing periods, and feature contribution.
7. Reconstruct a clean ledger for prior news/global/web/GDELT FreqAI runs from existing run folders.
8. Fix GKG loading gap or explicitly mark GKG unavailable for current FreqAI tests.
9. Add event-severity/topic/confluence features before rerunning news/GDELT tests.
10. Only after individual families show coherent behaviour, run staged confluence:
   - price-only
   - price + global
   - price + news/web
   - price + GDELT/GKG
   - price + orderbook
   - price + best families
   - full confluence
11. Keep full confluence runs labelled as research only until data coverage and timing are audited.

## Pre-Registered Focused Orderbook Tests

The focused test harness was executed on 2026-05-24 after exporting a fresh compact parquet snapshot. It is parquet-only by design and refuses to run unless called with both `--execute` and `--snapshot-confirmed`.

Setup review fixes made on 2026-05-24:

1. Added a third window, `volatile_control`, so a candidate cannot pass from only quiet-plus-crash behaviour.
2. Restricted feature selection to compact pre-registered column patterns instead of broad family-wide matching.
3. Added machine-checkable candidate summaries with same-sign, random-control, bootstrap, and shuffled-label checks.
4. Kept the harness snapshot-only: no live SQLite reads during research tests.
5. Kept execution gated until collectors have been paused and the exported parquet/feather snapshot paths are confirmed.
6. Added a trader-behaviour direct test harness for wall evaporation, pressure impulse, persistent zones, fragile books, and post-move wall rebuild/acceptance.
7. Added short-horizon event labels for `shock_down_6h`, `shock_down_24h`, `breakout_up_6h`, `breakout_up_24h`, and `breakout_up_72h`, while keeping the longer `shock_down_72h`, `breakout_up_720h`, and `regime_up_2160h` labels.
8. Added a setup-only FreqAI comparison manifest covering price-only controls, behaviour Ridge, behaviour LightGBM, and behaviour XGBoost across quiet, volatile-control, and event-stress windows.
9. Added matching 3h/6h/24h/72h target horizons to the price-only controls used in the behaviour comparison so price-only and behaviour models are evaluated on the same labels.
10. Stopped orderbook-only and price-only research strategies from loading context SQL during feature engineering; these setup paths now stay on price OHLCV plus parquet orderbook features.
11. Added `shock_down_3h` and `breakout_up_3h` labels and 1h/3h prediction bucket reports for short event-style checks.

Latest setup-only artifacts:

| Artifact | Path |
|---|---|
| Behaviour FreqAI manifest | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_behaviour_setup_20260524_192512\orderbook_behaviour_freqai_manifest.json` |
| Behaviour FreqAI run script | `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_behaviour_setup_20260524_192512\run_orderbook_behaviour_freqai.ps1` |
| Behaviour direct-test harness | `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\orderbook_behaviour_direct_tests.py` |

No FreqAI/orderbook behaviour tests were executed during this setup pass.

Execution update on 2026-05-24:

1. Direct parquet behaviour tests executed successfully.
2. Ridge and LightGBM FreqAI comparisons executed for quiet, volatile, and event windows.
3. Behaviour XGBoost quiet was stopped because it ran too long on a small fold; XGBoost is deferred until the feature profile is narrower.
4. Broad FreqAI lift was weak/inconsistent, but direct tests showed `zone_compression_score` as a useful focused candidate.
5. A narrow zone-compression FreqAI profile was then executed across the same quiet, volatile, and event windows.
6. Focused zone-compression results improved some 72h upside/breakout targets versus OHLCV-only, especially Ridge `future_max_upside_72h` and `breakout_up_72h`, but direction accuracy is still mostly flat.
7. Current next step is more windows plus direct tests for zone compression combined with price location and volume pressure, not a bigger model.

State experiment update on 2026-05-24:

1. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\orderbook_state_experiments.py`.
2. Ran 15 trader-style state experiments across 13 monthly windows from 2025-05 to 2026-05.
3. Results: `719` result rows, `61` candidate rows, `6` passing candidates.
4. Strongest surviving state: `compressed_near_resistance` reduced `breakout_up_72h` probability by a mean `-0.0773` across 13 windows, with 11 negative-effect windows.
5. Main interpretation changed from "zone compression may help breakout detection" to "compressed book near resistance is more useful as an upside breakout/rejection risk filter."
6. Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_state_experiments_20260524.md`.

Gap-fill update on 2026-05-25:

1. Extended `orderbook_state_experiments.py` with trader-style gap-fill hypotheses for wall chasing/migration, absorption, pressure exhaustion, spoof/flicker instability, post-breakout support rebuild, and post-breakdown resistance rebuild.
2. Ran the v2 direct state pass across 13 monthly windows from 2025-05 to 2026-05.
3. Results: `1070` window/target rows, `104` candidate rows, `7` strict passes.
4. The strongest result repeated the prior conclusion: `compressed_near_resistance` reduced `breakout_up_72h` probability by mean `-0.0773` across 13 windows.
5. New surviving result: `post_breakout_support_rebuild_acceptance` also reduced `breakout_up_72h` by mean `-0.0730`, meaning the current rebuild proxy is behaving like failed upside acceptance / rejection, not bullish continuation.
6. Added `ContextBybitOrderbookRejectionStateParquetFreqAIResearchStrategy`, a compact FreqAI profile built from the surviving rejection-state logic.
7. Ran price-only vs rejection-state Ridge and LightGBM across quiet, volatile, and event windows.
8. Best FreqAI lift was small: Ridge `breakout_up_72h` mean correlation delta `+0.0338` across 3 windows, with 2 positive-correlation-delta windows and flat direction accuracy.
9. LightGBM did not unlock a strong threshold edge; best compact lift was `future_return_24h` mean correlation delta `+0.0288`, again small.
10. Current verdict: orderbook context is most useful as a breakout skepticism / risk-filter lead, not yet as a standalone entry model.
11. Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_gap_fill_rejection_state_20260525.md`.

Objective List Alpha update on 2026-05-25:

1. The orderbook feature pipeline now has an Objective List Alpha layer in `C:\FreqTradeStuff\user_data\Custom_Launcher\orderbook\feature_compaction.py`.
2. Alpha features encode trader-style state: persistent support/resistance zones, zone added/removed strength, migration, stability, price touches/acceptance/rejection, absorption, breakout/breakdown failure, directional liquidity vacuum, pressure agreement/divergence, and market structure context.
3. Generated Alpha snapshot: `C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_alpha_latest.parquet`.
4. Snapshot status: `9337` rows, `2029` columns, `189` Alpha columns, `0` source-after-feature timestamp violations, `0` duplicate columns, `0` Alpha infinite cells, and `0` all-null Alpha columns.
5. Direct monthly Alpha tests were run with corrected price alignment so orderbook feature hour `T` is compared to the completed candle ending at `T`.
6. Direct test outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_state_experiments_alpha_report.csv` and `orderbook_state_experiments_alpha_candidates.csv`.
7. Direct results: `1709` result rows, `146` candidates, `7` strict passes, but `0` Alpha-specific strict passes. The old compression/rejection family still dominates direct bucket tests.
8. Added `ContextBybitOrderbookObjectiveAlphaParquetFreqAIResearchStrategy` and Alpha Ridge/LightGBM setup entries.
9. FreqAI comparison outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_objective_alpha_freqai_comparison_summary.csv`.
10. Best FreqAI result: Alpha Ridge improved `future_return_24h` versus price-only by mean correlation delta `+0.0843` across quiet/volatile/event windows, with positive delta in `2/3` windows and mean direction-accuracy delta `+0.0563`.
11. Other useful Alpha FreqAI lifts: Ridge `shock_down_24h` mean corr delta `+0.0566`, Ridge `future_max_drawdown_24h` `+0.0553`, and LightGBM `shock_down_24h` `+0.0355`.
12. Alpha did not improve `breakout_up_72h`; price-only remained better there.
13. Verdict: Objective List Alpha is a better representation layer and is worth keeping, but direct state evidence is still sparse. It should be refined with 1m/5m wall-zone identity tracking before calling acceptance/failure states robust.
14. Full report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_objective_alpha_20260525.md`.

Raw Bybit trader-state update on 2026-05-25:

1. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\orderbook_trader_state_features.py`.
2. Purpose: build raw-L2 trader-state features directly from Bybit OB200 archive ZIPs instead of relying only on hourly compact summaries.
3. Feature design follows Objective List Alpha but uses 1m samples and 5m zone identities before producing 1h rows: persistent zones, zone age/strength/stability, zone added/removed strength, price-zone touch/acceptance/rejection, absorption/exhaustion, directional liquidity vacuum, pressure agreement/divergence, and market structure context.
4. Live DB writers were paused first. News, web, global context, orderbook, and GKG/GDELT-related writers/backfills were stopped; unrelated entry-sieve/backtest workers were left alone.
5. Initial raw-archive build exposed a serious stale-gap bug: rows could appear through missing archive periods if sample timing carried across archive files.
6. Fix: reset sample timing and book state at each raw archive boundary, and require each archive/day to start from its own snapshot. All 98 available raw ZIPs start with a snapshot.
7. Final parquet: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet`.
8. Final parquet hash: `SHA256 0370FF27955E5E5C51E7A03108BE2B6F67E1B0E3DDF806465C54EDF6F57D0A5F`.
9. Final feature coverage: `2356` 1h rows, `183` columns, first hour `2025-05-01 01:00:00+00:00`, last hour `2026-02-16 01:00:00+00:00`.
10. Coverage by month: May 2025 `73`, July 2025 `503`, August 2025 `266`, October 2025 `743`, November 2025 `2`, January 2026 `407`, February 2026 `362`.
11. Validation: `0` duplicate feature hours, `0` source-after-feature timestamp violations, `0` infinite numeric cells, `0` all-null numeric columns. Coverage ratio min/mean/max: `0.0167 / 0.9983 / 1.0`.
12. Direct state tests were rerun after the rebuild. Outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_state_experiments_trader_state_v1_report.csv`, `orderbook_state_experiments_trader_state_v1_candidates.csv`, and `orderbook_state_experiments_trader_state_v1_summary.json`.
13. Direct test summary: `132` result rows, `28` candidates, `2` strict passes across four eligible monthly windows.
14. Strict passes: `trader_state_breakout_failure -> fakeout_next_24h` and `trader_state_breakout_failure -> breakout_failure_next_6h`.
15. Important interpretation: the strict pass is inverted relative to the feature name. High `trader_state_breakout_failure` selected rows had fewer future fakeouts and fewer next-6h breakout failures. It likely identifies a resolved rejection/cleared-resistance state, not a clean forward failure predictor.
16. Added `ContextBybitOrderbookTraderStateParquetFreqAIResearchStrategy` and `bybit_trader_state` FreqAI feature profile.
17. FreqAI setup/run: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_behaviour_setup_20260525_152257`.
18. FreqAI comparison outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_trader_state_freqai_metrics_all.csv`, `orderbook_trader_state_freqai_metrics_focus.csv`, `orderbook_trader_state_freqai_price_vs_trader.csv`, and `orderbook_trader_state_freqai_comparison_summary.csv`.
19. Headline FreqAI lift versus price-only Ridge: `future_return_24h` mean correlation delta `+0.2136` across `3/3` positive windows, `future_max_drawdown_24h` `+0.1938` across `3/3`, `future_return_6h` `+0.0738` across `3/3`.
20. LightGBM trader-state lift was mixed/weak: `future_return_24h` mean correlation delta `+0.0450` across `2/3` windows; `future_return_6h` degraded by `-0.0074`.
21. Ridge prediction scale was poorly calibrated on trader-state features, especially event/binary labels. Treat Ridge as ranking/correlation evidence only, not a deployable predictor.
22. Current verdict: raw-L2 trader-state features are useful enough to continue, mainly as a resistance/failure/acceptance risk-context layer. They are not yet a standalone entry model.
23. Full report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_trader_state_20260525.md`.

Raw Bybit trader-state refinement update on 2026-05-25:

1. Added gap-safe second-pass state calculation in `orderbook_trader_state_features.py`: rolling, shift, duration, and acceptance/hold features are now calculated inside contiguous archive coverage segments only. A gap larger than 2h starts a new segment.
2. Added `--recompute-second-pass-from` to rebuild the state layer from the already validated hourly parquet base without rereading all raw ZIPs.
3. Added refined state columns splitting the inverted winner into: `pre_breakout_failure_risk`, `resistance_rejection_resolved`, `resistance_cleared`, `post_failure_chop`, `ask_absorption_before_reaction`, and mirrored support/breakdown states.
4. Updated parquet: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet`.
5. Updated parquet hash: `SHA256 15A9AE964D0D29F209BE3F3413598CB6D9B1316F1A3CA856A3D8E0EC94CD1F05`.
6. Validation: `2356` rows, `193` columns, `0` duplicate dates, `0` source-after-feature violations, `0` infinite numeric cells, and `0` all-null numeric columns.
7. Direct refined test outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_state_experiments_trader_state_refined_v1_report.csv`, `orderbook_state_experiments_trader_state_refined_v1_candidates.csv`, and `orderbook_state_experiments_trader_state_refined_v1_summary.json`.
8. Direct refined test summary: `284` result rows, `50` candidates, `4` strict passes across four eligible monthly windows.
9. Strict passes: old `trader_state_breakout_failure -> fakeout_next_24h`, new `trader_state_support_bounce_resolved -> breakdown_failure_next_6h`, old `trader_state_breakout_failure -> breakout_failure_next_6h`, and new `trader_state_resistance_rejection_resolved -> breakout_failure_next_6h`.
10. Important interpretation: the useful refined signal is still mostly resolved-state detection. True pre-event risk states did not pass strict controls.
11. Added `ContextBybitOrderbookTraderRefinedParquetFreqAIResearchStrategy`, a narrow 37-column refined trader-state profile, and `ContextBybitOrderbookTraderEventPriceOnlyFreqAIResearchStrategy`, a fair price-only event-label control.
12. Focused FreqAI runs: refined Ridge/LightGBM versus price-event Ridge/LightGBM across quiet, volatile, and event windows.
13. Focused FreqAI artifacts: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\orderbook_behaviour_setup_20260525_192000`, `orderbook_behaviour_setup_20260525_201207`, `orderbook_trader_refined_with_price_event_metrics_all.csv`, `orderbook_trader_refined_vs_price_event.csv`, and `orderbook_trader_refined_vs_price_event_summary.csv`.
14. Best model result: refined LightGBM improved `breakout_failure_next_6h` over price-event LightGBM in `3/3` windows with mean correlation delta `+0.0348` and mean AUC delta `+0.0547`.
15. Refined LightGBM also improved `breakdown_failure_next_6h` in `3/3` windows with mean correlation delta `+0.0649` and mean AUC delta `+0.0520`.
16. Refined Ridge was weaker but still improved `future_max_drawdown_24h` by mean correlation delta `+0.0363` across `3/3` windows and `fakeout_next_24h` mean AUC delta `+0.0233` across `3/3` windows.
17. Current verdict: refined raw-L2 orderbook features are most promising as event/risk filters around failed support/resistance interactions, especially 6h breakout/breakdown failure labels. They are not yet a standalone entry model.
18. Full refinement report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_trader_state_refinement_20260525.md`.

Event-window refinement update on 2026-05-25:

1. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\orderbook_event_window_tests.py`.
2. Fixed the event-window flaw found in v1: the original near-support/near-resistance gates were too broad because approach flags were effectively always on.
3. Tightened event windows to require close distance to a stable/strong zone and range location near the relevant side of the 24h range.
4. Refined event base sizes are now narrow enough for trader-style tests: July 2025 resistance/support `127/69`, October 2025 `179/102`, January 2026 `85/120`, February 2026 `55/41`.
5. Added context-relative labels for top/worst outcomes inside each actual support/resistance event window, so sparse absolute crash/breakout labels do not dominate the test.
6. Direct event-window v3 outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_event_window_tests_event_windows_v3_report.csv`, `orderbook_event_window_tests_event_windows_v3_candidates.csv`, and `orderbook_event_window_tests_event_windows_v3_summary.json`.
7. Direct event-window v3 result: `53` candidates, `0` strict passes. No single feature currently proves stable breakout confirmation or crash detection across all four usable months.
8. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\orderbook_event_classifier_tests.py`, a leave-one-month-out classifier harness comparing price-only features against price plus compact orderbook behaviour features inside support/resistance event windows.
9. Classifier outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_event_classifier_tests_event_classifier_v1_report.csv`, `orderbook_event_classifier_tests_event_classifier_v1_summary.csv`, and `orderbook_event_classifier_tests_event_classifier_v1_meta.json`.
10. Classifier result after tightening the pass rule: `0` strict passes and `1` watchlist pass.
11. Watchlist result: `support_touch_failure -> fakeout_next_24h` improved from price-only AUC `0.4892` to price+orderbook AUC `0.5409`, with positive orderbook delta in `4/4` held-out months.
12. Crash near miss: `support_touch_crash_detection -> breakdown_success_next_6h` improved versus price-only in `4/4` windows, but mean orderbook AUC was only `0.4687`, so it is not useful yet.
13. Resistance near misses: `resistance_touch_failure -> context_worst_drawdown_24h` reached orderbook AUC `0.6079` but only added `+0.0262` over price-only; `breakout_failure_next_6h` reached orderbook AUC `0.6418` but only added `+0.0180`.
14. Current verdict: event-window testing is cleaner, but full FreqAI is not justified from these results yet. The only active lead is support-touch fakeout detection; breakout confirmation and crash detection need better sequence features, better labels, and more historical coverage.
15. Alternative paths to keep active: 1m/5m event-sequence features, better crash/path labels, acceptance-feature audit, support fakeout refinement, more historical L2 coverage, and cross-source confluence with news/global/macro/trends.
16. Full event-window report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\orderbook_event_window_classifier_20260525.md`.

Structure + orderbook confluence update on 2026-05-26:

1. User highlighted that local custom indicators should anchor the research because they encode better trader structure than generic qtpylib/talib indicators.
2. Inspected the active custom indicator stack: `complex_volume_profile.py`, `pivot_foundation.py`, `complex_trendline_projection_v2.py`, `pattern_bos_choch.py`, and pattern/geometry modules.
3. Key design decision: orderbook should not define the setup by itself. Define structure first with VP/TLV2/BOS/CHoCH, then test whether orderbook confirms or contradicts the setup.
4. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\structural_feature_cache.py`.
5. The cache builder stamps features at candle close / availability time and merges 4h/1d informative features backward onto the 1h close timeline.
6. Full 2020-2026 structural build was too slow for an interactive pass, so the workflow was corrected to support `--start`, `--end`, and `--warmup-days`.
7. Built the first structural cache for `2025-05-01` to `2026-02-17` with `240` warmup days, using VP, TLV2, and BOS/CHoCH on 1h/4h/1d.
8. Structural cache output: `C:\FreqTradeStuff\user_data\research_news_data\context_features\structural_cache\btc_structural_features_1h_latest.parquet`.
9. Cache metadata: `C:\FreqTradeStuff\user_data\research_news_data\context_features\structural_cache\btc_structural_features_1h.meta.json`.
10. Cache result: `176` numeric columns, `0` all-null numeric columns, timeframe rows computed: 1h `12769`, 4h `3193`, 1d `533`. Runtime was about `8m15s`.
11. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\structure_orderbook_confluence_tests.py`.
12. The confluence harness compares `price`, `structure`, `structure_vp`, `structure_orderbook`, and `structure_vp_orderbook` on rows where orderbook features are present.
13. First confluence outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_structure_orderbook_v1_report.csv`, `structure_orderbook_confluence_structure_orderbook_v1_summary.csv`, and `structure_orderbook_confluence_structure_orderbook_v1_meta.json`.
14. Strict passes: `0`.
15. Best lead: `support_fakeout -> breakdown_failure_next_6h`, with structure+VP+orderbook AUC `0.5550` versus structure+VP AUC `0.5307`; delta `+0.0243`, positive in `2/3` windows. This is a near-pass but below the `+0.03` delta and `3` positive-window thresholds.
16. Structure-only result worth noting: `support_fakeout -> failed_breakdown_next_24h` had structure-only AUC `0.6231`, but VP/orderbook reduced it, so the setup definition should be split instead of blindly adding more data.
17. Breakout confirmation and crash detection are still not solved. Current event masks do not create enough stable held-out evidence for those targets.
18. Current verdict: custom-indicator structure-first framing is better than orderbook-only testing, but the first setup definitions are too crude. Continue by splitting support fakeout and breakout failure into more specific trader subtypes, then retest before launching larger FreqAI runs.
19. Full confluence report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_20260526.md`.

Structure + orderbook confluence v2 / FreqAI handoff update on 2026-05-26:

1. Refined setup masks were added for support reclaim, VAL reclaim, TLV2 support reclaim, VP lower rejection, range-low reclaim, TLV2 resistance rejection, VAH rejection, and breakout acceptance.
2. V2 direct outputs: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_structure_orderbook_v2_report.csv`, `structure_orderbook_confluence_structure_orderbook_v2_summary.csv`, and `structure_orderbook_confluence_structure_orderbook_v2_meta.json`.
3. V2 result: `29` summary rows, `1` strict pass, and `5` FreqAI-ready rows.
4. Strict pass: `vah_rejection -> breakout_failure_next_6h`. It scored price AUC `0.4526`, structure+VP AUC `0.5313`, structure+VP+orderbook AUC `0.5929`, orderbook delta `+0.0616`, positive orderbook delta in `3/4` held-out months.
5. FreqAI-ready near leads: `crash_detection -> large_drawdown_next_6h`, `vp_lower_rejection -> failed_breakdown_next_24h`, `tlv2_resistance_reject -> breakout_failure_next_6h`, and `support_fakeout -> breakdown_failure_next_6h`.
6. Added FreqAI strategy variants: `ContextStructureVahRejectionFreqAIResearchStrategy` and `ContextStructureOrderbookVahRejectionFreqAIResearchStrategy`.
7. Added setup script: `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\structure_orderbook_confluence_freqai_setup.py`.
8. Added report script: `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\structure_orderbook_confluence_freqai_report.py`.
9. Created setup manifest: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs\structure_orderbook_confluence_setup_20260526_180006\structure_orderbook_confluence_freqai_manifest.json`.
10. Short smoke FreqAI window completed for `2026-01-01` to `2026-01-15` using LightGBM on both structure+VP control and structure+VP+orderbook candidate.
11. Smoke metrics: structure+VP control scored AUC `0.6259` on all predicted rows and `0.7500` on `vah_rejection` rows. Structure+VP+orderbook scored AUC `0.5765` on all predicted rows and `0.7000` on `vah_rejection` rows.
12. Smoke report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_freqai_structure_orderbook_vah_smoke_20260101_20260115_metrics.csv`.
13. Interpretation: the FreqAI path is functional, but the short smoke does not confirm the orderbook lift. The direct test still justifies a broader full-manifest run before rejecting the orderbook confluence idea.

Structure + orderbook confluence broad FreqAI result on 2026-05-26:

1. Full manifest run completed for `2025-07-01` to `2026-02-18`.
2. Control model: `ContextStructureVahRejectionFreqAIResearchStrategy` / `structure_vp_tree`.
3. Candidate model: `ContextStructureOrderbookVahRejectionFreqAIResearchStrategy` / `structure_vp_orderbook_tree`.
4. Both models wrote `18` broad prediction chunks.
5. Standard report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_freqai_structure_orderbook_vah_broad_20250701_20260218_metrics.csv`.
6. Orderbook-present filtered report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_freqai_broad_orderbook_present_metrics.csv`.
7. Important review finding: the direct test filtered to rows where orderbook was present, but the FreqAI strategy currently fills missing orderbook parquet rows with zero. Therefore the cleanest judgement is the `vah_rejection_orderbook_present` scope, not all predicted rows.
8. Broad filtered result on `vah_rejection_orderbook_present`: structure+VP rows `312`, AUC `0.5911`, average precision `0.2749`, top-quintile actual rate `0.2581`, bottom-quintile actual rate `0.1290`.
9. Broad filtered result on `vah_rejection_orderbook_present`: structure+VP+orderbook rows `312`, AUC `0.5867`, average precision `0.2757`, top-quintile actual rate `0.2581`, bottom-quintile actual rate `0.1452`.
10. Verdict: this FreqAI formulation does not confirm a meaningful orderbook lift for `vah_rejection -> breakout_failure_next_6h`. It is not a useful standalone candidate as currently built.
11. Month detail on orderbook-present VAH rows: orderbook improved October 2025 AUC (`0.5754` vs `0.5478`) but underperformed January 2026 (`0.3704` vs `0.4286`) and February 2026 (`0.7052` vs `0.7138`), with July/August equal.
12. Do not conclude orderbook is useless. Conclude only that VAH rejection plus the current refined orderbook profile does not survive broader FreqAI validation.
13. Next candidate list should include: `tlv2_resistance_reject -> breakout_failure_next_6h`, `crash_detection -> large_drawdown_next_6h`, `vp_lower_rejection -> failed_breakdown_next_24h`, `support_fakeout -> breakdown_failure_next_6h`, and a cleaner `breakout_acceptance -> breakout_success_next_6h` formulation.

| Test ID | Objective | Pass rule summary |
|---|---|---|
| `wall_proximity` | Check whether close bid/ask walls have stable directional or path-risk effects. | Same sign in at least two of three windows, larger than random controls in at least two windows, and weaker after shuffled labels. |
| `wall_notional_score` | Check whether unusually large wall notional/score predicts rejection, absorption, or drawdown risk. | Top-decile effect beats random controls in at least two windows and is not reproduced by shuffled labels. |
| `liquidity_vacuum` | Check whether thin near-price liquidity predicts downside path risk. | Fragility bucket has coherent path-risk effect in at least two windows, beats random controls, and has bootstrap support in at least one window. |
| `pressure_flip` | Check whether rapidly changing book pressure predicts short-term movement. | Top-decile flip/volatility bucket beats random and shuffled controls in at least two windows. |
| `spread_fragility` | Check whether spread and gap states identify fragile conditions. | Effect appears outside one crash-only window, beats random controls in at least two windows, and weakens under shuffled labels. |

## Important Non-Goals

1. Do not optimize entries/exits from these research runs.
2. Do not judge this by backtest profit.
3. Do not add scraping or API calls inside strategy code.
4. Do not feed raw article text directly to FreqAI.
5. Do not silently forward-fill missing compact orderbook features into historical gaps.
6. Do not delete raw orderbook data until compaction validation and retention tooling are deliberately approved.

Bybit historical orderbook preparation update on 2026-05-26:

1. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\orderbook\historical_bybit_raw_sync.py`, a small reusable sync tool for public Bybit orderbook ZIP archives. It reads the archive directory listing, skips local ZIPs, downloads via `.tmp` then atomic replace, validates ZIPs, and writes a manifest.
2. Free public archive coverage found:
   - Bybit spot `BTCUSDT` OB200: `2025-04-29` to `2026-05-25`, `392` daily ZIPs.
   - Bybit linear `BTCUSDT` OB500/OB200: `2023-01-18` to `2026-05-25`, `1224` daily ZIPs.
   - Bybit inverse `BTCUSD` OB500/OB200: `2023-01-18` to `2026-05-25`, `1224` daily ZIPs.
   - No free archive coverage back to 2020 was found in this Bybit orderbook source.
3. Original storage decision at launch: keep existing spot raw ZIPs under `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw`; put large linear/inverse raw archives on `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\` because `C:` had about 140 GB free while `D:` had about 732 GB free. This was superseded on 2026-05-29; see the later raw archive location update.
4. Active raw sync jobs launched:
   - Spot log: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_ob200_raw_sync_20260526_220741.out.log`.
   - Linear/inverse log: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_linear_inverse_raw_sync_20260526_221308.out.log`.
5. Active feature-build chains launched:
   - Spot waits for spot raw sync, then writes `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet`.
   - Linear waits for linear raw sync, then writes `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_linear_latest.parquet`.
   - Inverse waits for inverse raw sync, then writes `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_inverse_latest.parquet`.
6. Rebuilt the custom-structure cache over the orderbook era with VP, TLV2 support/resistance, BOS/CHoCH, and pattern geometry enabled. Target range: `2025-04-29` to `2026-05-12`, limited by local 4h/1d OHLCV freshness.
7. Tightened `structural_feature_cache.py` so all-null numeric indicator columns are dropped before writing FreqAI-ready parquet. This avoids handing empty geometry columns to FreqAI.
8. Exported the current context feature SQL snapshot to `C:\FreqTradeStuff\user_data\research_news_data\context_features\exports\context_features_1h_20260526_220839.parquet`; coverage is `2020-01-01` to `2026-05-24`, `56041` rows, and no lookahead violations were reported by the existing validator.
9. Timing rule preserved: raw orderbook ZIPs are converted directly to 1h parquet snapshots; no live SQLite collector tables are used for this historical preparation. The raw-ZIP builder resets book state per archive and second-pass rolling state is segmented across archive gaps.
10. Open issue: once the long raw sync and chained feature builds complete, validate row counts, date coverage, duplicate dates, source timestamp alignment, and all-null numeric columns before using these caches for FreqAI.
11. Format improvement added after review: `orderbook_trader_state_features.py` now stamps each 1h row with `market_key`, numeric `obts_feature_present`, and numeric `obts_market_id`. This makes missing-orderbook gaps explicit and makes spot/linear/inverse caches distinguishable when tested separately or merged later.
12. Added guarded raw-retention tool: `C:\FreqTradeStuff\user_data\Custom_Launcher\orderbook\historical_bybit_retention.py`. It validates derived feature parquet coverage by raw archive day, duplicate dates, source timestamp alignment, all-null numeric columns, and active `.tmp` files. It will not delete raw ZIPs unless validation passes and both `--delete-raw` and `--execute` are supplied.
13. Raw retention rule: delete raw Bybit ZIPs only after the matching 1h parquet is complete, validated, and stored under `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features`. Do not delete raw archives while sync or feature-build logs are still active.

Structure/orderbook FreqAI setup fixes on 2026-05-26:

1. Fixed the P1 orderbook-present mismatch in `ContextFreqAIResearchStrategy.py`. Compact orderbook feature merges now add both unprefixed `orderbook_present` for target masking and numeric FreqAI features `%-orderbook_present` / `%-orderbook_missing`.
2. For orderbook-confluence strategies, `&-so_vah_rejection_breakout_failure_next_6h` is now masked to rows where both the structure event and real orderbook data are present. This prevents missing orderbook rows from being silently trained as zero-valued orderbook context.
3. Fixed the reusable FreqAI report path in `structure_orderbook_confluence_freqai_report.py`: it now keeps `orderbook_present`, reports an extra `{event_id}_orderbook_present` scope, and can read `event_id`, `actual_column`, and `prediction_column` from the manifest or CLI.
4. Re-ran the report against the existing broad manifest. The report now emits `vah_rejection_orderbook_present` rows automatically at `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\structure_orderbook_confluence_freqai_validation_after_present_fix_metrics.csv`.
5. Reduced setup overhead in `structure_orderbook_confluence_freqai_setup.py`: `--plot-feature-importances` now defaults to `0`, so repeated test queues do not spend time writing HTML feature-importance plots unless explicitly requested.
6. Generalised setup manifest metadata with `candidate_id`, `event_id`, `actual_column`, `prediction_column`, and objective text. This reduces duplication for future candidate batches, although genuinely new candidate logic may still require a matching research strategy/event feature.
7. Validation: Python compile passed for the strategy, setup, report, raw orderbook builder, and retention tool. `git diff --check` passed on the touched files.

FreqAI research queue groundwork on 2026-05-26:

1. Sanity check against the earlier objective list found a real gap: feature caches and one-candidate setup existed, but the reusable queued research harness, feature profile registry, executor, and result ledger did not.
2. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\feature_profile_registry.py`.
   - Current profiles: `structure_vp_tree` control and `structure_vp_orderbook_tree` candidate.
   - Profiles declare feature family, strategy, model, required parquet files, event id, target column, prediction column, objective, pass rule, and control profile.
3. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\freqai_experiment_queue.py`.
   - Creates queue manifests under `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_*`.
   - Generates per-experiment FreqAI configs with feature-importance plotting disabled by default.
4. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\run_freqai_experiment_queue.py`.
   - Runs only the next pending experiment.
   - Refuses to run when another `freqtrade` process is active unless explicitly overridden.
   - Writes stdout/stderr logs in the experiment directory.
5. Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\score_freqai_experiment.py`.
   - Scores predictions using the shared structure/orderbook scorer.
   - Appends machine-readable rows to `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\freqai_research_results_ledger.csv`.
6. Created initial queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260526_223209\freqai_experiment_queue.json`.
   - Contains 2 pending experiments: structure baseline and structure+orderbook candidate over `20250701-20260218`.
7. Created automation `FreqAI research queue runner`.
   - Runs every 3 hours.
   - Picks the newest queue.
   - Runs at most one pending experiment.
   - Skips if another FreqAI/freqtrade process is active.
   - Updates the ledger and summarizes new results into this research file.
8. Current limitation: this is the minimal queue framework, not yet the full family list. News/context, macro/global, Google Trends, full confluence, and ablation profiles should be added only once their parquet caches have enough validated coverage for the target windows.

FreqAI queue/setup review fixes on 2026-05-26:

1. Full review found the previous queue groundwork was directionally correct but had several implementation gaps.
2. Fixed a P1 queue-runner guard flaw: `_freqai_running()` could match its own PowerShell process query and skip forever. The guard now excludes its own process and only matches actual `freqtrade` command lines.
3. Fixed stale running-state handling: if an experiment is marked `running` but no FreqAI process is active and no return code exists, the runner resets it to `pending` with `stale_running_reset_at`.
4. Fixed file-handle handling in the queue runner by using context managers for stdout/stderr log files.
5. Fixed FreqAI data cleaning: `market_key` and `obts_schema_version` are now treated as metadata, not coerced into all-null numeric feature columns in full orderbook profiles.
6. Fixed current orderbook parquet formatting by recomputing the second-pass layer on `orderbook_trader_state_1h_latest.parquet`; it now has `market_key`, `obts_feature_present`, and `obts_market_id`.
7. Fixed scorer flexibility: binary metrics are only calculated for binary targets; regression/path targets keep correlation and bucket stats without forcing AUC/AP.
8. Added feature-cache validation to queue creation. It now rejects required parquet files with missing dates, duplicate dates, unparseable timestamps, or all-null numeric columns.
9. Added candidate-vs-control comparison rows to the result ledger when a candidate profile declares `control_profile_id`.
10. Validation after fixes:
    - `orderbook_trader_state_1h_latest.parquet`: `2356` rows, `196` columns, `0` duplicate dates, `0` all-null numeric columns.
    - `btc_structural_features_1h_latest.parquet`: `9073` rows, `252` columns, `0` duplicate dates, `0` all-null numeric columns.
    - `context_features_1h_20260526_220839.parquet`: `56041` rows, `404` columns, `0` duplicate dates, `0` all-null numeric columns.
11. Created a refreshed validated queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260526_223559\freqai_experiment_queue.json`.

FreqAI overnight queue launch update on 2026-05-26:

1. First overnight launch check found a material setup flaw before leaving the queue running: broad orderbook FreqAI profiles tried to train across periods before enough orderbook coverage existed. FreqAI dropped all training rows for the broad orderbook candidate, so that broad orderbook run is invalid.
2. Fixed the queue design at the profile level:
   - Structure-only profiles can still run broad and monthly windows.
   - Orderbook profiles are restricted to contiguous orderbook-present windows only.
   - Orderbook profiles use shorter 14-day training and 7-day backtest periods because the current archive coverage is sparse and segmented.
3. Added Ridge controls/candidates alongside LightGBM tree profiles:
   - `structure_vp_tree`
   - `structure_vp_ridge`
   - `structure_vp_orderbook_tree`
   - `structure_vp_orderbook_ridge`
4. Added queue manifest locking in `run_freqai_experiment_queue.py` so the scheduled automation and an overnight queue loop cannot write the same queue manifest at the same time.
5. Created the coverage-safe overnight queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260526_224438\freqai_experiment_queue.json`.
6. Queue contents:
   - 8 structure-only runs across broad, July/August, October, and January/February windows.
   - 6 orderbook runs across orderbook-present July/August, October, and January/February windows.
7. Guardrail: do not interpret the earlier failed broad orderbook experiment as a model result. It was a data-coverage/setup failure, not evidence about orderbook usefulness.
8. Launched the coverage-safe queue with `run_freqai_experiment_queue_until_idle.py`; the runner writes per-experiment logs under `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260526_224438\`.
9. Initial queue progress after launch:
   - `structure_vp_tree-broad-20250701-20260218-20260526_224438`: completed and scored.
   - `structure_vp_ridge-broad-20250701-20260218-20260526_224438`: completed and scored.
   - Remaining experiments continue through the overnight runner.

Expanded FreqAI spot-orderbook queue on 2026-05-27:

1. The full Bybit spot OB200 historical feature cache is now available at `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet`.
2. Validation before queue launch:
   - Rows: `9406`.
   - Date range: `2025-04-29 04:00:00+00:00` to `2026-05-26 01:00:00+00:00`.
   - Duplicate dates: `0`.
   - All-null numeric columns: `0`.
   - `obts_feature_present` rows: `9406`.
3. Expanded the FreqAI profile registry to include generated research profiles across:
   - Feature families: price baseline, structure/VP, Bybit spot orderbook, structure/VP + Bybit spot orderbook.
   - Models: LightGBM tree and Ridge.
   - Targets: `future_return_6h`, `future_return_24h`, `future_max_drawdown_24h`, `breakout_success_next_6h`, `breakout_failure_next_6h`, `breakdown_success_next_6h`, and `fakeout_next_24h`.
4. Added `all_rows` event-mask support so generic return/path labels can be scored without pretending they are tied to a specific setup event.
5. Created and launched expanded queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_015359\freqai_experiment_queue.json`.
6. Queue size: `168` experiments:
   - `42` price baseline.
   - `42` structure/VP.
   - `42` Bybit spot orderbook.
   - `42` structure/VP + Bybit spot orderbook.
   - Split evenly across `spot_full`, `spot_q4_2025`, and `spot_q1_2026`.
7. The queue was launched with `run_freqai_experiment_queue_until_idle.py` and is allowed to run alongside existing non-FreqAI backtests. The queue runner still avoids concurrent FreqAI jobs and uses the queue manifest lock.
8. Treat this as the first serious spot-orderbook FreqAI batch. Linear and inverse Bybit archives are still not included because their downloads/features are not complete.

Expanded queue results and follow-up stability queue on 2026-05-27:

1. Queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_015359\freqai_experiment_queue.json` completed `168 / 168` experiments with `0` failures.
2. Strongest event-scope results from that queue:
   - `orderbook_tree_breakdown_success_6h` on `spot_q4_2025`, crash-detection rows: AUC `0.7367`, AP `0.4905`, prediction/actual corr `0.3911`, top bucket event rate `35.56%`, bottom bucket `2.22%`.
   - `price_ridge_breakout_success_6h` on `spot_q1_2026`, breakout-acceptance rows: AUC `0.7226`, AP `0.5570`, corr `0.3752`, top bucket `57.75%`, bottom bucket `7.04%`.
   - `price_tree_breakdown_success_6h` on `spot_full`, crash-detection rows: AUC `0.7171`, AP `0.3956`, corr `0.3298`, top bucket `36.50%`, bottom bucket `7.30%`.
   - `orderbook_tree_breakout_failure_6h` on `spot_q1_2026`, VAH-rejection rows: AUC `0.7076`, AP `0.3408`, corr `0.2884`, top bucket `40.68%`, bottom bucket `8.47%`.
3. Weakest event-scope results were Ridge structure-heavy crash-detection variants on `spot_full`, around AUC `0.452`, with inverted bucket behaviour. Do not prioritize those forms unless a later stability test contradicts this.
4. Generic return-regression targets were weak or directionally unstable compared with event labels. The strongest evidence is currently in event classification, not ordinary future-return regression.
5. Created the follow-up stability queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_042809\freqai_experiment_queue.json`.
6. Follow-up queue purpose: test whether the promising event signals survive narrower month/recent windows rather than only broad or quarterly windows.
7. Follow-up queue size: `120` experiments:
   - Families: price, Bybit spot orderbook, structure/VP + Bybit spot orderbook.
   - Targets: `breakdown_success_next_6h`, `breakout_success_next_6h`, `breakout_failure_next_6h`, `fakeout_next_24h`.
   - Windows: `spot_recent`, August 2025 through April 2026 monthly windows.
8. Follow-up queue launched with `run_freqai_experiment_queue_until_idle.py`.

Monthly stability queue results and next ablation on 2026-05-27:

1. Queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_042809\freqai_experiment_queue.json` completed `120 / 120` experiments with `0` failures.
2. Strongest monthly stability results:
   - `price_tree_breakout_failure_6h` on November 2025 VAH-rejection rows: AUC `0.8750`, AP `0.4908`, corr `0.4762`, top bucket event rate `42.86%`, bottom bucket `0.00%`. Sample size: `70` event rows.
   - `price_ridge_breakout_success_6h` on January 2026 breakout-acceptance rows: AUC `0.8394`, AP `0.7492`, corr `0.5533`, top bucket `75.00%`, bottom bucket `0.00%`. Sample size: `163` event rows.
   - `orderbook_tree_breakdown_success_6h` on October 2025 crash-detection rows: AUC `0.8177`, AP `0.4746`, corr `0.5012`, top bucket `28.57%`, bottom bucket `7.14%`. Sample size: `70` event rows.
   - `orderbook_tree_breakout_failure_6h` on November 2025 VAH-rejection rows: AUC `0.8167`, AP `0.4764`, corr `0.4415`, top bucket `42.86%`, bottom bucket `0.00%`. Sample size: `70` event rows.
3. Weak/inverted results:
   - `structure_vp_orderbook_tree_breakdown_success_6h` on November 2025 crash-detection rows: AUC `0.2518`, corr `-0.3587`, top bucket `8.33%`, bottom bucket `58.33%`.
   - `structure_vp_orderbook_tree_breakdown_success_6h` on March 2026 crash-detection rows: AUC `0.2530`, corr `-0.2324`.
   - `structure_vp_orderbook_tree_fakeout_24h` on February and April 2026 was also poor/inverted.
4. Interpretation so far:
   - The strongest patterns remain event labels, not generic return regression.
   - Price/tree and orderbook/tree often work better than structure/VP+orderbook combined.
   - The combined structure/VP+orderbook feature set may be over-noisy, over-constrained, or training on conflicting setup semantics.
   - Small monthly event samples mean the high AUCs are promising but not yet sufficient as final proof.
5. Next queue should isolate whether structure/VP alone is useful or whether it is the source of the combined-model instability.
6. Created and launched ablation queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_055758\freqai_experiment_queue.json`.
7. Ablation queue size: `80` experiments:
   - Family: structure/VP only.
   - Models: LightGBM tree and Ridge.
   - Targets: `breakdown_success_next_6h`, `breakout_success_next_6h`, `breakout_failure_next_6h`, `fakeout_next_24h`.
   - Windows: `spot_recent`, August 2025 through April 2026 monthly windows.
8. Purpose: determine whether structure/VP alone is useful, neutral, or harmful before combining it with orderbook again.

Structure/VP ablation results and context queue on 2026-05-27:

1. Queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_055758\freqai_experiment_queue.json` completed `80 / 80` experiments with `0` failures.
2. Best structure/VP-only results:
   - `structure_vp_tree_breakout_failure_6h` on December 2025 VAH-rejection rows: AUC `0.8019`, AP `0.3769`, corr `0.3504`, top bucket event rate `26.09%`, bottom bucket `0.00%`. Sample size: `115`.
   - `structure_vp_tree_breakdown_success_6h` on October 2025 crash-detection rows: AUC `0.7747`, AP `0.4790`, corr `0.3596`, top bucket `28.57%`, bottom bucket `7.14%`. Sample size: `70`.
   - `structure_vp_tree_breakout_failure_6h` on November 2025 VAH-rejection rows: AUC `0.7650`, AP `0.4771`, corr `0.4158`, top bucket `50.00%`, bottom bucket `14.29%`. Sample size: `70`.
3. Worst structure/VP-only results:
   - `structure_vp_tree_breakdown_success_6h` on November 2025 crash-detection rows: AUC `0.2401`, corr `-0.3597`, top bucket `4.17%`, bottom bucket `58.33%`.
   - `structure_vp_ridge_breakout_failure_6h` on September 2025 VAH-rejection rows: AUC `0.2438`.
   - `structure_vp_ridge_breakdown_success_6h` on March 2026 crash-detection rows: AUC `0.2879`.
4. Interpretation:
   - Structure/VP can work for VAH rejection/failure setups in some months, but it is unstable for crash detection and Ridge variants.
   - The combined structure+orderbook instability likely comes from feature-family/target mismatch, not just orderbook noise.
   - Future structure use should be event-gated by setup type rather than blindly combined into every target.
5. Added context/news/global feature profiles using the existing `context_features.sqlite` store and fixed the queue validator so SQLite feature stores are validated as SQLite instead of read as parquet.
6. Created and launched context queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_065856\freqai_experiment_queue.json`.
7. Context queue size: `96` experiments:
   - Families: context-only and price+context.
   - Models: LightGBM tree and Ridge.
   - Targets: `breakdown_success_next_6h`, `breakout_success_next_6h`, `breakout_failure_next_6h`, `fakeout_next_24h`.
   - Windows: `spot_recent`, October/November/December 2025, January 2026, and March 2026.
8. Purpose: test whether existing external context/news/global/macro features add signal to the strongest event labels without touching live collectors.

Context queue results and replication queue on 2026-05-27:

1. Queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_065856\freqai_experiment_queue.json` completed `96 / 96` experiments with `0` failures.
2. Best context-related results:
   - `price_context_tree_breakout_failure_6h` on November 2025 VAH-rejection rows: AUC `0.8650`, AP `0.4267`, corr `0.3984`, top bucket event rate `50.00%`, bottom bucket `0.00%`. Sample size: `70`.
   - `price_context_ridge_breakout_failure_6h` on December 2025 VAH-rejection rows: AUC `0.8267`, AP `0.2605`, corr `0.2957`, top bucket `30.43%`, bottom bucket `0.00%`. Sample size: `115`.
   - `price_context_tree_breakdown_success_6h` on November 2025 crash-detection rows: AUC `0.8259`, AP `0.6623`, corr `0.5289`, top bucket `66.67%`, bottom bucket `4.17%`. Sample size: `124`.
3. Weak context-only results:
   - `context_ridge_breakdown_success_6h` on December 2025 crash-detection rows: AUC `0.1778`, corr `-0.3811`.
   - `context_tree_breakdown_success_6h` on January 2026 crash-detection rows: AUC `0.1959`, corr `-0.3347`.
4. Interpretation:
   - Context-only is not reliable enough.
   - Price+context can strongly improve event ranking in some windows, especially VAH rejection failure and crash detection.
   - Context features should be used with price/event state, not as standalone market predictors.
5. Created and launched replication queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_092750\freqai_experiment_queue.json`.
6. Replication queue size: `72` experiments:
   - Profiles: strongest price, price+context, orderbook, and structure/VP candidates from prior runs.
   - Targets: `breakout_failure_next_6h`, `breakout_success_next_6h`, and `breakdown_success_next_6h`.
   - Windows: `spot_full`, `spot_q4_2025`, `spot_q1_2026`, `spot_recent`, and October/November/December 2025 plus January 2026 monthly windows.
   - Train/backtest: `90` train days and `14` backtest days, to test whether signals survive a less short-memory setup.
7. Purpose: verify the strongest apparent correlations are not artifacts of the shorter 30-day training queues.

Replication queue results and linear Bybit feature build on 2026-05-27:

1. Queue `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_092750\freqai_experiment_queue.json` completed `72 / 72` experiments with `0` failures.
2. Replication queue strongest results:
   - `price_tree_breakout_failure_6h` on November 2025 VAH-rejection rows: AUC `0.8750`, AP `0.4908`, corr `0.4762`, top bucket event rate `42.86%`, bottom bucket `0.00%`.
   - `price_context_tree_breakout_failure_6h` on November 2025 VAH-rejection rows: AUC `0.8650`, AP `0.4267`, corr `0.3984`, top bucket `50.00%`, bottom bucket `0.00%`.
   - `price_ridge_breakout_success_6h` on January 2026 breakout-acceptance rows: AUC `0.8394`, AP `0.7492`, corr `0.5533`, top bucket `75.00%`, bottom bucket `0.00%`.
   - `price_context_ridge_breakout_failure_6h` on December 2025 VAH-rejection rows: AUC `0.8267`, AP `0.2605`, corr `0.2957`, top bucket `30.43%`, bottom bucket `0.00%`.
3. Replication queue weak results:
   - `price_context_tree_breakdown_success_6h` on January 2026 crash-detection rows: AUC `0.4674`.
   - `price_context_ridge_breakout_failure_6h` on `spot_recent` VAH-rejection rows: AUC `0.4899`.
4. Interpretation:
   - VAH rejection / breakout-failure remains the most repeatable signal so far.
   - Breakout-acceptance had one strong January result but needs more supporting windows.
   - Crash detection is promising but not stable across all windows.
   - Strong signals are still event-scoped, not broad all-candle predictors.
5. Bybit linear raw archive download completed: `1224` ZIPs, about `257.2 GB`, range appears complete through `2026-05-25`.
6. Bybit inverse raw archive download is active and partially complete: `340` ZIPs, about `19.4 GB`, currently around `2023-12-23`.
7. Started Bybit linear 1h trader-state feature build from `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`.
   - Output: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_linear.parquet`.
   - Latest cache: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_linear_latest.parquet`.
   - This should enable spot-vs-linear orderbook testing once build validation passes.

FreqAI queue runner automation check on 2026-05-27:

1. Latest queue checked: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260526_224438\freqai_experiment_queue.json`.
2. Ran `run_freqai_experiment_queue.py` once against that queue. The runner returned `{"status":"idle","reason":"no_pending_experiments"}`, so this automation run did not launch a new experiment.
3. Queue manifest detail at check time: top-level `status` still reads `pending`, but all `14/14` experiments inside the manifest are already `completed`. Treat this run as queue exhausted, not as an in-progress FreqAI job.
4. Newest ledger rows remain the final January/February orderbook-present experiments from `2026-05-26 23:01:47`:
   - `structure_vp_orderbook_tree` on `ob_jan_feb` was the stronger of the final pair, with `roc_auc` about `0.665` on `vah_rejection_only` / `vah_rejection_orderbook_present` rows (`81` rows, top-quintile actual rate `0.3125`, bottom-quintile `0.0`).
   - `structure_vp_orderbook_ridge` on the same window was weak-to-marginal, with `roc_auc` about `0.514` on event rows and `0.245` on all predicted rows, so no evidence yet of reliable linear lift from the added orderbook state.

FreqAI queue runner automation check on 2026-05-27 at 04:37 +01:00:

1. Latest queue checked: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_042809\freqai_experiment_queue.json`.
2. Ran `run_freqai_experiment_queue.py` once against that queue. The runner returned `{"status":"skipped","reason":"queue_locked"}`, so this automation run did not launch a new experiment.
3. Queue manifest at check time still shows live work rather than exhaustion: top-level `status` is `pending`, with `11` experiments `completed`, `109` still `pending`, and `0` `failed`.
4. Newest ledger rows now come from `orderbook_tree_fakeout_24h` on `spot_recent` (`20260301-20260520`):
   - On `breakout_failure_only` rows (`393` rows), `roc_auc` was about `0.5878` and `average_precision` about `0.3393`.
   - Versus the `price_tree_fakeout_24h` control, event-row `roc_auc` was slightly worse (`-0.0032`), while `average_precision` was slightly better (`+0.0050`), so this looks like only marginal orderbook lift so far.

FreqAI queue runner automation check on 2026-05-27 at 07:38 +01:00:

1. Latest queue checked: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_065856\freqai_experiment_queue.json`.
2. Ran `run_freqai_experiment_queue.py` once against that queue. The runner returned `{"status":"skipped","reason":"queue_locked"}`, so this automation run did not launch a new experiment.
3. Queue manifest at check time still shows live work rather than exhaustion: top-level `status` is `pending`, with `50` experiments `completed`, `46` still `pending`, and `0` `failed`.
4. Newest ledger rows now come from `price_context_tree_breakdown_success_6h` on `spot_dec_2025` (`20251201-20251231`):
   - On `crash_detection_only` rows (`33` rows), `roc_auc` was `0.6000` and `average_precision` was about `0.4142`.
   - Versus the `price_tree_breakdown_success_6h` control, event-row lift was still negative (`roc_auc_delta_vs_control` about `-0.0111`, `average_precision_delta_vs_control` about `-0.0018`), while all-predicted-row comparison was materially worse (`roc_auc_delta_vs_control` about `-0.1582`), so this added context blend still does not beat the price baseline on this December crash-detection slice.

FreqAI queue runner automation check on 2026-05-27 at 10:29 +01:00:

1. Latest queue checked: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_092750\freqai_experiment_queue.json`.
2. Ran `run_freqai_experiment_queue.py` once against that queue. The runner returned `{"status":"idle","reason":"no_pending_experiments"}`, so this automation run did not launch a new experiment.
3. Queue manifest detail at check time: all `72/72` experiments inside the manifest are already `completed`, with `0` pending and `0` failed. Treat this queue as exhausted rather than locked or still in progress.
4. Newest ledger rows tied to this queue come from the January 2026 breakout-failure replications:
   - `orderbook_tree_breakout_failure_6h` on `spot_jan_2026` was strong on event rows (`76` rows), with `roc_auc` about `0.7758`, `average_precision` about `0.4008`, and positive lift versus `price_tree_breakout_failure_6h` (`roc_auc_delta_vs_control` about `+0.0879`, `average_precision_delta_vs_control` about `+0.0178`).
   - `structure_vp_tree_breakout_failure_6h` on the same slice was weaker, with event-row `roc_auc` about `0.5955`, `average_precision` about `0.2739`, and negative lift versus the same price control (`roc_auc_delta_vs_control` about `-0.0924`, `average_precision_delta_vs_control` about `-0.1090`).

FreqAI queue runner automation check on 2026-05-27 at 13:40 +01:00:

1. Latest queue checked: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_092750\freqai_experiment_queue.json`.
2. Ran `run_freqai_experiment_queue.py` once against that queue. The runner returned `{"status":"idle","reason":"no_pending_experiments"}`, so this automation run did not launch a new experiment.
3. Queue manifest still has stale top-level `status` = `pending`, but the actual experiment list is exhausted: `72` completed, `0` pending, `0` failed.
4. The newest ledger rows remain the January 2026 `structure_vp_tree_breakout_failure_6h` control-comparison entries on `spot_jan_2026`:
   - On `vah_rejection_only` / `vah_rejection_orderbook_present` rows (`76` rows), event-row `roc_auc` was about `0.5955` and `average_precision` about `0.2739`.
   - Versus `price_tree_breakout_failure_6h`, lift stayed negative (`roc_auc_delta_vs_control` about `-0.0924`, `average_precision_delta_vs_control` about `-0.1090`), so the structure/VP feature family still trails the price baseline on this exhausted queue's latest scored slice.

Bybit orderbook build hygiene update on 2026-05-27 at 16:00 +01:00:

1. Latest FreqAI queue remains exhausted: `72` completed, `0` pending, `0` failed. No new FreqAI queue was launched because the new Bybit linear/inverse feature parquet files are not validated yet.
2. Bybit inverse raw archive download has completed to `1224` ZIPs, about `121.3 GB`, newest file `2026-05-25_BTCUSD_ob200.data.zip`.
3. Found and stopped a duplicate Bybit linear feature-builder process group that was writing to the same linear output parquet as the configured build. The original configured build remains active with explicit `--market-key bybit_linear --market-id 2.0`.
4. Found an inverse feature-builder process using default spot metadata. Stopped it and relaunched the inverse feature build with explicit `--market-key bybit_inverse --market-id 3.0`.
5. Next safe step is to validate the linear and inverse parquet outputs once the builders finish, then create a new hypothesis-led queue comparing spot, linear, inverse, and price/context controls.

Current-objectives review update on 2026-05-27:

1. Parallel objective reviews were started for:
   - FreqAI ledger meta-analysis.
   - Event-label quality and leakage/timestamp review.
   - News/context feature audit.
   - Custom-indicator prebuild and next-queue design.
2. Ledger meta-analysis:
   - Strongest repeatable target remains VAH rejection / breakout failure.
   - `price_tree_breakout_failure_6h` on monthly VAH rows: `9/9` windows above AUC `0.55`, mean AUC about `0.6860`, best `spot_nov_2025` AUC `0.8750`.
   - Cleanest breakout-acceptance setup is `price_ridge_breakout_success_6h`: `9/9` windows above AUC `0.55`, mean AUC about `0.6596`, best `spot_jan_2026` AUC `0.8394`.
   - Crash/breakdown detection remains promising but less stable.
   - Fakeout labels, context-only, and generic future-return/drawdown tests remain weak/noisy.
3. Ledger/scoring caveat:
   - `_only` and `_orderbook_present` scopes often have identical metrics, so orderbook-present scoring must be validated before relying on orderbook lift claims.
4. Label/timestamp review:
   - Current event labels are better than plain returns, but many are still future path/outcome proxies rather than strict current-setup labels.
   - Context feature DB validation found `0` `max_source_available_at > date` violations across `56,041` rows.
   - Strategy context merge still uses backward asof without a max-age tolerance, which can carry stale context if timeranges exceed feature coverage.
   - Orderbook exact timestamp merge avoids stale carry, but missing values become `0.0`; downstream tests must enforce `orderbook_present` plus coverage gating.
5. Context audit:
   - Current context builder partially encodes severity, persistence, source confluence, topic escalation, and macro/geopolitical/liquidity channels.
   - Missing pieces: explicit first mention/follow-through features, compact macro/liquidity state features, and rebuilt GKG coverage.
   - Recommended compact features include `topic_first_mention_flag_24h`, `topic_escalation_score`, `severity_confluence_score`, `macro_liquidity_shock`, `risk_off_cross_market_confirmation`, and GDELT/GKG coverage-quality flags.
6. Indicator/prebuild audit:
   - Existing structural cache already prebuilds VP, TLV2, BOS/CHoCH, volume-pressure, and range-position into `st_*` columns.
   - `btc_structural_features_1h_latest.parquet` has about `9073` rows and `252` columns, covering `2025-04-29` through `2026-05-12`; refresh/coverage extension is needed before full structure/VP queues.
7. Next gated work:
   - Validate/fix orderbook-present scoring.
   - Add/enforce orderbook coverage thresholds.
   - Add context max-age preflight checks.
   - Refresh/validate structural cache.
   - Repair/rebuild GKG/context before final context-only judgment.
   - Prepare but do not launch linear/inverse Bybit queues until parquet outputs validate.

Current-objectives implementation update on 2026-05-27:

1. Added pre-queue guardrails:
   - Queue creation now writes `preflight_audit.json`.
   - Preflight validates required feature files for duplicate dates, gaps, source timestamp violations, window coverage, and usable orderbook coverage.
   - Preflight only checks the windows that each selected profile can actually run, so mixed structure/orderbook queues no longer fail orderbook files against irrelevant broad windows.
   - Queue experiment folders now use short `exp_0001` style paths, and FreqAI runtime identifiers are short queue/experiment slugs.
2. Added safer feature gating:
   - Context features are merged with a `48h` max-age tolerance and feature age/present/missing flags.
   - Orderbook presence now requires `obts_feature_present > 0` plus `obts_coverage_ratio >= 0.50` when coverage exists.
   - Direct structure/orderbook report loading applies the same orderbook coverage rule.
3. Added target balance reporting:
   - Preflight now reports event rows, label rows, positive rows, negative rows, and event rate per profile/window.
4. Added current-setup labels:
   - New labels include `breakout_failure_from_current_setup_6h`, `breakout_success_from_current_setup_6h`, `breakdown_success_from_current_setup_6h`, and `fakeout_from_current_setup_24h`.
   - Setup-scoped profile targets were added alongside the old future-path labels.
   - Legacy VAH confluence and setup-scoped VAH confluence now use separate prediction columns to avoid target/scoring mismatch.
5. Review fixes applied:
   - Fixed over-broad preflight window checks.
   - Fixed Windows path-length risk in generated queues.
   - Added usable orderbook coverage per window.
   - Scoped class-level strategy feature caches by class/file/profile/pair.
   - Lazy-loaded the work-in-progress orderbook helper only when live DB orderbook features are requested.
   - Moved queue-runner scoring imports into the post-backtest scoring path.
6. Validation:
   - Python compile checks passed for touched scripts.
   - Price-only queue smoke creation passed.
   - Structure stale-window smoke correctly failed where structural cache ended before requested timerange.
   - Mixed structure/orderbook queue smoke passed with relevant file/window auditing.
   - Setup-scoped VAH target balance smoke passed with both positive and negative labels.
7. Still blocked before broad new FreqAI queues:
   - Bybit linear/inverse feature parquet outputs are still not complete/validated.
   - Structural cache needs refresh through the intended queue end date.
   - GKG/context rebuild remains open before final context-only judgment.

GDELT/GKG historical repair update on 2026-05-27:

1. Investigation found GDELT event aggregates are loaded through `2026-05-12T23:00:00+00:00`, but GKG document/theme rows were effectively stalled after `2022-07-28T00:45:00+00:00`.
2. Root cause: `gkg_backfill.py` treated any existing `gdelt_gkg_file_features` row as complete, including rows with `parse_error` set. Repeated connection-refused failures were therefore advancing `--next-missing-chunk-days` into 2023 without usable documents.
3. Code fix started:
   - `--skip-existing` and `--next-missing-chunk-days` now count only `parse_error IS NULL` rows as complete.
   - Added `--retry-failed-only` for controlled repair of already-failed GKG rows.
   - Raised CSV parser field-size limit for large GKG records that previously failed with `field larger than field limit (131072)`.
   - Follow-up review fixed dry-run planning for `--skip-existing` / `--retry-failed-only`, moved `--limit-files` after completion filtering, and ensured direct parser calls also apply the larger CSV field-size limit.
4. Validation:
   - Focused regression tests passed after review additions: `6 passed`.
   - Retried `2022-07-28T01:00:00+00:00` to `2022-07-28T02:00:00+00:00`: `4/4` files written, `4645` documents, `0` failures.
   - Retried known large-field row `2022-07-27T13:15:00+00:00`: `1/1` file written, `1548` documents, `0` failures.
   - Extended live retry through `2022-07-28T07:45:00+00:00`: additional `24/24` files written, `22938` documents, `0` failures.
   - Completed the rest of `2022-07-28`: additional `64/64` files written, `99521` documents, `0` failures.
   - Builder sanity check for `2022-07-28` loaded `96` repaired GKG rows, `132216` source documents, and produced nonzero hourly GKG feature rows.
5. Current GKG DB state after repair start:
   - Successful GKG rows: `45593`.
   - Failed GKG rows remaining: `75467`.
   - Latest successful GKG row: `2022-07-28T23:45:00+00:00`.
   - Next missing/failed cursor from the repaired segment: `2022-07-29T00:00:00+00:00`.
6. Remaining work:
   - Continue GKG retry from `2022-07-29T00:00:00+00:00` through 2026.
   - After GKG source rows are repaired, rebuild `context_features.sqlite`; current feature DB still has stale/near-empty GKG contribution.

Context/FreqAI queue hygiene update on 2026-05-27 late:

1. Context queued research now uses a parquet snapshot instead of reading live SQLite during FreqAI:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\exports\context_features_1h_latest.parquet`.
   - Snapshot rows: `56,041`.
   - Snapshot columns: `409`.
   - Date range: `2020-01-01T00:00:00+00:00` to `2026-05-24T00:00:00+00:00`.
   - Duplicate dates: `0`.
   - Source timestamp lookahead rows: `0`.
2. Review found and fixed a context freshness flaw:
   - Before fix, `%-ctx_feature_present` and `%-ctx_feature_age_hours` were based on the hourly feature row timestamp.
   - After fix, presence and age are based on `max_source_available_at`; null, future, or older-than-`48h` source timestamps are treated as missing/stale.
3. Preflight now audits source availability in parquet caches:
   - Checks `source_max_ts` or `max_source_available_at`.
   - Reports null source timestamps.
   - Reports window-scoped source age over `48h`.
4. News/context acceleration window logic was clarified:
   - `_window_between()` now uses explicit `earlier, later` parameters and rejects reversed ranges.
   - `news_volume_acceleration_6h` and related family/topic acceleration features now compare `(hour-6h, hour]` against `(hour-12h, hour-6h]` with less ambiguous code.
5. A fresh hypothesis-led queue was launched after those fixes:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_232436_720772\freqai_experiment_queue.json`.
   - Experiments: `84`.
   - Target families: current-setup breakout failure, breakout success, and breakdown/crash continuation.
   - Feature families: price, spot orderbook, context-only, and price+context.
   - Preflight errors: `0`.
   - Initial runner status: `1` completed, `83` pending.
6. Known-event validation windows were expanded:
   - Added Evergrande/property stress, Ukraine invasion, Fed hiking-cycle start, Celsius freeze, Fed 75bp hike, SVB/bank stress sources, Country Garden/property stress, 2025 Israel/Iran escalation, and 2025 AI valuation-risk warning.
   - These are for sanity checking context features and predictions against known market-moving periods, not trading rules.

Clean tree setup queue result snapshot on 2026-05-28:

1. Completed queue:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260527_232436_720772\freqai_experiment_queue.json`
   - Status: `84/84` completed, `0` failed.
   - Profiles: price, spot orderbook, context-only, and price+context.
   - Targets: current-setup breakout failure, breakout success, and breakdown/crash continuation.
2. Strongest event-row results with at least `50` rows:
   - `price_tree_breakout_failure_setup_6h`, `spot_oct_2025`, `vah_rejection_only`: AUC about `0.768`, AP about `0.261`, `62` rows.
   - `price_context_tree_breakout_failure_setup_6h`, `spot_dec_2025`, `vah_rejection_only`: AUC about `0.765`, AP about `0.314`, `87` rows.
   - `price_context_tree_breakdown_success_setup_6h`, `spot_nov_2025`, `crash_detection_only`: AUC about `0.728`, AP about `0.571`, `74` rows.
   - `price_tree_breakdown_success_setup_6h`, `spot_nov_2025`, `crash_detection_only`: AUC about `0.712`, AP about `0.561`, `74` rows.
3. Best positive lift versus price controls:
   - `context_tree_breakout_success_setup_6h`, `spot_jan_2026`, all predicted rows: AUC lift about `+0.362`, AP lift about `+0.153`.
   - `context_tree_breakout_success_setup_6h`, `spot_jan_2026`, breakout-acceptance rows: AUC lift about `+0.204`, AP lift about `+0.089`.
   - `context_tree_breakout_failure_setup_6h`, `spot_feb_2026`, VAH rejection rows: AUC lift about `+0.214`, AP lift about `+0.100`.
4. Main negative findings:
   - Context features were harmful on several breakout-failure slices, especially `spot_oct_2025` and `spot_dec_2025`.
   - Spot orderbook did not materially improve crash-detection event rows in the strongest November slice; it trailed the price baseline there.
   - Several breakdown/crash slices have low event-row counts, so high AUC on `spot_apr_2026` should be treated as exploratory only.
5. Next queue launched:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_004256_909465\freqai_experiment_queue.json`
   - Purpose: repeat the same current-setup hypotheses with Ridge to check whether signals are robust/simple or mainly tree-threshold behaviour.
   - Preflight: `84` experiments, `0` errors, `14` warnings.

Ridge setup queue result snapshot on 2026-05-28:

1. Completed queue:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_004256_909465\freqai_experiment_queue.json`
   - Status: `84/84` completed, `0` failed.
   - Purpose: repeat the same current-setup hypotheses with a simple Ridge model.
2. Strongest event-row results with at least `50` rows:
   - `orderbook_ridge_breakout_failure_setup_6h`, `spot_oct_2025`, VAH rejection rows: AUC about `0.866`, AP about `0.371`, `62` rows.
   - `context_ridge_breakout_success_setup_6h`, `spot_nov_2025`, breakout-acceptance rows: AUC about `0.684`, AP about `0.492`, `81` rows.
   - `orderbook_ridge_breakout_failure_setup_6h`, `spot_dec_2025`, VAH rejection rows: AUC about `0.658`, AP about `0.174`, `87` rows.
3. Best positive lifts versus price controls:
   - `context_ridge_breakout_success_setup_6h`, `spot_nov_2025`, breakout-acceptance rows: AUC lift about `+0.236`, AP lift about `+0.157`.
   - `orderbook_ridge_breakout_failure_setup_6h`, `spot_oct_2025`, VAH rejection rows: AUC lift about `+0.176`, but AP lift was negative, so this is not a clean win.
   - `orderbook_ridge_breakout_failure_setup_6h`, `spot_jan_2026`, all predicted rows: AUC lift about `+0.205`, AP lift about `+0.062`.
4. Main negative findings:
   - Ridge context breakout-failure was badly harmful in `spot_oct_2025`, with VAH rejection AUC lift about `-0.345`.
   - Ridge price+context crash detection was harmful in `spot_nov_2025`.
   - Ridge orderbook crash detection underperformed price in January and April event rows.
5. Interpretation:
   - Some context/orderbook effects survive a simple model, but not consistently across months.
   - Strongest simple-model candidate is still event-scoped, especially VAH rejection / breakout failure and breakout acceptance.
   - Orderbook/context should be treated as conditional modifiers, not universally useful inputs.
6. Next queue launched:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_014316_660551\freqai_experiment_queue.json`
   - Purpose: bring custom structure/VP back into the comparison using validated structural parquet and spot orderbook parquet.
   - Profiles: structure/VP tree, structure/VP + spot orderbook tree, structure/VP Ridge, structure/VP + spot orderbook Ridge, plus current-setup VAH variants.
   - Preflight: `21` experiments, `0` errors, `1` warning for a single low-coverage spot orderbook row.

Structure/VP confluence queue snapshot on 2026-05-28:

1. Queue:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260528_014316_660551\freqai_experiment_queue.json`
   - Final status: `15` completed, `6` failed, `0` pending.
2. Failed experiments:
   - All failures were structure+spot-orderbook confluence on `ob_oct` and `ob_jan_feb`.
   - Failure mode: FreqAI/datasieve raised `AttributeError: 'Pipeline' object has no attribute 'features_in'` during prediction transform.
   - Completed `ob_jul_aug` structure+orderbook runs were usable, so the failure appears data/window/model-pipeline specific rather than a missing file.
3. Useful completed results:
   - `structure_vp_ridge`, `oct`, VAH rejection rows: AUC about `0.677`, AP about `0.418`, `101` rows.
   - `structure_vp_orderbook_ridge`, `ob_jul_aug`, all predicted rows: AUC about `0.717`, AP about `0.226`, `408` rows.
   - `structure_vp_orderbook_tree`, `ob_jul_aug`, VAH rejection rows: AUC about `0.553`, AP about `0.180`, `80` rows.
4. Interpretation:
   - Structure/VP has some usable VAH rejection signal.
   - Spot orderbook + structure/VP is not yet stable enough to trust because later orderbook windows hit a FreqAI pipeline failure.
   - Do not treat failed `ob_oct` / `ob_jan_feb` structure+orderbook comparisons as negative model evidence; they are invalid runs until the pipeline failure is root-caused.

Bybit raw archive retention cleanup check on 2026-05-27 at 21:28 +01:00:

1. Retention cleanup was skipped before `py_compile`, dry-runs, or any delete-capable command because Bybit sync/build activity still appears active.
2. Evidence collected during the guard check:
   - Log `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_inverse_trader_state_feature_build_20260527_165821.out.log` was created at `2026-05-27 16:58:21 +01:00`.
   - Log `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_inverse_trader_state_feature_build_20260527_165821.err.log` was created at `2026-05-27 16:58:21 +01:00`.
   - Log `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_linear_inverse_feature_build_after_sync_20260526_221543.err.log` was last written at `2026-05-27 16:58:19 +01:00`.
   - Python processes were still present with start times matching the recent build launches: two Python processes started at `2026-05-27 16:58:21 +01:00` and two more at `2026-05-27 11:28:34 +01:00`.
3. Because the cleanup guard failed on active build evidence, no market validation was attempted for spot, linear, or inverse.
4. Deleted raw ZIP count: `0`.
5. Reclaimed bytes: `0`.
6. Skipped markets: `spot`, `linear`, `inverse`.
7. Validation blockers:
   - Active or recently-active Bybit feature-build chain detected from log timestamps and matching Python process start times.
   - Per retention rule, raw ZIPs must not be deleted while sync or feature-build logs are still active.

Bybit historical raw archive location update on 2026-05-29:

1. Reason:
   - `C:` was near full and the only remaining Bybit historical raw ZIPs on `C:` were the spot archives.
   - Historical Bybit raw downloads had completed to Bybit's available archive end with no failed files.
2. Final raw locations:
   - Spot raw ZIPs: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\spot_raw`.
   - Linear raw ZIPs: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`.
   - Inverse raw ZIPs: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\inverse_raw`.
3. Files left on `C:`:
   - Feature parquet caches, logs, and manifests remain under `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit`.
4. Validation:
   - Spot raw move copied and verified `392` ZIP files, about `25.81` GiB, before deleting the old `C:` raw folder.
   - A follow-up sync found and downloaded three newly available files for each market, bringing spot to `395` files and linear/inverse to `1227` files each through Bybit's `2026-05-28` available end.
   - No directory junction was created.
5. Code/default path updates:
   - Spot raw defaults in the downloader, historical import, raw-retention validator, and trader-state feature builder now point to the D: `spot_raw` directory.
   - The spot manifest `raw_dir` field was updated to the D: `spot_raw` directory.

GDELT/GKG terminal missing-file planner fix on 2026-05-28:

1. Problem:
   - The automated GKG loop repeatedly reprocessed `2020-01-20T04:00:00+00:00` to `2020-02-03T04:00:00+00:00`.
   - All files except `2020-01-20T04:00:00+00:00` were successful; that single row returned `HTTP 404`.
   - Because `--next-missing-chunk-days` treated every non-success row as retryable, the sweep could not advance past a confirmed missing source archive.
2. Fix:
   - `gkg_backfill.py` now treats terminal missing-source errors, currently `HTTP 404`, as complete for chunk planning only.
   - The row remains stored with `parse_error`; `_load_gkg_context()` still filters it out, so it does not create fake document counts or fake features.
   - Hourly GKG completeness remains sparse because successful file counts are still calculated from `parse_error IS NULL` rows only.
3. Validation:
   - Focused tests: `.\.venv\Scripts\python.exe -m pytest tests\research\test_gkg_backfill.py -q` => `8 passed`.
   - Compile check: `.\.venv\Scripts\python.exe -m py_compile user_data\Custom_Launcher\research\context_features\gkg_backfill.py` passed.
   - Real DB dry-run after the fix advanced the next window to `2020-02-03T04:00:00+00:00` through `2020-02-17T04:00:00+00:00`, confirming the stuck `HTTP 404` no longer blocks forward progress.
4. Sequencing note:
   - Full historical sweep should prioritize forward progress over repeated retries of terminal missing-source rows.
   - Retryable failures such as connection refusals should still be retried, while terminal source gaps should be audited after the sweep as explicit coverage gaps.

Live news/web source repair on 2026-05-28:

1. Problem:
   - Live news showed repeated `HTTP 429` and non-JSON failures from six `gdelt_doc` sources using `https://api.gdeltproject.org/api/v2/doc/doc`.
   - Live news also had dead or non-feed URLs for ECB RSS, CFTC RSS, and ONS release calendar.
   - Live web had dead URLs for Cointelegraph RSS and Kraken asset listings.
2. Source changes:
   - Replaced ECB URLs with `https://www.ecb.europa.eu/rss/press.html` and `https://www.ecb.europa.eu/rss/statpress.html`.
   - Replaced CFTC URLs with `https://www.cftc.gov/RSS/RSSGP/rssgp.xml`, `https://www.cftc.gov/RSS/RSSENF/rssenf.xml`, and `https://www.cftc.gov/RSS/RSSST/rssst.xml`.
   - Replaced ONS release calendar with its RSS endpoint query: `https://www.ons.gov.uk/releasecalendar?highlight=true&limit=10&page=1&release-type=type-published&rss=&sort=date-newest`.
   - Replaced Cointelegraph RSS with `https://cointelegraph.com/rss`.
   - Replaced Kraken asset listings with `https://blog.kraken.com/category/product/asset-listings`.
   - Disabled the six live `gdelt_doc` sources because dedicated GDELT/GKG historical backfill is the right GDELT path until a throttled live DOC collector exists.
3. Collector fix:
   - News/web collectors now sync disabled config sources into source health, so disabled sources do not remain as stale active/error rows in SQLite.
4. Validation:
   - One-shot news collector returned zero source errors; ECB, CFTC, and ONS sources returned HTTP 200.
   - One-shot web collector returned zero source errors; Cointelegraph and Kraken returned HTTP 200.
   - Live news/web collectors were restarted and showed `last_error: null`; the live news DB had six disabled GDELT rows with no stale errors, and all checked replacement URLs had HTTP 200.

Bybit raw archive retention cleanup check on 2026-05-28 at 20:29 +01:00:

1. Retention cleanup was skipped before `py_compile`, dry-runs, or any delete-capable command because Bybit sync/build activity still appears active.
2. Evidence collected during the guard check:
   - Log `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_linear_trader_state_feature_build_20260527_112834.out.log` was last written at `2026-05-28 11:14:38 +01:00`.
   - Log `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\logs\bybit_inverse_trader_state_feature_build_20260527_165821.out.log` was last written at `2026-05-28 11:19:05 +01:00`.
   - No `.tmp` files were present under `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw`, `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`, or `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\inverse_raw`.
   - Running Python processes were still present with start times `2026-05-28 09:55:03 +01:00`, `2026-05-28 10:19:16 +01:00`, and `2026-05-28 20:12:48 +01:00`.
3. Observed build state from current log tails:
   - Linear build log reports `files_processed: 1224`, `sample_rows_1m: 1762024`, `zone_rows_5m: 4731548`, and `feature_rows_1h: 29370`, ending at `2026-05-26 01:00:00+00:00`.
   - Inverse build log reports `files_processed: 1224`, `sample_rows_1m: 1762024`, `zone_rows_5m: 7188054`, and `feature_rows_1h: 29370`, ending at `2026-05-26 01:00:00+00:00`.
4. Because the cleanup guard failed on still-active build evidence, no market validation was attempted for spot, linear, or inverse.
5. Deleted raw ZIP count: `0`.
6. Reclaimed bytes: `0`.
7. Skipped markets: `spot`, `linear`, `inverse`.
8. Validation blockers:
   - Active or recently-active Bybit feature-build chain detected from same-day log writes and live Python workers.
   - Per retention rule, raw ZIPs must not be deleted while sync or feature-build logs are still active.

Trader-confluence foundation on 2026-05-29:

1. Purpose:
   - Move from opaque one-off FreqAI tests toward named trader hypotheses with source-present validation, controls, ablations, and readable outputs.
   - Keep objectives in `ai_guidance_docs/current_objectives.md` untouched; record only progress/results here and in `objectives_progress.md`.
2. New implementation files:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_hypotheses.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_features.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_snapshot.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_column_dictionary.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_direct_tests.py`
3. Generated artifacts:
   - Snapshot: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Validation: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h.validation.json`
   - Column dictionary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.csv`
   - Direct-test summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_trader_confluence_kickoff_reviewed_summary.csv`
   - Model-ablation details: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_trader_confluence_kickoff_fixed_ablation_model_ablations.csv`
4. Snapshot validation:
   - Output rows: `55,971`.
   - Output columns: `1,408`.
   - Duplicate dates: `0`.
   - Base hourly gaps: `15`; tests must not assume continuous candles through these gaps.
   - Structure-present rows: `9,073`.
   - Context-present rows: `54,579`.
   - Spot orderbook-present rows: `9,320`.
   - Bybit linear-present rows: `29,280`.
   - Bybit inverse-present rows: `29,280`.
   - Structure + context + orderbook overlap rows: `8,653`.
   - Low-coverage Bybit linear/inverse rows were excluded from present flags.
5. First direct-test watchlist:
   - `orderbook_support_removal_breakdown -> breakdown_success_next_6h`: AUC about `0.681`, AP about `0.311`, trigger event rate about `0.610`, same-regime-without-trigger event rate about `0.111`, random-control event rate about `0.140`. This is currently the cleanest trader-readable breakdown signal.
   - `orderbook_resistance_evaporation_breakout -> breakout_success_next_6h`: AUC about `0.627`, AP about `0.247`, trigger event rate about `0.459`, same-regime event rate about `0.128`, random-control event rate about `0.115`. This is the cleanest breakout/orderbook signal.
   - `context_pressure_plus_structure_break -> breakout_success_next_6h`: AUC about `0.576`, trigger event rate about `0.283`, same-regime event rate about `0.109`. Worth investigating as a context/structure confluence concept.
   - `context_pressure_plus_structure_break -> breakdown_success_next_6h`: AUC about `0.571`, trigger event rate about `0.284`, same-regime event rate about `0.089`. Worth investigating.
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_24h`: AUC about `0.614`, trigger event rate about `0.219`, same-regime event rate about `0.100`. Monthly stability is weaker than the headline score.
   - LVN fast-travel up/down path metrics showed directional lift, but row counts are small and need focused follow-up before modelling.
6. Negative or weak findings:
   - Failed breakout/breakdown exhaustion masks are still poor in this first implementation. They likely need better volume-fade, acceptance/reclaim, and post-break support/resistance rebuild definitions.
   - Multi-timeframe resistance/support rejection/bounce is not yet strong enough as encoded.
   - Multi-venue orderbook agreement has broad coverage but weak direct score lift; it may be useful as a filter rather than a standalone signal.
7. Model-ablation caution:
   - Fixed model ablations excluded raw OHLC and absolute level prices, but full feature sets still often underperformed compact price/structure baselines.
   - Treat broad all-feature FreqAI as high overfit risk.
   - Next FreqAI work should promote only compact named hypotheses and compare against price/structure controls.
8. Open issues:
   - Structural cache needs extension over the Bybit 2023-2026 range before full historical confluence can be tested.
   - GKG/context backfill and rebuild are still needed before treating rich news/context as final.
   - Direct-test trigger thresholds may be too strict for some concepts; several triggers have very small row counts.
   - Review agents are currently checking code/data safety and logic/objective alignment.

Trader-confluence strict review pass on 2026-05-29:

1. Status of earlier loose-gate results:
   - Treat the `trader_confluence_kickoff_reviewed` and `trader_confluence_kickoff_fixed_ablation` watchlists as exploratory only.
   - Review found that gates could pass on one strong component, watchlist control lift used absolute rather than directional comparison, diagnostic metadata could enter models, and the report was too opaque.
2. Fixes applied:
   - Named hypotheses now use component-count gates as well as mean component scores.
   - Historical `px_*` features are masked across base timeline gaps.
   - Watchlist binary checks now require trigger event rate to beat same-regime and random controls directionally.
   - Direct summaries include a price/structure baseline score and watchlist requires the confluence score to beat that baseline.
   - Diagnostic source coverage/freshness/missing/stale metadata is excluded from model features.
   - `minus_volume_pressure` now removes volume-dependent components/composites without deleting unrelated orderbook or context pressure fields.
   - Markdown reports list setup, trigger, and score component columns for each hypothesis.
3. Strict artifacts:
   - Snapshot: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Validation: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h.validation.json`
   - Strict report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_trader_confluence_review_fixed_strict_report.md`
   - Strict summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_trader_confluence_review_fixed_strict_summary.csv`
   - Strict focused ablation: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_trader_confluence_review_fixed_strict_ablation_model_ablations.csv`
4. Strict validation:
   - Rows: `55,971`; columns: `1,618`.
   - Duplicate dates: `0`.
   - Source future violations: `0`.
   - Inactive context confluence source signals: `0`.
   - Inactive orderbook confluence source signals: `0`.
   - Remaining warnings are expected: base hourly gaps and three low-coverage Bybit linear/inverse rows excluded from present flags.
5. Current strict watchlist:
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`.
   - Rows: `4,750`; trigger rows: `127`.
   - Trigger event rate: `0.079`; same-regime event rate: `0.042`; random event rate: `0.024`.
   - Direct score AUC: `0.650`; price/structure baseline AUC: `0.606`; shuffled AUC: `0.512`.
   - Top-bucket lift: `0.059`; baseline top-bucket lift: `0.033`.
   - Monthly AUC windows above `0.55`: `7/8`.
6. Focused ablation for the strict watchlist:
   - `confluence_only`: mean AUC `0.606`, positive windows `6/8`.
   - `price`: mean AUC `0.599`, positive windows `5/8`.
   - `only_ob_inverse`: mean AUC `0.581`, positive windows `3/8`.
   - `orderbook`, `all_venues_no_composites`, and `minus_context`: mean AUC about `0.570`, positive windows `5/8`.
   - `context` and `minus_orderbook`: mean AUC about `0.497`, so context alone is not stable in this window even though the named confluence score uses macro/context stress as a gate.
7. Interpretation:
   - Strict gating confirms one currently useful crash-risk candidate.
   - The earlier orderbook support-removal and resistance-evaporation concepts remain plausible but need threshold calibration because strict gates reduce trigger rows too aggressively.
   - Price/structure baselines are strong and must stay in every report; confluence has to beat them, not just show a standalone AUC above `0.55`.
8. Next required research step:
   - Build threshold calibration / component-sweep tooling for each hypothesis to find settings that keep multi-component confluence while producing enough trigger rows.
   - Then retest the orderbook breakout/breakdown, support bounce, resistance rejection, and context-plus-structure concepts before any FreqAI queue.

Trader-confluence threshold sweep on 2026-05-29:

1. Tool:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_threshold_sweep.py`
   - Purpose: sweep component thresholds, setup thresholds, trigger thresholds, and required component counts without globally weakening the strict confluence gates.
2. Outputs:
   - CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_trader_confluence_review_fixed_strict.csv`
   - Markdown: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_trader_confluence_review_fixed_strict.md`
3. Sweep configuration:
   - Component thresholds: `0.20`, `0.30`, `0.35`, `0.45`.
   - Setup/trigger score thresholds: `0.20`, `0.30`, `0.35`, `0.45`, `0.55`.
   - Minimum setup rows: `100`.
   - Minimum trigger rows: `50`.
   - Passing rule requires trigger event rate to beat same-regime and random controls, score AUC `>= 0.55`, score AUC to beat price/structure baseline by at least `0.01`, and enough positive monthly windows.
4. Results:
   - Sweep rows: `9,857`.
   - Passing candidates: `115`.
   - Every passing candidate is from `macro_context_liquidity_stress_breakdown`.
   - Passing targets are `large_drawdown_next_6h` and `large_drawdown_next_24h`.
5. Best 24h drawdown family:
   - Common setting: component threshold `0.45`, two setup components, two trigger components.
   - Setup rows: `3,145`.
   - Trigger rows: usually `61` to `79`.
   - Trigger event rate: about `0.182` to `0.197`.
   - Same-regime event rate: about `0.131`.
   - AUC: about `0.559`.
   - Price/structure baseline AUC: about `0.486`.
   - AUC lift over baseline: about `+0.074`.
   - Monthly positive windows: `4/7`.
6. Best 6h drawdown family:
   - Common setting: component threshold `0.45`, two setup components, one trigger component.
   - Trigger rows vary from about `156` to `509` depending on trigger threshold.
   - Trigger event rate ranges from about `0.083` to `0.096` in the best passing rows.
   - AUC: about `0.634`.
   - Price/structure baseline AUC: about `0.587`.
   - AUC lift over baseline: about `+0.047`.
   - Monthly positive windows: `7/8`.
7. Negative finding:
   - No current orderbook resistance-evaporation, support-removal, support-bounce, resistance-rejection, LVN travel, multi-venue orderbook, or context-plus-structure-break hypothesis passed the full sweep rules.
   - These should be refined rather than pushed into FreqAI as-is.
8. Research implication:
   - The only immediate FreqAI candidate is a narrow crash-risk / drawdown-risk hypothesis.
   - The broader confluence objective still requires better component definitions for trader behaviours that were expected to matter, especially breakout confirmation, failed breakout, orderbook wall relocation, pressure divergence, and support/resistance acceptance.

GDELT/GKG historical raw archive location update on 2026-05-29:

1. Reason:
   - `C:` was near full and historical GDELT/GKG raw ZIP caches dominated the news data footprint.
   - The extracted SQLite database remains small and stays on `C:`.
2. Final raw locations:
   - GKG raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`.
   - GDELT event export raw ZIPs: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`.
3. Files left on `C:`:
   - SQLite database remains at `C:\FreqTradeStuff\user_data\research_news_data\gdelt\gdelt_context.sqlite`.
   - Derived context feature stores and reports remain under `C:\FreqTradeStuff\user_data\research_news_data\context_features`.
4. Validation:
   - Raw move copied and verified `23,163` files, about `137.49` GiB, before deleting the old `C:\FreqTradeStuff\user_data\research_news_data\gdelt\raw` folder.
   - No directory junction was created.
5. Code/default path updates:
   - `gdelt_backfill.py` now defaults raw event export cache to `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`.
   - `gkg_backfill.py` now defaults raw GKG cache to `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`.
6. Operational note:
   - Active `gkg_backfill` was stopped before the move. It was not automatically restarted because the previous run was looping on bad cached ZIPs in the `2020-07-31` to `2020-08-14` window; restart should be done deliberately after deciding whether to purge/redownload those corrupt raw cache files or run with `--discard-raw`.

Bybit and GDELT/GKG raw path review on 2026-05-29:

1. Scope:
   - Reviewed path references for the raw Bybit historical archives and raw GDELT/GKG historical archives after moving raw ZIPs to `D:`.
   - Feature parquet paths, logs, manifests, SQLite databases, and reports were intentionally left on `C:`.
2. Code/default review result:
   - No runnable code still points to `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw`.
   - No runnable code still points to `C:\FreqTradeStuff\user_data\research_news_data\gdelt\raw`.
   - The remaining old `C:` raw-path mentions are explanatory notes warning future agents not to infer incomplete coverage from the old folders being absent.
3. Bybit manifest refresh:
   - Refreshed `bybit_ob200_raw_manifest.json`, `bybit_linear_ob_raw_manifest.json`, and `bybit_inverse_ob_raw_manifest.json` after the move.
   - Spot, linear, and inverse manifests now report their `raw_dir` fields on `D:` and `missing_files_planned: 0`.
   - Current manifest coverage: spot `395` files through `2026-05-28`; linear `1227` files through `2026-05-28`; inverse `1227` files through `2026-05-28`.
4. Runtime validation:
   - `historical_bybit_raw_sync.py` dry/non-dry no-op checks used the D raw paths and found no missing downloads.
   - `orderbook_trader_state_features.py --dry-run` used `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\spot_raw`.
   - `gkg_backfill.py --dry-run` used `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`.
   - `gdelt_backfill.py --dry-run` used `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\export`.
5. Remaining non-path issue:
   - The Bybit spot retention validator correctly refuses raw deletion because the feature parquet has not yet been rebuilt for the newest raw days: `2026-05-26` has only `1` feature hour, and `2026-05-27` through `2026-05-28` have `0`.

GDELT/GKG restart after raw move on 2026-05-29:

1. Starting state:
   - No `gdelt_backfill.py` or `gkg_backfill.py` process was active after the raw move.
   - GKG still had non-terminal failed rows, led by `52218` connection-refused rows, `876` old CSV field-limit rows, `238` bad cached ZIP rows, and `152` timeout rows.
   - GDELT event export was complete where source files exist through `2026-05-28 23:00 UTC`, with `4763` historical `HTTP 404` source holes.
2. Root-cause cleanup:
   - A focused retry of the first GKG failure window after cache cleanup succeeded for the first `8` files from `2020-07-31 03:30` through `2020-07-31 05:15`, writing `10381` documents with `0` failures.
   - During raw-cache cleanup, the local GKG raw ZIP cache on `D:` was accidentally cleared because the PowerShell ZIP validation path treated every file as invalid. The SQLite extracted rows were not deleted, the GDELT event export raw cache was not affected, and the active GKG loop is rebuilding the GKG raw cache as it fills incomplete periods.
3. Active restart:
   - Started `run_gkg_backfill_loop.ps1`, which repeatedly runs `gkg_backfill.py --next-missing-chunk-days 14` until the requested range is complete.
   - Active range: `2020-01-01` to `2026-05-29`.
   - Active raw path: `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg`.
   - Active log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_backfill_loop_20260529_123455.log`.
4. Event export catch-up:
   - Ran `gdelt_backfill.py` for `2026-05-13` through `2026-05-29`.
   - Result: `384` files written, `423651` event rows read, `0` failures.
   - GDELT event success range now reaches `2026-05-28 23:00 UTC`.
5. Event export source-hole check:
   - Retried failed GDELT event hours with `--fill-failed-quarter-hours --skip-existing`.
   - Result: all `4763` candidates remained `HTTP 404`; treat these as upstream missing source files, not local path/cache errors.

GKG backfill chunk-size restart on 2026-05-29:

1. Restarted the existing GKG loop with larger chunks at the user's request.
2. Active command uses `run_gkg_backfill_loop.ps1 -Start 2020-01-01 -End 2026-05-29 -ChunkDays 30 -MaxWorkers 6 -ProgressEvery 200`.
3. First active 30-day window: `2020-08-14T05:30:00+00:00` to `2020-09-13T05:30:00+00:00`.
4. Planned files for the first chunk: `2880`.
5. Active log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_backfill_loop_20260529_210249.log`.

Source-detail reporting split on 2026-05-29:

1. Reason:
   - User flagged that broad `context` and `structure` labels were too opaque.
   - `context` must be reported by actual source block; `structure` must be reported by custom indicator family rather than blending rare pattern geometry with always-on VP/TLV2/BOS/CHoCH state.
2. Code changes:
   - Added `trader_confluence_feature_taxonomy.py`.
   - Updated `trader_confluence_column_dictionary.py` to include `source_family` and `source_detail`.
   - Updated `trader_confluence_direct_tests.py` to report source-detail groups and add granular model feature sets/ablations.
3. Current source-detail groups include:
   - `context_article_source_activity`, `context_topic_severity`, `context_gdelt_events`, `context_gkg_documents`, `context_google_trends`, `context_btc_etf_flows`, `context_global_market_macro`.
   - `structure_volume_profile`, `structure_tlv2_support_resistance`, `structure_bos_choch_market_structure`, `structure_pattern_geometry`, `structure_cached_price_volume_state`.
4. Regenerated artifacts:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_source_detail_split_report.md`
5. Important coverage correction:
   - Historical context coverage back to 2020 is mostly GDELT event aggregates.
   - Live news/web/global/Google Trends/ETF sources are recent and must not be described as having full 2020-2026 coverage.

Source-detail gated trader-confluence tests on 2026-05-29:

1. Direct-test harness corrections:
   - `trader_confluence_direct_tests.py` now gates each hypothesis by the required source-detail blocks rather than broad `context_present` or any-orderbook eligibility.
   - Walk-forward ablations now train on prior months only and test on the holdout month.
   - Empty/all-null ablation feature sets are skipped.
2. Direct-test run:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_source_detail_gated_20260529_report.md`.
   - Summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_source_detail_gated_20260529_summary.csv`.
   - Model ablations: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_source_detail_gated_20260529_model_ablations.csv`.
   - Detailed rows: `237`; summary rows: `43`.
3. Direct-test result:
   - Watchlist-positive candidate: `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`.
   - Rows: `4747`; trigger rows: `127`; trigger event rate: `0.0787`.
   - Same-regime event rate: `0.0420`; random event rate: `0.0394`.
   - AUC: `0.6497`; baseline AUC: `0.6062`; shuffled AUC: `0.5348`.
   - Monthly AUC positive windows: `7/8`.
4. Threshold/component sweep:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_source_detail_gated_20260529.md`.
   - Rows evaluated: `8077`; passing candidates: `120`.
   - All passing candidates were `macro_context_liquidity_stress_breakdown`.
   - Passing targets: `large_drawdown_next_6h` and `large_drawdown_next_24h`.
   - Strong representative 24h setup: component threshold `0.45`, setup requires `2` components, trigger requires `2` components, setup rows `3142`, trigger rows `71`, trigger event rate `0.1972`, same-regime rate `0.1309`, random rate `0.0704`, AUC `0.5590`, baseline AUC `0.4856`.
   - Strong representative 6h setup: component threshold `0.45`, setup requires `2` components, trigger requires `1` component, setup rows `3142`, trigger rows `156`, trigger event rate `0.0962`, same-regime rate `0.0489`, AUC `0.6339`, baseline AUC `0.5865`.
5. FreqAI preflight:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_freqai_preflight_source_detail_gated_20260529_macro_dd6h.md`.
   - Verdict: `conditional_pilot`.
   - Required source-detail blocks: price OHLCV, VP, TLV2 support/resistance, BOS/CHoCH market structure, spot orderbook, Bybit linear, Bybit inverse, context topic severity, GDELT events, global macro.
   - Usable intersection rows: `8646`, range `2025-04-29T04:00:00+00:00` to `2026-05-12T00:00:00+00:00`.
   - Setup rows: `4747`; trigger rows: `127`; trigger positives: `10`.
6. Interpretation:
   - Corrected testing currently supports one conditional drawdown-risk avenue, not broad full-confluence success.
   - The most promising signal is macro/context risk pressure plus support-break/orderbook/volume confluence.
   - It should be tested as a pilot, with source-detail ablations, before claiming any strategy-level result.

Bybit raw archive retention cleanup on 2026-05-29:

1. Guard status:
   - No currently visible `python` worker matched the Bybit sync/feature-build checks in this session.
   - Latest Bybit historical log writes were on `2026-05-28 11:14:38 +01:00` for linear and `2026-05-28 11:19:05 +01:00` for inverse, so the cleanup proceeded to validation.
   - `historical_bybit_retention.py` passed `py_compile` before any dry-run.
2. Spot dry-run result:
   - Raw dir checked exactly as requested: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\raw`.
   - Feature parquet: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet`.
   - Pattern: `*_BTCUSDT_ob200.data.zip`.
   - Blocker: `No raw archives matched the requested pattern.`
   - Validation details: feature rows `9406`, duplicate dates `0`, source-after-feature rows `0`, all-null numeric columns `0`, `.tmp` files `0`.
3. Linear dry-run result:
   - Raw dir: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\linear_raw`.
   - Feature parquet: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_linear_latest.parquet`.
   - Pattern: `*_BTCUSDT_ob*.data.zip`.
   - Raw archives matched: `1227`, raw bytes retained: `276679706684`.
   - Validation details: duplicate dates `0`, source-after-feature rows `0`, all-null numeric columns `0`, `.tmp` files `0`.
   - Blocker: archive days below the `20`-hour minimum threshold:
     - `2023-01-18`: `18` feature hours
     - `2026-05-26`: `1` feature hour
     - `2026-05-27`: `0` feature hours
     - `2026-05-28`: `0` feature hours
4. Inverse dry-run result:
   - Raw dir: `D:\FreqTradeStuffLargeData\orderbook_data\historical_bybit\inverse_raw`.
   - Feature parquet: `C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_bybit_inverse_latest.parquet`.
   - Pattern: `*_BTCUSD_ob*.data.zip`.
   - Raw archives matched: `1227`, raw bytes retained: `130406624768`.
   - Validation details: duplicate dates `0`, source-after-feature rows `0`, all-null numeric columns `0`, `.tmp` files `0`.
   - Blocker: archive days below the `20`-hour minimum threshold:
     - `2023-01-18`: `18` feature hours
     - `2026-05-26`: `1` feature hour
     - `2026-05-27`: `0` feature hours
     - `2026-05-28`: `0` feature hours
5. Cleanup outcome:
   - Spot skipped.
   - Linear skipped.
   - Inverse skipped.
   - Raw files deleted: `0`.
   - Reclaimed bytes: `0`.
6. Operational read:
   - Spot remains blocked because the requested `C:` raw path currently has no matching archives.
   - Linear and inverse remain blocked because the latest feature parquet coverage still stops before the newest retained raw days.

GDELT/GKG historical rework implementation wave on 2026-05-29:

1. Objective:
   - Build an additive Bronze/Silver/reporting foundation for historical GDELT event export and GKG rows before promoting any new trader-facing features.
   - Keep the active GKG backfill running independently.
2. Added components:
   - `gdelt_gkg_normalized_schema.py` creates raw-file metadata, GDELT event Silver, GKG document Silver, story tables, and a future Gold hourly feature table.
   - `gdelt_event_normalize.py` parses raw `.export.CSV.zip` rows, uses `DATEADDED` as `available_at` with file-stamp fallback, and stores CAMEO-derived fields as weak metadata.
   - `gkg_document_normalize.py` parses raw `.gkg.csv.zip` rows, uses the GKG batch stamp as conservative `available_at`, preserves GKG themes/entities/tone/GCAM/extras as JSON, and stores weak topic/entity candidates.
   - `gdelt_gkg_quality_reports.py` writes coverage, parse-error, source-hole, raw-ZIP validation, and normalized-completeness reports.
3. Integration fixes applied after worker review:
   - GDELT shared-schema insertion now populates the required `available_at` column and normalized topic/impact/direction/severity/raw-reference fields.
   - GKG shared-schema initialization now calls `init_schema(conn)` directly before falling back.
4. Verification:
   - Focused GDELT/GKG tests passed: `17 passed`.
   - GDELT dry-run parsed 5 rows from 1 D-drive raw export ZIP with 0 skipped.
   - GKG dry-run parsed 5 rows from 1 D-drive raw GKG ZIP with 0 skipped.
   - The `implementation_check` quality report showed existing aggregate SQLite coverage for `2020-01-01T00:00:00+00:00` through `2020-01-01T02:00:00+00:00`, but raw ZIP validation for that old window is missing on D because current raw archives are incomplete/rebuilding.
   - Live historical Silver tables are not present yet in the production SQLite; this is expected because production normalization has not been run.
5. Current interpretation:
   - The rework foundation is ready for controlled raw-to-Silver runs after enough raw coverage exists.
   - Existing aggregate GDELT/GKG features remain diagnostic/supporting only, not final trader-readable news extraction.
   - Do not infer that missing raw ZIPs in early 2020 mean extracted aggregate rows are absent; the old aggregate SQLite rows still exist while raw GKG is being rebuilt on D.

GDELT/GKG historical rework second wave on 2026-05-29:

1. Added raw inventory:
   - `gdelt_gkg_raw_inventory.py` scans D-drive GDELT event export and GKG raw ZIP archives.
   - It infers `file_kind` and `gdelt_stamp`, optionally validates ZIP structure, and can upsert `gdelt_raw_files` metadata.
   - A performance defect found during real-data dry-run was fixed: `--limit-files` now applies before optional ZIP validation.
2. Added date-window controls:
   - `gdelt_event_normalize.py` and `gkg_document_normalize.py` now support `--start` and `--end`.
   - Filters are applied by file/batch stamp before parsing ZIP contents.
   - Window semantics are start-inclusive and end-exclusive.
3. Added read-only gates:
   - `gdelt_gkg_quality_gates.py` checks aggregate coverage, source holes, raw metadata, Silver completeness, timestamp safety, and feature DB lookahead.
   - It writes CSV/JSON gate rows and a manifest under the context feature reports directory.
4. Important mapping correction:
   - GKG shared-schema writes now populate `canonical_url` and `url_hash`; without this, the new gate would flag rows produced by the normalizer itself.
5. Verification:
   - Focused test suite: `26 passed`.
   - Limited GDELT event and GKG document raw dry-runs parsed rows successfully from D-drive archives.
   - Limited raw inventory with ZIP validation completed on 4 files and reported all 4 as `zip_valid`.
   - Quality gates on an early 2020 two-hour window produced expected Silver-table failures because production Silver normalization has not been run yet.
6. Current raw GKG progress:
   - At check time the active GKG backfill had `4790` ZIPs / `28.16` GB on D.
   - Highest observed raw GKG filename was `20200919010000.gkg.csv.zip`.

GKG raw-first split on 2026-05-29:

1. Rationale:
   - The combined GKG backfill mixed network download, ZIP/CSV parsing, aggregate extraction, and SQLite writes.
   - For speed and reliability, raw coverage should be completed first, then local processing/conditioning can run later without re-downloading.
2. Implementation:
   - Added `gkg_raw_download.py` as a raw-only downloader.
   - Added `run_gkg_raw_download_loop.ps1` as the long-running loop.
   - The downloader validates newly downloaded ZIP bytes before final rename and writes via temporary files.
   - Download status is tracked in `gkg_raw_download_status`; terminal `HTTP 404` rows are not retried forever.
3. Operational switch:
   - Stopped the old combined `gkg_backfill` process chain.
   - Started raw-only GKG download with 60-day chunks and 10 workers.
   - Active log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_raw_download_loop_20260529_225154.log`.
4. Initial verification:
   - One-hour smoke window downloaded 4 raw ZIP files successfully.
   - Focused tests passed: `24 passed`.
   - At post-start check, raw cache had `6738` ZIPs / `40.07` GB.
   - Highest observed raw filename: `20201014000000.gkg.csv.zip`.
   - Newest observed write from raw-only loop: `20200101061500.gkg.csv.zip`, indicating it was filling earlier gaps from the start of the requested range.
5. Interpretation:
   - Raw coverage is now the priority.
   - Production parsing, Silver normalization, story clustering, topic enrichment, and final feature rebuild should wait until raw coverage is materially complete or intentionally processed in independent later phases.

GDELT/GKG extraction and formatting scripts prepared on 2026-05-29:

1. Added orchestration:
   - `gdelt_gkg_silver_pipeline.py` plans or executes chunked post-download processing.
   - It is plan-only by default; `--execute` is required for production mutation.
   - Phases are `inventory`, `gdelt_events`, `gkg_documents`, `gold_features`, and `quality_gates`.
2. Added first-pass Gold formatting:
   - `gdelt_gkg_gold_features.py` builds `gdelt_gkg_topic_features_1h` from Silver records.
   - It uses `available_at` windows for market-safe alignment.
   - It emits compact trader-readable fields aligned to `news_formatting_objectives.md`, including source diversity, geopolitics intensity, banking confluence, oil first mention, regulation persistence, and macro severity max.
3. Important limitation:
   - Current Gold features use weak Silver topics from CAMEO/GKG parsing.
   - They are explicitly not final classifier/story-enriched signals yet.
   - The metadata says `weak_silver_labels_pending_classifier_enrichment` so later agents should not mistake them for final high-quality news facts.
4. Verification:
   - Plan-only pipeline command generated the expected commands without mutating production data.
   - Focused tests passed: `35 passed`.
5. Operational state:
   - Raw-only GKG downloader remained active while these scripts were prepared.
   - D-drive raw GKG cache reached `7209` ZIPs / `42.29` GB during the check.

Beta trader-confluence hypothesis expansion on 2026-05-29:

1. Reason:
   - The strict source-detail gate previously left only one promoted candidate.
   - The user requested another `30` hypotheses and tests because a single result was too narrow for the intended trader-confluence research.
2. Implementation:
   - Added `31` beta hypotheses, intentionally one more than requested to keep both failed-breakout and failed-breakdown context variants.
   - Code files:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_features.py`
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_hypotheses.py`
   - New families cover structure continuation, support/resistance fakeout, LVN travel, orderbook wall removal, absorption, directional vacuum, multi-venue pressure, context-risk breakdown/rejection, context attention breakout/breakdown, quiet technical breaks, and macro/orderbook pressure flips.
3. Validation:
   - `py_compile` passed for the touched files.
   - Snapshot rebuild completed:
     - Latest parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
     - Rows: `55,971`
     - Columns: `2,096`
   - The rebuild emitted pandas fragmentation warnings because many columns are inserted incrementally. This should be cleaned up later if the feature count keeps growing, but it did not fail validation.
4. Direct beta test:
   - Command tag: `beta_source_detail_gated_20260529`
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_beta_source_detail_gated_20260529_report.md`
   - Summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_beta_source_detail_gated_20260529_summary.csv`
   - Model ablations: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_beta_source_detail_gated_20260529_model_ablations.csv`
   - Detailed rows: `557`
   - Summary rows: `107`
5. Fixed-gate watchlist positives:
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`: trigger rows `127`, trigger event rate `0.0787`, same-regime `0.0420`, random `0.0394`, AUC `0.6497`, baseline AUC `0.6062`, shuffled AUC `0.5348`, monthly positive `7/8`.
   - `beta_context_risk_support_breakdown -> large_drawdown_next_6h`: trigger rows `99`, trigger event rate `0.1010`, same-regime `0.0417`, random `0.0404`, AUC `0.6495`, baseline AUC `0.6064`, shuffled AUC `0.4982`, monthly positive `7/8`.
   - `beta_structure_support_breakdown_continuation -> large_drawdown_next_6h`: trigger rows `96`, trigger event rate `0.0938`, same-regime `0.0536`, random `0.0208`, AUC `0.5632`, baseline AUC `0.5419`, shuffled AUC `0.4882`, monthly positive `5/7`.
6. Notable non-watchlist reads:
   - Several breakdown-success rows had high AUCs, but the price/structure baseline was stronger.
   - Examples include `beta_context_risk_support_breakdown`, `beta_context_attention_breakdown`, `beta_structure_support_breakdown_continuation`, and `beta_ob_downside_vacuum_breakdown` against `breakdown_success_next_6h`.
   - Interpretation: the direction/breakdown-success condition may already be mostly captured by price/structure, while drawdown-risk conditions show better incremental confluence value.
7. Beta threshold/component sweep:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_beta_source_detail_gated_20260529.md`
   - CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_beta_source_detail_gated_20260529.csv`
   - Rows evaluated: `24,455`
   - Passing threshold candidates: `831`
   - Passing hypothesis/target families:
     - `beta_structure_support_breakdown_continuation -> large_drawdown_next_6h`
     - `beta_context_risk_support_breakdown -> large_drawdown_next_6h`
     - `beta_context_attention_breakdown -> large_drawdown_next_24h`
     - `beta_context_risk_support_breakdown -> large_drawdown_next_24h`
     - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`
     - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_24h`
     - `beta_ob_downside_vacuum_breakdown -> large_drawdown_next_6h`
8. Strongest threshold-sweep beta result:
   - `beta_ob_downside_vacuum_breakdown -> large_drawdown_next_6h`
   - Thresholds: component `0.20`, setup `0.20`, trigger `0.30`, setup count `2`, trigger count `1`.
   - Setup rows: `367`
   - Trigger rows: `54`
   - Trigger event rate: `0.1111`
   - Same-regime event rate: `0.0383`
   - Random event rate: `0.0741`
   - AUC: `0.6457`
   - Baseline AUC: `0.5315`
   - AUC minus baseline: `0.1141`
   - Monthly positive: `4/6`
9. Current interpretation:
   - The beta pack did solve the "only one result" problem at the direct-test stage.
   - The strongest new evidence is still mostly downside-risk oriented, especially 6h drawdown risk rather than clean directional breakout success.
   - Orderbook downside vacuum after support removal is now the most interesting orderbook-specific follow-up.
   - Context attention/risk features show more promise for drawdown-risk timing than generic article count.
   - FreqAI should not be launched broadly from all `31` beta hypotheses. Promote only a compact pilot set after preflight:
     - `beta_ob_downside_vacuum_breakdown -> large_drawdown_next_6h`
     - `beta_context_risk_support_breakdown -> large_drawdown_next_6h`
     - `beta_context_attention_breakdown -> large_drawdown_next_24h`
     - `beta_structure_support_breakdown_continuation -> large_drawdown_next_6h`
     - existing `macro_context_liquidity_stress_breakdown` for `6h` and `24h` drawdown risk.

Gamma bullish breakout suite on 2026-05-29:

1. Reason:
   - User clarified that structure-only trading hypotheses should not be treated as inferior merely because they are not news/orderbook confluence.
   - The goal was to add a bullish-market breakout suite and explain results in plain trader language.
2. Implementation:
   - Added `16` gamma bullish hypotheses.
   - Source mix:
     - `9` price/structure-only.
     - `4` price/structure/orderbook.
     - `1` price/structure/context.
     - `2` wider confluence.
   - Code files:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_features.py`
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_hypotheses.py`
3. New state features:
   - `conf_structure_bullish_state_score`: whether custom structure features show bullish market structure across timeframes.
   - `conf_price_bull_trend_regime`: whether price action is already behaving like a bullish regime.
   - `conf_price_compression_24h`: whether recent price action is compressed enough for range expansion to matter.
   - `conf_volume_bullish_impulse_short`: shorter-term bullish volume impulse.
4. Validation:
   - Compile passed.
   - Snapshot rebuild completed:
     - Rows: `55,971`
     - Columns: `2,352`
     - Latest parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Snapshot rebuild still emits pandas fragmentation warnings due incremental column insertion. This is a performance issue to refactor later, not a failed test.
5. Direct test:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_gamma_bull_breakout_20260529_report.md`
   - Summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_gamma_bull_breakout_20260529_summary.csv`
   - Detailed rows: `802`
   - Summary rows: `156`
   - One fixed-gate watchlist row appeared: `gamma_bull_range_expansion_volume -> future_max_upside_24h`. Treat this as exploratory because it is a continuous upside-path lift, not a binary breakout-success result.
6. Threshold sweep:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_gamma_bull_breakout_20260529.md`
   - CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_gamma_bull_breakout_20260529.csv`
   - Rows evaluated: `29,305`
   - Gamma rows evaluated: `4,850`
   - Gamma passing threshold variants: `180`
   - Passing family: `gamma_bull_compression_breakout -> breakout_success_next_6h`.
7. Plain-English result:
   - Good: when price had been compressed/ranging and then broke upward with bullish volume, breakout success over the next 6 hours became much more common.
   - Representative result: `54` trigger rows had `51.9%` breakout success.
   - Similar setup rows without the full trigger had only about `10.9%` breakout success.
   - Random eligible controls had about `13.0%` breakout success.
   - The ranking score was useful across `11/13` monthly windows.
   - The ranking score beat the price/structure baseline: AUC `0.658` versus baseline AUC `0.629`.
8. What did not work yet:
   - Orderbook-confirmed bullish breakout variants did not beat the cleaner structure/volume compression breakout family.
   - Context/news bullish breakout variants did not add clear value in this run.
   - Several bullish structure hypotheses had high absolute breakout-success ranking, but the baseline was slightly stronger, so they are not yet independent improvements.
9. Current interpretation:
   - For bullish market breakout work, structure plus volume currently looks stronger than broad confluence.
   - Confluence should be treated as an optional enhancer, not a requirement.
   - Next step should be a focused FreqAI/direct-test pilot around compression breakout, range expansion, and volume impulse, with orderbook/context tested as ablations rather than mandatory gates.

FreqAI/model artifact storage check on 2026-05-29:

1. Size summary:
   - `C:\FreqTradeStuff\user_data\models`: about `28.2 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs`: about `0.45 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue`: about `0.18 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports`: about `0.09 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache`: about `0.51 GB`.
2. Largest model artifact groups:
   - `context-freqai-structure-lightgbm-20260520`: about `3.53 GB`.
   - `context-freqai-event-combined-lgbm-20260521`: about `3.31 GB`.
   - `context-freqai-event-context-lgbm-20260521`: about `3.31 GB`.
   - `context-freqai-event-price-lgbm-20260521`: about `3.31 GB`.
   - `structure-ob-vah-structure_vp_orderbook_tree-broad-20250701-20260218-v1`: about `2.35 GB`.
   - `structure-ob-vah-structure_vp_tree-broad-20250701-20260218-v1`: about `2.35 GB`.
3. Deletion recommendation:
   - Keep reports, CSV summaries, queue manifests, and confluence parquet cache.
   - Old model directories can likely be deleted if only summary findings are needed.
   - Do not delete them automatically because deletion would prevent rerunning old prediction-level analysis without retraining.

Feature discovery layer V1 on 2026-05-29:

1. Reason:
   - User identified that the research may be starting too far along by hand-defining hypotheses before discovering which values and transforms matter.
   - The new workflow should generate controlled metrics first, screen them, then convert stable discoveries back into trader-readable hypotheses.
2. Added scripts:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\feature_discovery_candidate_builder.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\feature_discovery_screen.py`
3. Candidate builder:
   - Input snapshot: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Output parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_feature_discovery_v1.parquet`
   - Dictionary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_feature_discovery_v1_dictionary.csv`
   - Rows: `55,971`
   - Columns: `2,996`
   - Feature columns: `2,988`
4. Candidate transforms:
   - `level`: current value.
   - `delta_3h`: recent 3h change.
   - `delta_24h`: 24h change.
   - `z_168h`: abnormality versus roughly one week.
   - `pct_rank_720h`: position versus roughly one month.
   - `above_mean_24h`: whether the value is above its 24h mean.
   - `activity_24h` and `activity_72h` for binary flags.
5. Screener:
   - Output combined CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_feature_discovery_v1_fixed_combined.csv`
   - Markdown report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_feature_discovery_v1_fixed.md`
   - Rows scored: `11,761`
   - Stable shortlist rows: `213`
   - Targets screened: breakout success, breakdown success, 6h/24h large drawdown, upside-first path, downside-first path, fakeout.
6. Stability controls used:
   - Top-decile lift versus base rate.
   - Top-decile lift versus deterministic random comparable rows.
   - Shuffled-label control.
   - Monthly stability ratio.
   - Tree-model shortlist check using ExtraTrees on chronological train/test split.
7. Stable shortlist by target:
   - `breakdown_success_next_6h`: `67` rows.
   - `breakout_success_next_6h`: `66` rows.
   - `large_drawdown_next_6h`: `49` rows.
   - `large_drawdown_next_24h`: `31` rows.
8. Stable shortlist by source:
   - `price_ohlcv`: `119` rows.
   - `orderbook_bybit_linear`: `47` rows.
   - `hypothesis_confluence`: `15` rows.
   - `structure_composite`: `14` rows.
   - `orderbook_bybit_inverse`: `14` rows.
   - `orderbook_composite`: `3` rows.
   - `context_composite`: `1` row.
9. Plain-English discoveries:
   - Breakout success: price being high in its recent 24h/72h range is the strongest discovered signal. This means breakout success is mostly explained by price already pressing toward or through the top of the recent range.
   - Breakdown success: price being low in its recent 24h/72h range is the mirror signal. This is strong and stable.
   - Large drawdown risk: low compression score, high 1h range, and high volume percentile are repeatedly associated with larger future drawdowns. Plain English: when the market is already moving wide/heavy, downside risk rises.
   - Bybit orderbook: lower support/resistance zone persistence and stretched wall distances show up as drawdown-risk clues. Plain English: when persistent visible liquidity support is weaker or farther away, downside risk is more likely.
   - Context/news: context did not survive meaningfully in this broad screen. This supports improving the raw context extraction before judging news value.
10. Important caution:
   - This is a discovery screen, not proof.
   - Many top discoveries are highly related versions of the same idea, especially range position and price momentum.
   - The next step is to collapse these into a small number of trader-readable hypotheses before adding more features.

Delta discovery hypothesis pack on 2026-05-31:

1. Purpose:
   - Convert the feature-discovery shortlist into trader-readable hypotheses instead of leaving it as opaque column rankings.
   - Preserve the user objective that good structure-only signals are valid; confluence is only valuable when it improves or explains the edge.
2. Code updated:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_features.py`
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_confluence_hypotheses.py`
3. New/changed states:
   - `conf_price_compression_24h`: now uses historical rolling percentile only, not full-column rank.
   - `conf_price_range_expansion_24h`: high recent range/volatility versus recent history.
   - `conf_range_position_high_24h` / `conf_range_position_low_24h`: whether price is near the top/bottom of the recent 24h range.
   - `conf_range_position_high_stack` / `conf_range_position_low_stack`: 24h/72h stacked range-position pressure.
   - `conf_volume_bearish_impulse_short`: short-horizon sell-volume impulse.
   - `conf_ob_support_persistence_weak`: weak/absent persistent bid support across Bybit venues.
   - `conf_ob_resistance_persistence_weak`: weak/absent persistent ask resistance across Bybit venues.
   - `conf_ob_wall_distance_stretched`: visible wall/support/resistance is relatively far from current price.
4. New hypothesis pack:
   - `delta_range_high_breakout_success`
   - `delta_range_low_breakdown_success`
   - `delta_expansion_drawdown_risk`
   - `delta_compression_breakout_release`
   - `delta_compression_breakdown_release`
   - `delta_orderbook_support_weak_drawdown`
   - `delta_orderbook_resistance_weak_breakout`
   - `delta_structure_orderbook_drawdown_confluence`
   - `delta_structure_orderbook_breakout_confluence`
5. Snapshot rebuild:
   - Output: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Rows: `55,971`
   - Columns: `2,510`
   - Warning: pandas fragmentation warnings remain due component columns being inserted repeatedly. This should be refactored for speed later, but it did not stop the build.
6. Direct test outputs:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_delta_discovery_20260531_report.md`
   - Summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_delta_discovery_20260531_summary.csv`
7. Threshold sweep outputs:
   - Report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_delta_discovery_20260531.md`
   - CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_delta_discovery_20260531.csv`
   - Rows scored: `34,485`
   - Delta rows scored: `5,180`
   - Delta pass-candidate threshold rows: `2,338`
8. Best direct delta results:
   - `delta_structure_orderbook_drawdown_confluence -> breakdown_success_next_6h`: `129` trigger rows, `53.5%` success, versus `9.9%` same-regime rows and `11.6%` random control; AUC `0.755`, shuffled AUC `0.498`, monthly positive `14/14`.
   - `delta_structure_orderbook_breakout_confluence -> breakout_success_next_6h`: `129` trigger rows, `58.9%` success, versus `11.4%` same-regime rows and `10.9%` random control; AUC `0.734`, shuffled AUC `0.499`, monthly positive `13/13`.
   - `delta_compression_breakdown_release -> breakdown_success_next_6h`: `129` trigger rows, `53.5%` success, versus `14.2%` same-regime rows and `7.8%` random control; AUC `0.722`, shuffled AUC `0.498`, monthly positive `13/13`.
   - `delta_compression_breakout_release -> breakout_success_next_6h`: `126` trigger rows, `60.3%` success, versus `17.4%` same-regime rows and `13.5%` random control; AUC `0.705`, shuffled AUC `0.514`, monthly positive `13/13`.
   - `delta_range_low_breakdown_success -> breakdown_success_next_6h`: `127` trigger rows, `53.5%` success, versus `25.0%` same-regime rows and `10.2%` random control; AUC `0.687`, monthly positive `12/12`.
   - `delta_range_high_breakout_success -> breakout_success_next_6h`: `124` trigger rows, `61.3%` success, versus `25.6%` same-regime rows and `8.9%` random control; AUC `0.686`, monthly positive `13/13`.
9. Best threshold-sweep interpretation:
   - `delta_structure_orderbook_breakout_confluence` and `delta_structure_orderbook_drawdown_confluence` are strongest after threshold calibration.
   - They show that orderbook does not need to predict alone; it is more useful when it confirms a range/structure state.
   - Compression breakout/breakdown release remains useful and should stay on the candidate list.
10. Weak/limited results:
   - Standalone orderbook support-weakness and resistance-weakness hypotheses are not strong enough alone.
   - They are better treated as confluence filters on structure/range setups.
   - Context/news still did not contribute materially to this delta pack.
11. Next implementation recommendation:
   - Add stricter delta-trigger scoped FreqAI scoring if FreqAI should be compared directly against the best direct-test trigger rows.
   - Score delta candidates against price/structure baseline and structure+orderbook variants, not broad full-feature blends.
   - Include month/window stability and shuffled/control comparisons in the scorer.

Delta FreqAI first-pass queue on 2026-05-31:

1. Purpose:
   - Check whether the discovery-derived delta feature family helps FreqAI rank breakout/breakdown outcomes beyond price-only controls.
   - This is broader than the direct trigger tests: it asks whether the model can use the delta features across event scopes, not only when the hand-built delta trigger is active.
2. Code added:
   - `ContextTraderConfluenceDeltaFreqAIResearchStrategy` in `C:\FreqTradeStuff\user_data\strategies\ContextFreqAIResearchStrategy.py`
   - `trader_confluence_delta` feature family in `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\feature_profile_registry.py`
3. Queue:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260531_210705_354735\freqai_experiment_queue.json`
   - Completed experiments: `12/12`
   - Windows: `spot_q4_2025`, `spot_q1_2026`, `spot_recent`
   - Model: LightGBM tree, small profile.
4. Breakout-success FreqAI result:
   - Q4 2025 all predicted rows: delta AUC `0.598` vs price-only AUC `0.542`; top bucket success `15.6%` vs price `11.0%`.
   - Q4 2025 breakout-acceptance rows: delta AUC `0.587` vs price-only AUC `0.546`; top bucket success `35.8%` vs price `32.8%`.
   - Q1 2026 breakout-acceptance rows: delta AUC `0.496` vs price-only AUC `0.482`; top bucket was worse (`28.2%` vs `31.0%`), so this is not a clean improvement.
   - Recent breakout-acceptance rows: delta AUC `0.528` vs price-only AUC `0.514`; top bucket improved (`32.9%` vs `26.3%`).
5. Breakdown-success FreqAI result:
   - Q4 2025 crash-detection rows: delta AUC `0.664` vs price-only AUC `0.671`; top bucket equal (`37.8%` vs `37.8%`).
   - Q1 2026 crash-detection rows: delta AUC `0.672` vs price-only AUC `0.690`; top bucket worse (`31.1%` vs `35.6%`).
   - Recent crash-detection rows: delta AUC `0.561` vs price-only AUC `0.633`; top bucket worse (`14.3%` vs `23.8%`).
6. Interpretation:
   - Good: breakout-side delta features have some real FreqAI value, especially Q4 and recent event-row bucket separation.
   - Bad: breakdown-side delta FreqAI did not improve on price-only. The direct-test breakdown triggers may be too sparse/specific for the broad FreqAI event scope, or the feature family is not adding useful ranking information beyond price.
   - Worth investigating: add a scorer scope for exact delta trigger rows, and compare FreqAI only where the delta setup is active. This would better match the direct-test evidence.

GDELT/GKG raw-download hardening and restart on 2026-05-31:

1. Objective:
   - Finish the historical GKG raw archive first, then process/format later.
   - Avoid another stall where a single bad timestamp causes the loop to repeat the same chunk indefinitely.
2. Implemented changes:
   - `gkg_raw_download.py` now persists completed futures incrementally.
   - Worker exceptions become persisted retryable failure rows.
   - `--validate-existing` can detect corrupt existing ZIPs, move them aside as `.corrupt`, and redownload.
   - Status writes now update `gdelt_raw_files` as well as `gkg_raw_download_status`.
   - Raw metadata uses local archive path as the canonical identity so inventory and downloader status converge on the same raw-file row.
   - `--defer-retry-failures-hours` lets the forward sweep skip recent retryable failures temporarily.
   - `run_gkg_raw_download_loop.ps1` defaults to 30-day chunks, 4 workers, and 72-hour retry deferral.
   - `gkg_raw_coverage_report.py` adds JSON/CSV coverage reporting with optional ZIP sample validation.
3. Review findings:
   - Download-critical issues fixed before restart:
     - Existing corrupt ZIPs could previously be treated as complete forever.
     - Download status was disconnected from the unified raw-file table.
     - Retryable failures could stall the loop on the same early timestamp.
   - Downstream issues still open before Silver/Gold features should be trusted:
     - Silver extraction still needs streaming for large chunks.
     - Raw file identifiers should be made consistent across all GDELT/GKG inventory/parser paths.
     - Gold expects some topic families not emitted by the weak parser.
     - Single-topic storage loses multi-topic articles/events.
     - Some quality gates use event/document time rather than availability time.
     - Gold dry-run still creates artifacts and should become read-only.
     - Some raw GKG fields are not carried into Silver.
4. Verification:
   - Focused tests passed: `43 passed`.
   - PowerShell loop syntax parsed successfully.
   - Coverage report succeeded against the real D-drive raw archive.
   - One-file live smoke download succeeded and persisted status.
5. Coverage snapshot before restart:
   - Expected files: `224,640`.
   - Present raw ZIPs: `17,352`.
   - Percent complete: `7.7244%`.
   - Missing raw files: `207,288`.
   - Terminal 404s: `1`.
   - Failed retry statuses: `0`.
   - Estimated remaining size: about `1.40 TB`.
   - Latest coverage report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gkg_raw_coverage_gkg_raw_coverage_restart_check.json`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gkg_raw_coverage_gkg_raw_coverage_restart_check.csv`
6. Active process:
   - Restarted at `2026-05-31 21:49:40 Europe/London`.
   - Loop log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_raw_download_loop_20260531_214940.log`.
   - Running with `MaxWorkers=4`, `ChunkDays=30`, `RetryDeferHours=72`.
   - First chunk after smoke: `2020-04-20T19:45:00+00:00` to `2020-05-20T19:45:00+00:00`, `2,877` planned files.
   - Early post-launch count: `17,409` raw ZIPs and `56` successful active-chunk rows.

Delta FreqAI/scoring refinement on 2026-05-31:

1. Purpose:
   - Fix the mismatch between strong direct trigger tests and broad FreqAI event-scope scoring.
   - Check whether delta features still help when FreqAI receives only component evidence instead of pre-baked setup/trigger/score flags.
   - News/context was intentionally ignored for this pass.
2. Code changes:
   - `ContextTraderConfluenceDeltaComponentsFreqAIResearchStrategy` added as a component-only FreqAI strategy.
   - `trader_confluence_delta_components` profile added to the feature profile registry.
   - Trader-confluence FreqAI merge now emits `%-tc_feature_present` and `%-tc_feature_missing`.
   - Trader-confluence loader now drops duplicate dates before merge.
   - FreqAI scorer now appends exact `conf_delta_*_setup` and `conf_delta_*_trigger` scopes for delta-family profiles.
   - Direct tests now add ingredient-missing controls where component groups exist: structure, volume, and orderbook.
   - Threshold sweep now writes a collapsed pass summary so threshold variants are grouped by hypothesis and target.
3. Sub-agent review:
   - Worker A implemented direct-test controls and collapsed sweep outputs.
   - Reviewer B found no high-severity issues in the FreqAI/profile/scorer changes.
   - Reviewer B flagged duplicate-date handling as a low-risk concern; fixed in the strategy loader.
4. Verification:
   - `py_compile` passed for the edited strategy, registry, scorer, direct-test, and sweep scripts.
   - Focused FreqAI queue completed `12/12` experiments:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260531_214826_680862\freqai_experiment_queue.json`
   - Direct-test rerun completed:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_delta_review_controls_20260531_summary.csv`
     - Added `293` ingredient-missing control rows.
   - Sweep smoke test completed and wrote collapsed summary:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_threshold_sweep_delta_review_sweep_smoke_20260531_collapsed_pass_summary.csv`
5. FreqAI rerun highlights:
   - Q4 2025 breakout acceptance:
     - Price-only AUC `0.546`, top bucket `32.8%`.
     - Delta-with-flags AUC `0.587`, top bucket `35.8%`.
     - Delta-components-only AUC `0.587`, top bucket `38.8%`.
     - Interpretation: useful, and not dependent on aggregate trigger flags.
   - Recent breakout acceptance:
     - Price-only AUC `0.514`, top bucket `26.3%`.
     - Delta-with-flags AUC `0.528`, top bucket `32.9%`.
     - Delta-components-only AUC `0.507`, top bucket `21.1%`.
     - Interpretation: flag/profile signal exists, component-only did not hold up recently.
   - Q4 2025 crash detection:
     - Price-only AUC `0.671`, top bucket `37.8%`.
     - Delta-with-flags AUC `0.664`, top bucket `37.8%`.
     - Delta-components-only AUC `0.678`, top bucket `40.0%`.
     - Interpretation: component-only was slightly better, but margin is small.
   - Recent crash detection:
     - Price-only AUC `0.633`, top bucket `23.8%`.
     - Delta-with-flags AUC `0.561`, top bucket `14.3%`.
     - Delta-components-only AUC `0.570`, top bucket `14.3%`.
     - Interpretation: not useful versus price-only in recent data.
6. Exact trigger-scope examples from the rerun:
   - `delta_orderbook_resistance_weak_breakout_trigger_only` scored well on Q4 breakdown target too: AUC around `0.705` on `182` rows. This may be an inverse-risk/failure-state relationship and needs trader interpretation before promotion.
   - `delta_range_high_breakout_success_trigger_only` scored AUC `0.683` on Q4 breakout target, but only `29` rows.
   - `delta_structure_orderbook_breakout_confluence_trigger_only` scored AUC `0.617` on Q4 breakout target, but only `31` rows.
7. Current interpretation:
   - Strongest practical avenue remains bullish breakout/acceptance around range-high, compression release, and structure/orderbook confluence.
   - Downside/crash FreqAI is still not broadly improved by delta features; use direct trigger tests for downside candidates until stronger evidence appears.
   - Component-only profiles should be preferred for discovery. Flag-inclusive profiles are useful as validation/gating tools only.

GKG compact hourly metadata prototype on 2026-06-02:

1. Purpose:
   - Test whether pre-2023 GKG raw ZIPs can be converted into compact 1h metadata features before deleting old raw files for disk relief.
   - Avoid full Silver materialization because full per-document SQL would not solve the storage problem.
2. Code added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\gkg_hourly_metadata_extract.py`
3. Method:
   - Streams GKG ZIPs directly.
   - Aggregates to 1h feature rows.
   - Keeps coverage, document/source counts, tone, topic counts, severity, source confluence, persistence, first-mention flags, and direction-state scores.
   - Adds audit samples by topic.
   - Joins existing aggregate `gdelt_hourly_features` event columns where available.
4. January 2020 corrected run:
   - Output parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_gkg_hourly_metadata_jan2020_review_v3.parquet`
   - Audit CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_audit_gkg_hourly_metadata_jan2020_review_v3.csv`
   - Summary JSON: `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\gkg_hourly_metadata_summary_gkg_hourly_metadata_jan2020_review_v3.json`
   - Feature rows: `744`.
   - Feature columns: `188`.
   - GKG ZIPs read: `2,975` of `2,976`.
   - Rows streamed: `5,085,793`.
   - Corrupt files: `0`.
   - Parquet size: `0.854 MB`.
   - Raw January 2020 GKG size: `18.57 GB`.
5. Feature distribution notes:
   - GKG document count per hour median: `6,228`; p95: `11,011.85`; max: `12,207`.
   - Unique source count per hour median: `1,478.5`; p95: `2,276.85`.
   - GDELT event count per hour median from existing aggregate table: `1,568`.
   - Corrected v3 topic median docs/hour:
     - `war_geopolitics`: `784`.
     - `banking_credit`: `10`.
     - `oil_energy`: `90`.
     - `regulation`: `1,043.5`.
     - `macro_policy`: `123`.
     - `security_cyber`: `124`.
     - `crypto_market`: `8`.
6. Review findings:
   - First taxonomy pass was too broad.
   - Fixed obvious false positives:
     - `ETHNICITY` matching `eth` for crypto.
     - `DEFICIT` matching `defi`.
     - generic `financial emergency` inflating banking stress.
     - `runner` matching banking `run`.
   - Remaining issue: high-volume GKG topics fire frequently, especially `regulation`, `war_geopolitics`, `oil_energy`, `macro_policy`, and `security_cyber`. These should be interpreted as intensity/regime features after normalization, not binary news-present features.
   - Story clustering/deduplication is still incomplete. URL-level duplicate ratio was `0` because syndicated/reposted stories often have different URLs.
7. Deletion gate:
   - Do not delete 2020-2022 GKG raw yet.
   - The compact format is promising for storage, but the full target window still needs extraction, coverage reporting, audit review, and an explicit deletion manifest.
   - Present deletion opportunity if gates pass:
     - 2020 GKG raw: `201.13 GB`.
     - 2021 GKG raw: `75.22 GB` through `2021-06-03T08:00`.
     - 2022 GKG raw: `0 GB` currently present.

GKG pre-2023 conversion/deletion orchestration update on 2026-06-03:

1. Objective clarified:
   - Process and delete raw GKG archives for 2020-2022 only.
   - Keep 2023-2026 raw GKG archives for later raw-data testing against the historical Bybit orderbook overlap.
   - Do not treat deletion of currently available raw as final completeness; missing intervals must remain documented and become targeted download/extract/delete work.
2. Current active extraction:
   - 2020 full-year compact extraction is running from available raw ZIPs.
   - Active process log: `C:\FreqTradeStuff\user_data\research_news_data\gdelt\logs\gkg_hourly_metadata_extract_2020_20260602_183055.log`.
   - Last observed progress: `28,000` of `32,551` files processed, `46,621,330` streamed rows, `0` corrupt files.
3. Coverage before 2020 extraction:
   - Expected files: `35,136`.
   - Present files: `32,551`.
   - Coverage: `92.6429%`.
   - Missing files: `2,585`.
   - Coverage report path: `C:\FreqTradeStuff\user_data\research_news_data\context_features\news_gkg_hourly\pre2023_orchestrator\coverage_reports\gkg_raw_coverage_year2020_latest.json`.
4. Orchestrator correction:
   - `gkg_pre2023_orchestrator.py` now records missing intervals before deleting available raw.
   - It uses targeted gap windows after deletion instead of running a full-year downloader that would redownload already-processed/deleted files.
   - Gap flow is download exact missing interval, extract exact interval to a separate parquet/summary, then delete only raw files inside that interval.
5. Safety status:
   - Heartbeat `gkg-pre-2023-conversion-monitor` is active again.
   - Deletion is allowed only through `run_gkg_pre2023_orchestrator_tick.ps1`.
   - Raw deletion remains scoped to `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg` and files ending `.gkg.csv.zip`.

Goal feature-definition research pass on 2026-06-05:

1. Objective:
   - Find trader-readable feature definitions that make price/volume, custom indicators, and orderbook more useful for predicting BTC behaviour.
   - News/context was intentionally parked unless a source block was proven ready.
2. Readiness audit:
   - Source coverage report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\source_coverage_audit_goal_feature_research_after_epsilon_20260605.csv`
   - Clean testing windows:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\clean_testing_windows_goal_feature_research_after_epsilon_20260605.csv`
   - Usable structure/orderbook windows:
     - `2025-04-29T04:00:00+00:00` to `2025-08-21T00:00:00+00:00`
     - `2025-08-21T04:00:00+00:00` to `2026-05-12T00:00:00+00:00`
   - Full confluence remains invalid because several context blocks are too sparse.
3. Code/features added:
   - Added future answer columns:
     - `downside_continuation_next_3h`
     - `downside_continuation_next_6h`
     - `downside_exhaustion_next_6h`
     - `support_reclaim_next_6h`
   - Added early-downside hypotheses:
     - `epsilon_downside_first_break_continuation`
     - `epsilon_downside_first_break_exhaustion`
     - `epsilon_support_reclaim_after_break`
     - `epsilon_orderbook_panic_after_break`
   - Added `conf_ob_spread_fragility`.
   - Added epsilon FreqAI strategy/profile support:
     - `ContextTraderConfluenceEpsilonFreqAIResearchStrategy`
     - `ContextTraderConfluenceEpsilonComponentsFreqAIResearchStrategy`
4. Snapshot/dictionary:
   - Rebuilt snapshot:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
     - Rows: `55,971`
     - Columns: `2,576`
   - Refreshed dictionary:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_column_dictionary.csv`
5. Direct-test outputs:
   - Summary CSV:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_goal_feature_research_epsilon_20260605_summary.csv`
   - Markdown:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_confluence_direct_tests_goal_feature_research_epsilon_20260605_report.md`
   - Detail rows: `1,314`
   - Summary rows: `197`
6. Strongest current direct-test watchlist:
   - `delta_structure_orderbook_drawdown_confluence` against `breakdown_success_next_6h`:
     - Trigger event rate `0.535`
     - Same-regime event rate `0.099`
     - Random event rate `0.116`
     - AUC `0.755`
   - `delta_compression_breakdown_release` against `breakdown_success_next_6h`:
     - Trigger event rate `0.535`
     - Same-regime event rate `0.142`
     - Random event rate `0.078`
     - AUC `0.722`
   - `delta_compression_breakout_release` against `breakout_success_next_6h`:
     - Trigger event rate `0.603`
     - Same-regime event rate `0.174`
     - Random event rate `0.135`
     - AUC `0.705`
   - `delta_range_low_breakdown_success` against `breakdown_success_next_6h`:
     - Trigger event rate `0.535`
     - Same-regime event rate `0.250`
     - Random event rate `0.102`
     - AUC `0.687`
   - `delta_range_high_breakout_success` against `breakout_success_next_6h`:
     - Trigger event rate `0.613`
     - Same-regime event rate `0.256`
     - Random event rate `0.089`
     - AUC `0.686`
7. Early-downside epsilon interpretation:
   - `epsilon_support_reclaim_after_break` ranked `support_reclaim_next_6h` strongly with AUC `0.791`, but the active trigger rows had a lower reclaim rate than same-regime rows.
   - Plain-English read: this may be better as a "support reclaim unlikely / danger remains" score than a bullish reclaim score.
   - `epsilon_downside_first_break_continuation` has too few trigger rows (`9`) for continuation targets, so it is not yet a trusted result.
   - `epsilon_orderbook_panic_after_break` has too few trigger rows (`3`) and is rejected for now as too sparse.
8. Feature-discovery outputs:
   - Tiny candidate parquet:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\feature_discovery\feature_discovery_candidates_goal_feature_research_tiny_20260605.parquet`
   - Combined discovery result:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\feature_discovery_screen_goal_feature_research_tiny_20260605_combined.csv`
   - Fast-path shortlisted rows: `172`
   - Top repeated clue:
     - Volume pressure level/rank/z-score repeatedly ranked breakout and breakdown outcomes better than random controls.
     - This supports more focused hypotheses around volume pressure confirming or rejecting structural breaks.
9. Feature-discovery limitation:
   - Full candidate screens of `1,088` to `3,080` columns timed out in synchronous runs.
   - Added `--skip-tree` to `feature_discovery_screen.py` so agents can produce fast univariate/control shortlists first.
   - Later tree validation should run only on shortlisted features.
10. FreqAI validation queue:
    - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_010227_427619\freqai_experiment_queue.json`
    - Purpose:
      - Compare price-only controls against epsilon flagged and epsilon component-only profiles.
      - Targets: `downside_continuation_next_3h`, `support_reclaim_next_6h`.
      - Windows: `spot_q4_2025`, `spot_recent`.
    - Status:
      - Queue created during the feature-definition pass.
      - On continuation, duplicate obsolete full feature-discovery processes were stopped.
      - Compile and whitespace checks passed for touched research/FreqAI files.
      - Queue was launched in the background.
      - Final queue state after scorer fix and rescore: `12` completed, `0` pending.
      - Runner logs are under `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_010227_427619\queue_runner_logs`.
11. Queue scoring fix:
    - First three Q4 downside-continuation runs had valid FreqAI backtest outputs but failed scoring.
    - Error: `downside_continuation_next_3h` was not in the old structure/orderbook actual frame.
    - Root fix: `score_freqai_experiment.py` now merges missing trader-confluence answer columns from `trader_confluence_1h_latest.parquet`.
    - The three failed Q4 runs were rescored and marked completed.
12. FreqAI result: downside continuation after first break:
    - Trader question:
      - After an early downside break is visible, can the epsilon features tell whether the drop keeps going over the next 3h?
    - What was looked at:
      - Price/volume baseline versus epsilon early-downside features and epsilon component-only features.
      - Crash-detection rows and orderbook-present rows.
    - Q4 2025 crash rows:
      - Price-only AUC `0.331`.
      - Epsilon-with-flags AUC `0.332`.
      - Epsilon-components-only AUC `0.332`.
      - Top bucket was not improved in a useful way.
    - Recent crash rows:
      - Price-only AUC `0.510`.
      - Epsilon-with-flags AUC `0.337`.
      - Epsilon-components-only AUC `0.321`.
    - Verdict:
      - Reject for now. The current epsilon continuation feature definition does not help identify drops that keep falling.
13. FreqAI result: support reclaim / downside exhaustion:
    - Trader question:
      - After a support break or crash-type row, can the epsilon features tell whether price reclaims support within 6h?
    - What was looked at:
      - Price/volume baseline versus epsilon-with-flags and epsilon component-only features.
      - Crash-detection rows and orderbook-present rows.
    - Q4 2025 crash rows:
      - Price-only AUC `0.766`, top bucket `15.6%`, event rate `5.2%`.
      - Epsilon-with-flags AUC `0.776`, top bucket `15.6%`.
      - Epsilon-components-only AUC `0.764`, top bucket `15.6%`.
      - Interpretation: price already did most of the ranking; epsilon flags add a small ranking improvement only.
    - Recent crash rows:
      - Price-only AUC `0.576`, top bucket `9.5%`, event rate `7.5%`.
      - Epsilon-with-flags AUC `0.621`, top bucket `14.3%`.
      - Epsilon-components-only AUC `0.638`, top bucket `14.3%`.
      - Interpretation: the component evidence is useful here. It helped identify rows where support reclaim was more likely after a downside event.
    - Verdict:
      - Worth more testing. This should become a focused support-reclaim/exhaustion hypothesis with clearer event rows and controls.
14. Current rejected/weak ideas after the queue:
    - Epsilon downside continuation over 3h: rejected for now.
    - Orderbook panic after break: still rejected for sparse triggers.
    - Broad full-column discovery: still parked because it is too slow and overfit-prone without a smaller shortlist.
15. Current next candidates after the queue:
    - Redesign downside continuation around trader-visible momentum persistence rather than the current epsilon trigger.
    - Promote support-reclaim/exhaustion after a downside break into a focused direct-test and FreqAI pack.
    - Build explicit volume-pressure confirmation/fade hypotheses around VP, TLV2, BOS/CHoCH, and range breaks.

Parallel FreqAI branch-goal setup on 2026-06-05:

1. Purpose:
   - Split the next FreqAI research into four clean branches that can later merge into a trading-research stack.
   - Avoid a single opaque all-feature model.
2. Branch guidance:
   - `C:\FreqTradeStuff\ai_guidance_docs\freqai_parallel_branch_goals.md`
3. Branch A: structure + volume breakouts.
   - Trader question:
     - When price breaks a real level with volume/structure confirmation, does it continue or fail?
   - Queue:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020338_402028\freqai_experiment_queue.json`
   - Experiments: `16`.
   - Preflight: `0` errors, `0` warnings.
4. Branch B: downside risk and exhaustion.
   - Trader question:
     - After a downside break, can we separate continuation danger from reclaim/exhaustion?
   - Queue:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020454_865507\freqai_experiment_queue.json`
   - Experiments: `18`.
   - Preflight: `0` errors, `0` warnings.
5. Branch C: orderbook behaviour states.
   - Trader question:
     - Does orderbook state confirm or warn against structure/volume signals?
   - Queue:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020346_348724\freqai_experiment_queue.json`
   - Experiments: `24`.
   - Preflight: `0` errors, `1` warning for one low-coverage orderbook row.
6. Branch D: regime and confluence gate.
   - Trader question:
     - Which market states make other branch signals more or less trustworthy?
   - Queue:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020348_711597\freqai_experiment_queue.json`
   - Experiments: `16`.
   - Preflight: `0` errors, `0` warnings.
7. Code fix:
   - `freqai_experiment_queue.py` target-balance audit now appends missing answer columns from `trader_confluence_1h_latest.parquet`.
   - This keeps preflight aligned with the scorer for epsilon/delta answer columns.
8. Run status:
   - All four branch queues are created and pending.
   - They have not been executed yet in this setup pass.
11. Current rejected/weak ideas:
    - Full news/context confluence in this pass: rejected/deferred due source sparsity.
    - Orderbook panic after break: rejected for now due too few trigger rows.
    - Broad all-feature discovery: deferred because it is too slow and too likely to overfit without a smaller shortlist.
12. Next candidates:
    - Convert the strong volume-pressure discovery rows into explicit trader hypotheses:
      - break with rising pressure
      - break with fading pressure
      - volume pressure agrees/disagrees with VP/TLV2/BOS level break
    - Reframe epsilon support-reclaim as a danger-persistence/reclaim-unlikely score and retest.
    - Run the created epsilon FreqAI queue, then score against price controls and event rows.
# 2026-06-05 - Concept lifecycle tracker and first branch validations

1. New lifecycle artifacts:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\concept_lifecycle_ledger.py`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_concept_lifecycle_ledger.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_concept_lifecycle_summary.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_concept_evidence_inventory.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_concept_next_cycle_plan.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_concept_next_cycle_plan.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_independent_promising_leads.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_independent_promising_leads.md`
2. Ledger build behaviour:
   - Imports direct-test summaries, threshold-sweep collapsed summaries, feature-discovery combined results, FreqAI ledger rows, and evidence inventory paths.
   - Marks context/news/GDELT/GKG concepts as `deferred_for_data` while news/context extraction is being reworked separately.
   - Prioritises FreqAI `control_comparison` rows so baseline deltas are visible.
3. Fresh direct test:
   - Command: `python user_data/Custom_Launcher/research/context_features/trader_confluence_direct_tests.py --tag concept_lifecycle_cycle_20260605 --execute`
   - Output summary: `trader_confluence_direct_tests_concept_lifecycle_cycle_20260605_summary.csv`
   - Result: 197 summary rows imported.
4. Structure/volume FreqAI queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020338_402028`
   - Result: 14 completed, 2 failed.
   - Useful Q4 result: trader-confluence delta breakout success improved over price control on all predicted rows:
     - Delta components: AUC 0.605, AUC delta +0.063, top bucket delta +0.055.
     - Delta gated: AUC 0.598, AUC delta +0.056, top bucket delta +0.046.
   - Useful event-scope Q4 result: breakout acceptance rows improved over price control:
     - Delta components: AUC 0.587, AUC delta +0.041, top bucket delta +0.060.
   - Q1 structure-only failures:
     - `structure_vp_tree_breakout_success_6h-spot_q1_2026`
     - `structure_vp_tree_breakout_failure_6h-spot_q1_2026`
     - Cause from `freqai_stderr.log`: structural profile dropped 1439 of 1440 training points due to NaNs and left one sample. This is a source/window readiness issue, not evidence against the concept.
5. Downside/exhaustion FreqAI queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020342_609646`
   - Result: 18 completed, 0 failed.
   - Useful support reclaim result:
     - Epsilon components support-reclaim, Q1 crash-detection scope: AUC 0.807, AUC delta +0.032.
     - Epsilon support-reclaim, Q1 crash-detection scope: AUC 0.805, AUC delta +0.030.
   - Useful exhaustion result:
     - Epsilon downside exhaustion, Q4 crash-detection scope: AUC 0.631, AUC delta +0.013, top bucket delta +0.044.
   - Downside continuation remained weak compared with support-reclaim/exhaustion style framing.
6. Lifecycle counts after this pass:
   - `freqai_promising`: 24 validation records.
   - `sweep_promising`: 9 concept/target records.
   - `direct_promising`: 44 concept/target records.
   - `needs_rework`: 49 records.
   - `deferred_for_data`: 108 records.
   - `rejected_for_now`: 116 records.
7. Interpretation:
   - Treat the 24 FreqAI-promising records as validation records, not 24 independent concepts yet.
   - Current independent lead-family grouping has 33 lead families.
   - The most useful current lead families are:
     - compression/range breakout continuation
     - support breakdown continuation/drawdown risk
     - orderbook breakdown/crash-risk ranking
     - orderbook/structure resistance rejection and fakeout detection
     - support reclaim/downside exhaustion after an early break starts
   - Next research step should group duplicate scopes into independent concept cards, then continue branch-balanced testing.
8. Orderbook-state queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020346_348724`
   - Result: 18 completed, 6 failed.
   - Failed rows were Q1 structure-dependent profiles; orderbook-only rows completed.
   - Useful orderbook result: failed-breakout/rejection ranking improved against price controls:
     - Q1 VAH rejection scope: AUC 0.542, AUC delta +0.079, top bucket delta +0.085.
     - Q4 VAH rejection scope: AUC 0.594, AUC delta +0.063, top bucket delta +0.038.
   - Useful breakdown result:
     - Q4 all rows: orderbook breakdown success AUC 0.649, AUC delta +0.011.
     - Q4 crash-detection scope AUC 0.660 but weaker than price control, so this is useful as a component but not independently stronger in that scope.
9. Regime/confluence regression queue:
   - Queue: `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_queue\queue_20260605_020348_711597`
   - Result: 14 completed, 2 failed.
   - Failed rows were Q1 structure-only regression profiles with the same NaN-heavy structure coverage issue.
   - Regression rows do not use AUC. The lifecycle importer now classifies these using prediction/actual correlation delta and top-bucket separation.
   - Useful but weak regression lead:
     - Q4 structure/VP future return and drawdown rows improved correlation versus price control by roughly +0.07, but these should remain secondary until translated into event-style trader concepts.

# 2026-06-05 - Sieve-borrowed BTC trading-rule test

1. Objective:
   - Convert promising research leads into concrete BTC 1h long/short entry rules.
   - Borrow plausible successful behaviours from the Sieve strategy set instead of only using the confluence research hypotheses.
   - Test standalone rule variants and a combined multi-scenario rule block.
2. New harness:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_rule_backtests.py`
   - Input: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
   - Output prefix: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_backtests_*`
3. Test mechanics:
   - Signal at hourly candle close.
   - Enter next candle open.
   - Fixed stop/take-profit/time-exit per rule.
   - Stop wins if stop and take-profit are both touched inside the same candle.
   - Costs include `0.10%` round-trip fee plus `0.02%` slippage.
4. Borrowed Sieve sources:
   - `SIEVE3_GUARD_CANDIDATES.md`
   - `sieve3_candidates/from_20260603_revised_guard_profitable/MANIFEST.md`
   - Exact strategy review for TLV2 resistance breakout, TLV2 support breakdown, and TLV2/VP retest concepts.
5. Important code-review correction:
   - The harness initially reported active-period trade rate, but that allowed a 4-trade rule to pass.
   - Added `usable_sample_size = trades >= 10`.
   - `positive_quality` now requires sample size, usable trade rate, positive return, drawdown better than `-35%`, and either win rate at least `48%` or profit factor at least `1.20`.
6. Final corrected run:
   - Command: `python user_data\Custom_Launcher\research\context_features\trader_rule_backtests.py --tag sieve_exact_min10_20260605 --execute`
   - Summary: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_backtests_sieve_exact_min10_20260605_summary.csv`
   - Markdown: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_backtests_sieve_exact_min10_20260605.md`
   - Combined trades: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_backtests_sieve_exact_min10_20260605_combined_trades.csv`
7. Final selected combined block:
   - `94` trades.
   - `53.2%` win rate.
   - `37.8%` compounded return.
   - `-8.6%` max drawdown.
   - `1.542` profit factor.
   - Did not beat buy-and-hold over the full 2020-2026 BTC snapshot.
8. Combined contribution by rule:
   - `sieve_prior_day_low_break_vp_short`: `31` trades, `58.1%` win rate, `12.4%` compounded contribution.
   - `sieve_pattern_rectangle_breakdown_short`: `10` trades, `50.0%` win rate, `11.6%` compounded contribution.
   - `sieve_prior_day_high_break_vp_long`: `42` trades, `54.8%` win rate, `6.9%` compounded contribution.
   - `sieve_pattern_rectangle_breakout_long`: `11` trades, `36.4%` win rate, `2.8%` compounded contribution.
9. Combined year split:
   - 2025: `62` trades, `46.8%` win rate, `6.8%` compounded return.
   - 2026: `32` trades, `65.6%` win rate, `29.0%` compounded return.
10. Interpretation:
   - This is the first conversion of research leads into an actual rule block that looks worth developing further.
   - The strongest current theme is not raw FreqAI prediction. It is Sieve-style technical behaviour: prior 24h level breaks with VP/pressure confirmation, plus rectangle/compression breaks.
   - The result is promising for rule-development but not yet a production strategy because it does not beat buy-and-hold and the useful custom-indicator coverage is mostly 2025-2026.
11. Storage note:
   - `C:\FreqTradeStuff\user_data\models`: about `28.34 GB`.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\freqai_runs`: about `0.45 GB`.
   - No deletion was performed.
12. Freqtrade signal export:
   - The rule harness now writes:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_signals_sieve_exact_min10_20260605.parquet`
   - Signal rows: `94`.
   - Signal split: `53` long, `41` short.
13. Freqtrade research strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockResearchStrategy.py`
   - Loads the prebuilt signal parquet and merges by `date`.
   - Does not calculate slow custom indicators in `populate_indicators()`.
   - Uses per-rule custom exits:
     - Prior-day high/low VP pressure rules: `2%` take profit, `48h` time exit.
     - Rectangle breakout/breakdown rules: `5%` take profit, `10h` time exit.
   - Uses global stoploss: `-2%`.
14. Freqtrade command:
   - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockResearchStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
15. Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_21-31-27.zip`
16. Freqtrade result:
   - Window: `2025-05-01` to `2026-05-10`.
   - Trades: `94`.
   - Win rate: `57.4%`.
   - Total account return: `50.73%`.
   - CAGR: `49.25%`.
   - Profit factor: `1.66`.
   - Max account underwater: `9.73%`.
   - Market change over the same window: `-14.54%`.
17. Freqtrade enter-tag split:
   - `sieve_pattern_rectangle_breakdown_short`: `10` trades, `80.0%` win rate, `21.79%` profit contribution.
   - `sieve_prior_day_low_break_vp_short`: `31` trades, `64.5%` win rate, `19.12%` profit contribution.
   - `sieve_prior_day_high_break_vp_long`: `42` trades, `50.0%` win rate, `9.50%` profit contribution.
   - `sieve_pattern_rectangle_breakout_long`: `11` trades, `45.5%` win rate, `0.33%` profit contribution.
18. Interpretation:
   - The Freqtrade result is stronger than the fast harness because the final run used correct single-position capital sizing and Freqtrade's trade engine.
   - A previous Freqtrade run without `--max-open-trades 1` showed only `4.30%` because capital was effectively split for up to ten positions.
   - The selected technical rule block is worth further development.
   - It is not production-ready until walk-forward stability, broader data coverage, fees/slippage sensitivity, and exit/risk management are tested.

# 2026-06-05 - Lead registry and initial confluence screen

1. Goal shift:
   - The active development goal is no longer just "find a few promising concepts".
   - It is now:
     - maintain `50` to `100` trader-readable leads,
     - test confluence per lead,
     - tune exits for successful entry families,
     - then use orderbook/news/context/regime as risk and leverage overlays where they are actually ready.
2. New registry script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_registry.py`
3. Registry outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260605.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260605.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_latest.csv`
4. Registry counts:
   - Total leads: `100`
   - Structure/volume: `41`
   - Orderbook state: `18`
   - Downside risk: `18`
   - Sieve-borrowed: `18`
   - Regime confluence: `5`
5. Registry interpretation:
   - The registry is a research control surface, not a trading strategy.
   - It intentionally avoids promoting news/context leads while source-specific context features are not ready.
   - It records for each lead: trader question, visible market state, source block, evidence, confluence candidates, exit-research need, risk-sizing candidates, and next action.
6. New confluence-plan output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_plan_20260605.csv`
   - Planned checks: `572`
7. New confluence checker:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_confluence_checks.py`
8. Initial confluence test target:
   - Selected Sieve-derived entries only.
   - Input trades: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_backtests_sieve_exact_min10_20260605_combined_trades.csv`
   - Input feature cache: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
9. Strict confluence output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260605_strict.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260605_strict.md`
10. Trader question:
   - If the selected entry rules already look promising, does requiring extra evidence from another source family make the entries cleaner?
11. Full-block result:
   - Base block: `94` trades, `53.2%` win rate, `0.359%` average net return per trade, `1.542` profit factor.
   - Strong volume filter: `62` trades, `51.6%` win rate, `1.215` profit factor.
   - Structure-trigger filter: `31` trades, `51.6%` win rate, `1.170` profit factor.
   - VP context filter: `57` trades, `50.9%` win rate, `1.408` profit factor.
   - Range-break filter: `87` trades, `51.7%` win rate, `1.419` profit factor.
   - Orderbook-direction filter: `37` trades, `51.4%` win rate, `1.063` profit factor.
   - Result: broad global confluence filters made the full selected block worse.
12. Rule-specific watchlist:
   - Rectangle breakout long plus VP context:
     - `9` trades.
     - Win rate improved from `36.4%` to `44.4%`.
     - Average return improved from `0.264%` to `0.543%`.
     - Profit factor improved from `1.603` to `2.719`.
     - Verdict: watchlist only because sample size is still small.
   - Prior-day low break VP short plus strong volume and structure:
     - `11` trades.
     - Win rate improved from `58.1%` to `63.6%`.
     - Average return improved from `0.394%` to `0.466%`.
     - Profit factor improved from `1.545` to `1.616`.
     - Verdict: watchlist for tailored short-entry confluence.
13. Key lesson:
   - Confluence is not a universal add-on.
   - It must be tailored to the lead.
   - Broad filters can remove the good entries or duplicate logic already inside the entry rule.
14. Next best work:
   - Build per-lead confluence tests for the top `20` to `30` registry leads.
   - Keep broad readiness checks like `orderbook_present` separate from predictive filters.
   - Move only leads with enough trades and improved quality into exit research.
   - For exits, test per-family logic: next-level target, failed-acceptance exit, support/reclaim exit, pressure fade, and time stop.

# 2026-06-05 - Tailored exit and risk-overlay pass

1. New tailored-exit script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_exit_research.py`
2. Trader question:
   - Keep the selected entry signals fixed. Can we improve the trade block by giving each entry family its own stop, target, and hold time?
3. Exit sweep inputs:
   - Signal parquet: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_signals_sieve_exact_min10_20260605.parquet`
   - Feature/OHLCV cache: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
4. Exit sweep outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_combined_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605.md`
5. Exit variants tested:
   - `408`
6. Selected exit per family:
   - `sieve_pattern_rectangle_breakout_long`:
     - Hold `48h`, stop `3.5%`, target `7.5%`.
     - `11` trades, `72.7%` win rate, `29.7%` fast-harness return, `-1.4%` drawdown, `13.17` profit factor.
   - `sieve_pattern_rectangle_breakdown_short`:
     - Hold `10h`, stop `2.5%`, target `5.0%`.
     - `10` trades, `50.0%` win rate, `11.6%` fast-harness return, `-2.2%` drawdown, `4.68` profit factor.
   - `sieve_prior_day_low_break_vp_short`:
     - Hold `36h`, stop `3.5%`, target `5.0%`.
     - `31` trades, `51.6%` win rate, `18.3%` fast-harness return, `-7.2%` drawdown, `1.61` profit factor.
   - `sieve_prior_day_high_break_vp_long`:
     - Hold `72h`, stop `3.5%`, target `2.0%`.
     - `42` trades, `64.3%` win rate, `8.0%` fast-harness return, `-7.9%` drawdown, `1.25` profit factor.
7. Combined tailored-exit fast-harness result:
   - `76` trades.
   - `52.6%` win rate.
   - `39.3%` compounded return.
   - `-10.3%` max drawdown.
   - `1.563` profit factor.
8. Fast-harness interpretation:
   - Tailored exits improved average trade return and profit factor slightly versus the original fast-harness block.
   - They reduced trade count and increased drawdown.
   - This is a watchlist result, not a clear promotion.
9. Research strategy handling:
   - Baseline fixed-exit strategy remains:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockResearchStrategy.py`
   - Tailored-exit variant was added separately:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockTailoredExitResearchStrategy.py`
   - The tailored variant uses the selected take-profit and hold-hour values.
   - It sets global stoploss to `-3.5%`, which matches the widest selected stop.
   - Caveat: exact per-rule stop matching is not implemented yet; the rectangle-breakdown selected stop was `2.5%`, but the tailored Freqtrade run used the global wider stop.
10. Tailored-exit Freqtrade command:
   - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockTailoredExitResearchStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
11. Tailored-exit Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-00-34.zip`
12. Tailored-exit Freqtrade result:
   - `83` trades.
   - `60.2%` win rate.
   - `49.19%` account return.
   - `11.56%` max account underwater.
   - `1.60` profit factor.
   - Market change over the same window: `-14.54%`.
13. Comparison to previous fixed-exit Freqtrade validation:
   - Previous fixed-exit block: `94` trades, `57.4%` win rate, `50.73%` return, `9.73%` max drawdown, `1.66` profit factor.
   - Tailored-exit block: higher win rate, fewer trades, lower return, worse drawdown, lower profit factor.
   - Current best remains the previous fixed-exit Freqtrade result.
14. New risk-overlay script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
15. Risk-overlay trader question:
   - Keep entries and exits fixed. Can extra metrics safely size trades up/down?
16. Risk-overlay outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260605.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260605.md`
17. Risk overlays tested:
   - Entry score confidence.
   - Directional volume sizing.
   - Structure agreement sizing.
   - Orderbook warning sizing.
   - Volatility risk-off sizing.
   - Conservative combined sizing.
18. Risk-overlay result:
   - Equal `1x` sizing remains best.
   - Directional volume sizing raised return from `39.3%` to `40.7%`, but worsened drawdown from `-10.3%` to `-12.2%` and reduced profit factor from `1.563` to `1.498`.
   - Orderbook warning sizing reduced drawdown to `-7.7%`, but cut return to `28.9%`.
   - No risk overlay was promoted.
19. Current practical conclusion:
   - Keep the fixed-exit Freqtrade rule block as current best.
   - Preserve tailored exits as per-family evidence, especially rectangle breakout and prior-day-low short ideas.
   - Next work should test exits per family in Freqtrade-compatible form and split risk overlays by entry family, not broad whole-block sizing.

# 2026-06-05 - Per-rule confluence pass

1. New script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_per_rule_confluence.py`
2. Trader question:
   - Instead of asking whether one filter helps every setup, ask whether a specific extra condition improves a specific entry family.
3. Inputs:
   - Signals: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_rule_signals_sieve_exact_min10_20260605.parquet`
   - Feature cache: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
4. Outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605.md`
5. Exported signal sets:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_baseline_exit_selected_confluence.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_tailored_exit_selected_confluence.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_best_per_rule_selected_confluence.parquet`
6. Tests run:
   - `144` rule/filter/exit combinations.
   - Both baseline-exit and tailored-exit versions were tested.
7. Selected per-rule confluence findings:
   - Rectangle breakout long + VP context + tailored exit:
     - Kept `9` of `11` decision points.
     - Win rate improved from `72.7%` to `77.8%`.
     - Average return improved from `2.43%` to `2.99%`.
     - Profit factor improved from `13.17` to `15.25`.
   - Rectangle breakout long + VP context + baseline exit:
     - Kept `9` of `11` decision points.
     - Win rate improved from `36.4%` to `44.4%`.
     - Average return improved from `0.26%` to `0.54%`.
     - Profit factor improved from `1.60` to `2.72`.
   - Prior-day low break short + strong volume and structure + baseline exit:
     - Kept `11` of `31` decision points.
     - Win rate improved from `58.1%` to `63.6%`.
     - Average return improved from `0.39%` to `0.47%`.
     - Profit factor improved from `1.55` to `1.62`.
   - Prior-day low break short + strong volume + baseline exit:
     - Kept `25` of `31` decision points.
     - Win rate improved from `58.1%` to `60.0%`.
     - Profit factor improved from `1.55` to `1.60`.
     - Watchlist only.
8. New strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockConfluenceResearchStrategy.py`
9. Strategy purpose:
   - Freqtrade-validate the best per-rule confluence-filtered signal set.
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_best_per_rule_selected_confluence.parquet`
10. Freqtrade command:
   - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockConfluenceResearchStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
11. Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-05-53.zip`
12. Freqtrade result:
   - `16` trades.
   - `50.0%` win rate.
   - `4.78%` account return.
   - `8.35%` max account underwater.
   - `1.25` profit factor.
   - Market change over the same window: `-14.54%`.
13. Freqtrade split:
   - Rectangle breakout long:
     - `5` trades, `40.0%` win rate, `7.97%` profit contribution.
   - Prior-day low break VP short:
     - `11` trades, `54.5%` win rate, `-3.19%` profit contribution.
14. Interpretation:
   - Per-rule confluence is more sensible than global filtering, but the first filtered strategy is not strong enough.
   - The rectangle breakout + VP context idea still looks interesting but has too few trades.
   - The prior-day-low short confluence filter improved fast-harness metrics but failed in Freqtrade, so it needs rework.
   - Current best remains the unfiltered fixed-exit `TraderRuleBlockResearchStrategy`.
15. Next work:
   - Expand rectangle breakout variants to increase sample size.
   - Rework prior-day-low short confluence around exit timing or support-reclaim failure, not just strong volume/structure.
   - Avoid promoting filtered confluence until Freqtrade result improves over the unfiltered baseline or clearly reduces risk while preserving enough return.

# 2026-06-05 - Rectangle/VP expansion and Freqtrade validation

1. Trader question:
   - Can the strongest surviving clue, rectangle breakout plus VP context, be widened into more entries without losing the clean market story?
2. What was built:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_rectangle_vp_expansion.py`
3. What the script tests:
   - 4h/1d rectangle upper breaks with VP support.
   - Multi-timeframe rectangle upper breaks.
   - Rectangle/compression squeeze releases.
   - Rectangle breaks above VAH or below VAL.
   - Rectangle breaks into LVN/thin-liquidity areas.
   - Rectangle breaks with TLV2 or orderbook acceptance.
   - Compression plus recent range break with VP support.
4. Direct harness outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605_combined_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_expansion_20260605.md`
5. Direct harness result:
   - `221` variant/threshold/exit combinations tested.
   - `9` unique base variants selected.
   - Combined selected block: `66` trades, `53.0%` win rate, `16.3%` return, `-11.9%` max drawdown, `1.320` profit factor.
6. Full expanded Freqtrade validation:
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpExpansionStrategy.py`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_expansion_20260605_selected.parquet`
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockRectangleVpExpansionStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-15-56.zip`
   - Result:
     - `66` trades.
     - `56.1%` win rate.
     - `23.33%` return.
     - `9.95%` max account underwater.
     - `1.51` profit factor.
7. Full expanded interpretation:
   - This is positive enough to keep researching.
   - It does not beat the current best 94-trade fixed-exit block.
   - The broad `compression_range_break_vp_long` variant adds many entries but weak average quality.
8. Quality subset:
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_expansion_20260605_quality.parquet`
   - Kept:
     - `rect_break_below_val_short`
     - `rect_break_orderbook_accept_long`
     - `rect_squeeze_upper_release_long`
   - Excluded:
     - `compression_range_break_vp_long`
     - Weak small long variants that did not add much after overlap.
9. Quality Freqtrade validation:
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpQualityStrategy.py`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-16-45.zip`
   - Result:
     - `16` trades.
     - `62.5%` win rate.
     - `18.57%` return.
     - `3.44%` max account underwater.
     - `3.89` profit factor.
10. Trader-readable conclusion:
   - The cleanest current rectangle/VP story is not simply "breakout good".
   - Better current story:
     - price leaves a rectangle/compression area,
     - VP says value is accepting the move or the break is through VAL/VAH,
     - orderbook acceptance improves a small number of long cases,
     - downside VAL breaks can work but require better exit logic because many shorts time-exit weakly unless they reach target fast.
11. Next work:
   - Split rectangle/VP into high-quality sparse leads and broader rework leads.
   - Expand trade count around the quality subset without re-adding the broad weak compression-range version unchanged.
   - Rework `compression_range_break_vp_long` with stricter acceptance, volume persistence, and invalidation logic.
   - Develop tailored exits for `rect_break_below_val_short`; current winners are strong, but weak time exits dilute it.
12. Verification:
   - Compile passed for:
     - `trading_lead_rectangle_vp_expansion.py`
     - `TraderRuleBlockRectangleVpExpansionStrategy.py`
     - `TraderRuleBlockRectangleVpQualityStrategy.py`

# 2026-06-05 - Rectangle/VP focused refinement

1. Trader question:
   - Can the noisy broad compression breakout be repaired?
   - Can the VAL breakdown short be improved by avoiding cases where bullish volume/VP context fights the short?
2. What was built:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_rectangle_vp_refinement.py`
3. Focused refinement ideas:
   - Compression breakout with stronger compression.
   - Compression breakout with persistent 24h volume and positive pressure.
   - Compression breakout with VP value acceptance above VAH.
   - Compression breakout not already overextended at the range high.
   - Compression breakout with orderbook acceptance.
   - VAL breakdown with weak bullish volume.
   - VAL breakdown with weak long-side VP opposition.
   - VAL breakdown as a controlled, moderate-volume break.
   - VAL breakdown while price still has room toward the lower range.
4. Direct harness result:
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605_summary.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rectangle_vp_refinement_20260605.md`
   - `185` variant/threshold/exit combinations tested.
   - Direct-harness selected block:
     - `159` trades.
     - `52.2%` win rate.
     - `190.5%` compounded return.
     - `-18.4%` max drawdown.
     - `1.672` profit factor.
5. Freqtrade full-refinement validation:
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpRefinementStrategy.py`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-23-39.zip`
   - Result:
     - `54` trades.
     - `50.0%` win rate.
     - `13.30%` account return.
     - `12.46%` max account underwater.
     - `1.36` profit factor.
6. Important failure:
   - `compression_break_vp_long_volume_persistent` looked strong in the direct harness.
   - In Freqtrade it contributed:
     - `35` trades.
     - `48.6%` win rate.
     - `-1.17%` profit contribution.
   - Verdict:
     - Reject or rework this exact definition.
     - The direct harness overstated it because actual execution/overlap changed the trade set.
7. Refined-quality subset:
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_rectangle_vp_refinement_20260605_quality.parquet`
   - Kept:
     - `rect_val_short_vp_opposition_low`
     - `rect_squeeze_long_quality_keep`
     - `compression_break_vp_long_value_acceptance`
   - Excluded:
     - broad compression-volume long
     - orderbook-acceptance long after weak Freqtrade contribution
     - weak VAL range/mid and moderate-volume variants after overlap
8. Freqtrade refined-quality validation:
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleVpRefinedQualityStrategy.py`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-24-14.zip`
   - Result:
     - `10` trades.
     - `60.0%` win rate.
     - `20.88%` account return.
     - `2.94%` max account underwater.
     - `5.49` profit factor.
9. Plain-English interpretation:
   - The useful short story is:
     - price breaks below rectangle/value,
     - VP is not showing strong long-side support/opposition,
     - the trade either moves quickly to target or should probably not be held too long.
   - This is a sparse but credible lead.
   - It is not enough to be a whole strategy by itself.
10. Next work:
   - Expand this VAL/VP-opposition short into nearby concepts:
     - support break with weak VP long support,
     - TLV2 support break with weak VP opposition,
     - BOS/CHoCH bearish state plus weak VP long support,
     - orderbook support removal added as a risk/confirmation layer.
   - Rework compression-volume long using actual Freqtrade execution evidence, not direct-harness rankings.
11. Verification:
   - Compile passed for:
     - `trading_lead_rectangle_vp_refinement.py`
     - `TraderRuleBlockRectangleVpRefinementStrategy.py`
     - `TraderRuleBlockRectangleVpRefinedQualityStrategy.py`

# 2026-06-05 - Adjacent support-break short expansion

1. Trader question:
   - Does the sparse VAL/VP-opposition short lead generalize into broader support-break shorts?
2. What was built:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_support_break_short_expansion.py`
3. Concepts tested:
   - Prior-low break with weak VP long opposition.
   - Prior-low break with quiet bullish volume.
   - TLV2 support break with weak VP long opposition.
   - Bearish structure breakdown with weak VP long support.
   - VAL break plus LVN/thinness below.
   - Support break with orderbook breakdown acceptance.
   - Bearish pressure near TLV2 support.
   - Support break while price is still mid-range.
4. Direct harness result:
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605_summary.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_support_break_short_expansion_20260605.md`
   - `160` variant/threshold/exit combinations tested.
   - Selected direct-harness block:
     - `92` trades.
     - `38.0%` win rate.
     - `4.7%` return.
     - `-15.7%` max drawdown.
     - `1.093` profit factor.
5. Freqtrade validation:
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockSupportBreakShortExpansionStrategy.py`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-28-04.zip`
   - Result:
     - `85` trades.
     - `42.4%` win rate.
     - `-0.03%` return.
     - `12.27%` max account underwater.
     - `1.00` profit factor.
6. Plain-English conclusion:
   - The broader support-break short idea is not good enough as currently defined.
   - The narrow VAL/VP-opposition short is still useful, but it did not generalize to all nearby support-break patterns.
   - `support_break_orderbook_accept_short` is a weak watchlist clue only:
     - `30` Freqtrade entries.
     - `53.3%` win rate.
     - `2.8%` contribution.
   - This needs sharper orderbook support-removal or faster invalidation, not broader entry definitions.
7. Verdict:
   - Reject broad support-break expansion for now.
   - Park orderbook-acceptance short as rework candidate.
   - Keep narrow `rect_val_short_vp_opposition_low` as the promoted sparse short lead.
8. Verification:
   - Compile passed for:
     - `trading_lead_support_break_short_expansion.py`
     - `TraderRuleBlockSupportBreakShortExpansionStrategy.py`

# 2026-06-05 - Orderbook support-removal short refinement

1. Trader question:
   - Can the failed broad support-break short branch be repaired by requiring the orderbook to show support actually cleared or was removed?
2. What was built:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_orderbook_support_removal_short_refinement.py`
3. Concepts tested:
   - VAL breakdown with support removed.
   - VAL breakdown with downside vacuum after support removal.
   - VAL breakdown with low bounce/failure warnings.
   - Recent support break where orderbook support is cleared.
   - Support break with support removed and bounce warnings low.
   - Support break with bearish book pressure.
   - Fast target-or-out support break.
4. Direct harness result:
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_support_removal_short_refinement_20260605_summary.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_support_removal_short_refinement_20260605_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_support_removal_short_refinement_20260605.md`
   - `118` variant/threshold/exit combinations tested.
   - Direct selected block:
     - `43` trades.
     - `60.5%` win rate.
     - `26.2%` return.
     - `-6.5%` max drawdown.
     - `2.333` profit factor.
5. Direct selected variants:
   - `support_break_support_cleared_short`:
     - Trader saw: recent low/support breaks and orderbook support-cleared score agrees.
     - Direct result: `18` trades, `66.7%` win rate, `21.8%` return, `-4.5%` drawdown, `3.017` profit factor.
   - `val_break_downside_vacuum_after_support_removed_short`:
     - Trader saw: price breaks below value and orderbook shows downside vacuum after support removal.
     - Direct result: `27` trades, `55.6%` win rate, `5.4%` return, `-4.4%` drawdown, `1.638` profit factor.
6. Freqtrade validation:
   - Strategy:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockOrderbookSupportRemovalShortStrategy.py`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-32-42.zip`
   - Result:
     - `43` trades.
     - `58.1%` win rate.
     - `12.40%` account return.
     - `6.69%` max account underwater.
     - `1.49` profit factor.
7. Freqtrade split:
   - `support_break_support_cleared_short`:
     - `18` entries.
     - `61.1%` win rate.
     - `11.37%` contribution.
   - `val_break_downside_vacuum_after_support_removed_short`:
     - `25` entries.
     - `56.0%` win rate.
     - `1.03%` contribution.
8. Plain-English conclusion:
   - Generic support-break shorts failed.
   - Support-break shorts with explicit orderbook support-cleared evidence are materially better.
   - Downside vacuum after support removal is useful as a quick-short context, but it contributes much less than the support-cleared rule.
9. Verdict:
   - Promote `support_break_support_cleared_short` as a watchlist lead.
   - Keep `val_break_downside_vacuum_after_support_removed_short` as a secondary quick-short clue.
   - Continue improving exits/invalidation before merging into a final multi-scenario block.
10. Verification:
    - Compile passed for:
      - `trading_lead_orderbook_support_removal_short_refinement.py`
      - `TraderRuleBlockOrderbookSupportRemovalShortStrategy.py`

# 2026-06-05 - Support-removal exit/risk and split validation

1. Trader question:
   - Once support-removal shorts work, should both sub-leads use the same stop/exit/risk handling?
2. What was built:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_support_removal_exit_risk_research.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockOrderbookSupportRemovalTightStopStrategy.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockSupportRemovalSupportClearedStrategy.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockSupportRemovalValVacuumTightStrategy.py`
3. Direct exit/risk screen:
   - Selected combined block:
     - `43` trades.
     - `60.5%` win rate.
     - `26.2%` return.
     - `-6.5%` max drawdown.
     - `2.333` profit factor.
   - Best support-cleared exit:
     - `balanced_18h_25_50`.
     - `18` trades, `66.7%` win rate, `21.8%` return, `-4.5%` drawdown, `3.017` profit factor.
   - Best VAL/vacuum exit:
     - `quick_6h_14_25`.
     - `25` trades, `56.0%` win rate, `3.6%` return, `-4.4%` drawdown, `1.461` profit factor.
4. Risk overlay screen:
   - `score_confidence_0p6_1p4` and `bounce_warning_reduce` looked worth watching in the direct harness.
   - Size overlays based only on support-cleared, downside-vacuum, or bearish-pressure strength were rejected for now.
   - Equal sizing remains the baseline until per-rule sizing is validated in Freqtrade.
5. Freqtrade global tight-stop control:
   - Strategy:
     - `TraderRuleBlockOrderbookSupportRemovalTightStopStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-37-35.zip`
   - Result:
     - `43` trades.
     - `48.8%` win rate.
     - `1.93%` return.
     - `5.16%` max account underwater.
     - `1.09` profit factor.
   - Plain-English result:
     - A single tight stop was too blunt. It helped the quick short but cut off the support-cleared short too early.
6. Freqtrade split validation:
   - Support-cleared standalone:
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-40-03.zip`
     - `18` trades.
     - `61.1%` win rate.
     - `10.65%` return.
     - `3.73%` max account underwater.
     - `1.95` profit factor.
   - VAL/vacuum quick standalone:
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-40-14.zip`
     - `25` trades.
     - `56.0%` win rate.
     - `5.67%` return.
     - `2.85%` max account underwater.
     - `1.65` profit factor.
7. Plain-English conclusion:
   - These are two different short behaviours.
   - Support-cleared short says: "support has broken and the book no longer looks able to defend it"; it can be held a bit longer.
   - VAL/vacuum short says: "value has broken and liquidity below is thin"; it should work quickly or be stopped quickly.
8. Verdict:
   - Promote support-cleared short as the cleaner orderbook short lead.
   - Keep VAL/vacuum short as a secondary quick-short lead.
   - Reject one shared global tight stop for the mixed block.
9. Next step:
   - Add these as separate candidates in the 50-100 lead pool.
   - Test confluence separately for each:
     - support-cleared short plus bearish structure/volume continuation,
     - VAL/vacuum short plus fast downside momentum and no immediate bounce warning.
   - Only then consider per-rule sizing or leverage.
10. Verification:
    - Compile passed for:
      - `trading_lead_support_removal_exit_risk_research.py`
      - `TraderRuleBlockOrderbookSupportRemovalTightStopStrategy.py`
      - `TraderRuleBlockSupportRemovalSupportClearedStrategy.py`
      - `TraderRuleBlockSupportRemovalValVacuumTightStrategy.py`

# 2026-06-05 - Support-removal confluence validation

1. Trader question:
   - Do extra structure/VP/regime/volume/orderbook filters improve the two validated support-removal short leads?
2. What was changed:
   - `trading_lead_registry.py` now reads a generic validated-entry source file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_validated_entry_leads.csv`
   - `trading_lead_per_rule_confluence.py` now infers validated exit profiles for:
     - `support_break_support_cleared_short`
     - `val_break_downside_vacuum_after_support_removed_short`
3. Registry/queue state:
   - Registry:
     - `100` leads.
     - `41` structure/volume, `18` orderbook, `18` downside-risk, `18` Sieve-borrowed, `5` regime/confluence.
   - Next queue:
     - `48` tasks.
     - `36` per-lead confluence, `6` tailored confluence retests, `6` exit-research tasks.
4. Direct confluence screen:
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal_summary.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal.md`
   - Best-looking direct filters:
     - `support_break_support_cleared_short + structure_trigger_agrees`:
       - `5` of `18` rows, `80.0%` win rate, high profit factor, but sparse.
     - `support_break_support_cleared_short + vp_context_agrees`:
       - `14` of `18` rows, `71.4%` direct win rate.
     - `val_break_downside_vacuum_after_support_removed_short + regime_agrees`:
       - `24` of `25` rows, minor improvement in direct harness.
5. Freqtrade validation:
   - `TraderRuleBlockSupportRemovalSupportClearedConfluenceStrategy`
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-44-55.zip`
     - `16` trades.
     - `62.5%` win rate.
     - `10.30%` return.
     - `3.75%` drawdown.
     - `2.06` profit factor.
   - `TraderRuleBlockSupportRemovalValVacuumConfluenceStrategy`
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-45-06.zip`
     - `24` trades.
     - `54.2%` win rate.
     - `5.51%` return.
     - `2.85%` drawdown.
     - `1.63` profit factor.
6. Plain-English conclusion:
   - The first broad confluence filters did not clearly improve the support-removal shorts.
   - Support-cleared confluence may be useful for removing a couple of lower-quality trades, but standalone support-cleared still has slightly higher total return.
   - VAL/vacuum regime filtering did not improve Freqtrade results.
7. Verdict:
   - Keep standalone support-removal leads as the preferred evidence.
   - Mark first broad confluence pass as neutral, not promoted.
   - Rework confluence around specific failure warnings instead of generic agreement filters.
8. Next step:
   - Support-cleared short:
     - Test "no support reclaim", "no support rebuild", and "bearish pressure still present" filters.
   - VAL/vacuum short:
     - Test "fast downside momentum", "no immediate bounce", and "book does not refill below price" filters.
9. Verification:
   - Compile passed for:
     - `trading_lead_registry.py`
     - `trading_lead_per_rule_confluence.py`
     - `TraderRuleBlockSupportRemovalSupportClearedConfluenceStrategy.py`
     - `TraderRuleBlockSupportRemovalValVacuumConfluenceStrategy.py`

# 2026-06-05 - Specific no-reclaim filter improves VAL/vacuum quick short

1. Trader question:
   - When price breaks below value into a downside vacuum, does the quick short become cleaner if the orderbook does not warn that broken support is being reclaimed?
2. What we looked at:
   - Existing VAL/vacuum quick-short decision rows.
   - Support-removal/orderbook downside-vacuum state.
   - Support reclaim and support rebuild warning features.
   - Breakdown failure and downside panic/orderbook state features.
3. What was changed:
   - Added targeted filters to:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_confluence_checks.py`
   - Added validation strategies:
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy`
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExitTunedStrategy`
4. Direct result:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260605_support_removal_specific.md`
   - `no_support_reclaim_warning` kept `9` of `25` decision rows.
   - The kept rows were materially cleaner:
     - Win rate `77.8%` vs `56.0%`.
     - Average return `0.79%` vs `0.15%`.
     - Profit factor `4.90` vs `1.46`.
5. Freqtrade validation:
   - Baseline no-reclaim strategy:
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-49-29.zip`
     - `9` trades.
     - `77.8%` win rate.
     - `7.89%` return.
     - `1.45%` drawdown.
     - `4.41` profit factor.
   - Tuned-exit strategy:
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_22-51-42.zip`
     - Exit profile:
       - `8h` max hold.
       - `1.0%` stop.
       - `3.5%` target.
     - `9` trades.
     - `77.8%` win rate.
     - `7.35%` return.
     - `1.05%` drawdown.
     - `4.31` profit factor.
6. Plain-English conclusion:
   - This is the first clear example in this branch where confluence improved entry quality.
   - The trader story is:
     - price breaks below value,
     - support was removed,
     - liquidity below is thin,
     - and the book does not show broken support being reclaimed.
   - When those are true together, the quick short is much cleaner than the broader VAL/vacuum short.
7. Verdict:
   - Promote as a sparse confluence lead.
   - Do not treat it as production-ready because there are only `9` trades.
   - Prefer the 6h baseline exit when maximizing return; prefer the tuned 8h/1.0% stop profile when minimizing drawdown.
8. Next step:
   - Expand the no-reclaim/no-bounce family to increase trade count.
   - Add month/window stability.
   - Test against same-regime random short controls.
   - Only then test risk sizing.
9. Verification:
   - Compile passed for:
     - `trading_lead_confluence_checks.py`
     - `trading_lead_per_rule_confluence.py`
     - `trading_lead_exit_research.py`
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimStrategy.py`
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExitTunedStrategy.py`

# 2026-06-05 - VAL/vacuum quick-short expansion

1. Trader question:
   - Can the clean but sparse VAL/vacuum no-reclaim quick short be widened without losing the reason it worked?
2. What was tested:
   - Existing `25` VAL/vacuum support-removal short rows.
   - Threshold sweeps around:
     - support-reclaim warning,
     - support-rebuild warning,
     - breakdown-failure warning,
     - downside-vacuum strength,
     - bearish pressure,
     - orderbook panic/downside acceleration.
   - Two exit styles:
     - `6h` hold, `1.4%` stop, `2.5%` target.
     - `8h` hold, `1.0%` stop, `3.5%` target.
3. Script/output:
   - Script:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_val_vacuum_expansion.py`
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_val_vacuum_expansion_20260605_val_vacuum_no_reclaim_expansion.md`
   - Summary CSV:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_val_vacuum_expansion_20260605_val_vacuum_no_reclaim_expansion_summary.csv`
4. Code issue fixed:
   - The first run failed because the script tried to export raw signal rows through a helper that expects simulated trade rows.
   - Fixed by adding direct signal-row export inside the expansion script.
   - Added stability fields:
     - active years,
     - positive years,
     - active months,
     - positive months.
5. Freqtrade result - looser no-reclaim:
   - Trader story:
     - Price breaks below value after support removal.
     - Downside liquidity is thin.
     - Broken-support reclaim warning is still limited, but the threshold is looser than the strict 9-trade lead.
   - Strategy:
     - `TraderRuleBlockSupportRemovalValVacuumNoReclaimExpandedStrategy`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_no_reclaim_lte_46.parquet`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-03-36.zip`
   - Result:
     - `12` trades.
     - `66.7%` win rate.
     - `7.04%` return.
     - `1.58%` drawdown.
     - `2.84` profit factor.
     - Positive in both year buckets.
6. Freqtrade result - broader bearish-pressure version:
   - Trader story:
     - Price breaks below value after support removal.
     - Downside liquidity is thin.
     - Bearish pressure is still present.
   - Strategy:
     - `TraderRuleBlockSupportRemovalValVacuumPressureExpandedStrategy`
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_val_vacuum_no_reclaim_expansion_pressure_gte_20.parquet`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-04-09.zip`
   - Result:
     - `21` trades.
     - `61.9%` win rate.
     - `7.99%` return.
     - `2.94%` drawdown.
     - `2.36` profit factor.
     - Positive in both year buckets.
7. Plain-English conclusion:
   - The strict no-reclaim lead remains the cleanest.
   - The looser no-reclaim version gives a modest trade-count expansion but gives up some quality.
   - The pressure version gives better trade volume but is a separate adjacent lead: it asks whether sell pressure remains after support removal, not whether reclaim risk is absent.
8. Verdict:
   - Promote both as validated-watchlist support-removal short leads.
   - Do not merge them yet.
   - Next work should test overlap, same-regime controls, and whether these can be combined with support-cleared shorts without damaging the block.

# 2026-06-05 - Validated orderbook overlap and combined block

1. Trader question:
   - Are the validated support-removal short leads separate opportunities, or are several of them just different labels for the same market event?
2. Script/output:
   - Script:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_validated_overlap.py`
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_validated_overlap_20260605_validated_orderbook_overlap.md`
   - Core signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_validated_orderbook_overlap_core_validated_orderbook.parquet`
3. What was checked:
   - Exact overlap:
     - same signal hour.
   - Near overlap:
     - another validated lead fired within `8h`.
   - Non-overlapped combined execution:
     - take the highest-priority active lead and stay flat until exit.
4. Overlap finding:
   - Strict no-reclaim VAL/vacuum is fully nested inside broader VAL/vacuum.
   - Loose no-reclaim and bearish-pressure VAL/vacuum are also nested/mostly nested inside broad VAL/vacuum.
   - Support-cleared has zero near-overlap with strict no-reclaim VAL/vacuum.
5. All-validated block:
   - Strategy:
     - `TraderRuleBlockValidatedOrderbookCombinedStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-10-44.zip`
   - Result:
     - `43` trades.
     - `58.1%` win rate.
     - `11.99%` return.
     - `8.39%` drawdown.
     - `1.46` profit factor.
   - Plain-English read:
     - It makes money, but adding nested VAL/vacuum leftovers dilutes the cleaner evidence.
6. Core support-removal block:
   - Strategy:
     - `TraderRuleBlockValidatedOrderbookCoreStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-11-52.zip`
   - Result:
     - `27` trades.
     - `66.7%` win rate.
     - `18.02%` return.
     - `3.68%` drawdown.
     - `2.19` profit factor.
   - Legs:
     - Support-cleared breakdown short:
       - `18` entries.
       - `61.1%` win rate.
       - `11.44%` contribution.
     - Strict no-reclaim VAL/vacuum short:
       - `9` entries.
       - `77.8%` win rate.
       - `6.59%` contribution.
7. Plain-English conclusion:
   - The useful combination is not "all orderbook shorts."
   - The useful combination is two separate trader scenarios:
     - support was cleared after a support/recent-low break,
     - or price broke below value into a downside vacuum with no reclaim warning.
   - Broader nested VAL/vacuum variants add trade count but reduce the quality of the block.
8. Verdict:
   - Promote the core orderbook block as the current best orderbook-only short block.
   - Keep broad VAL/vacuum, loose no-reclaim, and pressure variants as rework/watchlist leads, not as automatic additions to the block.
   - Next work should tailor exits per leg, then test risk sizing.

# 2026-06-05 - Core orderbook tailored-exit validation

1. Trader question:
   - If the core orderbook block has two different entry stories, does each story need its own stop/target/hold?
2. Exit-research output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_orderbook_core_exit_research.md`
3. Fast-harness selected exits:
   - Support-cleared breakdown:
     - `18h` hold.
     - `2.5%` stop.
     - `5%` target.
   - Strict no-reclaim VAL/vacuum:
     - `8h` hold.
     - `1.0%` stop.
     - `3.5%` target.
4. Implementation detail:
   - The prior combined strategy could not fully validate this because Freqtrade static `stoploss` is strategy-wide.
   - Added:
     - `TraderRuleBlockValidatedOrderbookCoreTailoredExitStrategy`
   - It uses `custom_stoploss` so the strict no-reclaim lead can use a `1.0%` stop while support-cleared keeps a wider `2.5%` stop.
5. Freqtrade result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-15-33.zip`
   - Result:
     - `27` trades.
     - `66.7%` win rate.
     - `19.22%` return.
     - `3.50%` drawdown.
     - `2.46` profit factor.
6. Plain-English conclusion:
   - Tailored exits improved the core orderbook short block.
   - The no-reclaim quick short benefits from tighter invalidation and more room for a larger target.
   - Support-cleared still needs the slower, wider breakdown-continuation exit.
7. Verdict:
   - Current best orderbook-only short block is:
     - support-cleared breakdown short,
     - strict no-reclaim VAL/vacuum short,
     - per-lead exits,
     - no nested broad VAL/vacuum leftovers.
   - Next work should test:
     - pressure-fade exit for support-cleared,
     - support-reclaim invalidation exit for no-reclaim,
     - risk sizing only after those exit checks.

# 2026-06-05 - Core orderbook invalidation-exit screen

1. Trader question:
   - Once a core orderbook short is open, should we exit early when the bearish reason starts disappearing?
2. Script/output:
   - Script:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_core_invalidation_exit.py`
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_core_invalidation_exit_20260605_core_orderbook_invalidation_exit.md`
   - Summary CSV:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_core_invalidation_exit_20260605_core_orderbook_invalidation_exit_summary.csv`
3. What was tested:
   - Support-cleared short:
     - early exit on support reclaim/rebuild warning,
     - early exit on bearish pressure fade,
     - combined reclaim-or-pressure-fade exit.
   - Strict no-reclaim VAL/vacuum short:
     - early exit on support reclaim/rebuild warning,
     - early exit on breakdown-failure warning,
     - combined reclaim-or-breakdown-failure exit.
4. Result:
   - The best-ranked profile for both legs was still the existing fixed tailored exit.
   - Support-cleared best:
     - `18` trades.
     - `66.7%` win rate.
     - `21.8%` direct-harness return.
     - `4.5%` direct-harness drawdown.
     - `3.02` profit factor.
   - Strict no-reclaim best:
     - `9` trades.
     - `77.8%` win rate.
     - `9.8%` direct-harness return.
     - `1.1%` direct-harness drawdown.
     - `9.46` profit factor.
5. Interpretation:
   - The direct screen did not find a better main exit than the existing Freqtrade-validated tailored exits.
   - Reclaim-warning exits can reduce drawdown in some variants, but they give up too much profit to promote as the default exit.
6. Verdict:
   - Keep `TraderRuleBlockValidatedOrderbookCoreTailoredExitStrategy` as the promoted orderbook-only short validation.
   - Park invalidation exits for later risk-protection research.
   - Next step should be narrowly designed risk sizing or new independent entry leads, not another generic invalidation sweep.

# 2026-06-05 - Core orderbook risk-sizing screen

1. Trader question:
   - After the core orderbook short entry is chosen, can extra metrics tell us which trades deserve more or less size?
2. Script/output:
   - Script:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260605_core_orderbook_risk_overlay.md`
3. Code correction:
   - The old orderbook sizing overlay referenced stale/non-current feature names.
   - It was updated to use the current core orderbook state columns:
     - `conf_ob_support_removed_strength`
     - `conf_ob_downside_vacuum_after_support_removed`
     - `conf_ob_support_cleared`
     - `conf_ob_bearish_pressure_agreement`
     - reclaim/rebuild/bounce warning columns.
4. What happened:
   - Baseline equal size:
     - `27` trades.
     - `33.8%` direct-harness return.
     - `4.5%` drawdown.
     - `3.66` profit factor.
   - Watchlist:
     - Signal-score sizing:
       - `43.9%` direct-harness return.
       - `5.7%` drawdown.
       - `3.72` profit factor.
     - Structure-agreement sizing:
       - `35.3%` direct-harness return.
       - `4.5%` drawdown.
       - `3.74` profit factor.
   - Rejected for now:
     - orderbook warning sizing,
     - core orderbook quality sizing,
     - directional volume sizing,
     - volatility sizing.
5. Interpretation:
   - The current orderbook metrics help identify entries, but they did not yet help decide trade size.
   - The model/rule signal score and structure agreement may be better first sizing inputs.
6. Verdict:
   - Keep score sizing and structure-agreement sizing on the watchlist.
   - Do not promote leverage/risk logic until it can be validated in a Freqtrade-compatible path.

# 2026-06-05 - Rectangle breakout VP-context isolated validation

1. Trader question:
   - When a rectangle/compression area breaks upward and VP/value-area context supports the move, does the breakout continue?
2. Why this was tested:
   - The per-rule confluence report showed `vp_context_agrees` improved the rectangle breakout long in the fast harness.
   - The previous combined confluence strategy mixed this long lead with a prior-day-low short filter, which made the result less interpretable.
3. Strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRectangleBreakoutVpConfluenceStrategy.py`
4. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_tailored_exit_selected_confluence.parquet`
   - Contains `9` VP-supported rectangle breakout long signal rows.
5. Freqtrade command:
   - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockRectangleBreakoutVpConfluenceStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20250501-20260510 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
6. Freqtrade result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-30-19.zip`
   - `5` actual trades.
   - `40.0%` win rate.
   - `7.97%` return.
   - `3.67%` max drawdown.
   - `2.73` profit factor.
   - Market change over the same window:
     - `-14.54%`.
7. Plain-English interpretation:
   - This is a useful sparse long-side clue: when rectangle breakout and VP support agree, the few actual trades were profitable.
   - It is not yet broad enough. The `9` signal rows clustered into only `5` actual trades because max-open-trades was `1`.
8. Verdict:
   - Add as a sparse validated long lead.
   - Rework/expand nearby rectangle/VP breakout variants before exit/risk sizing.
   - Do not merge into the main rule block yet.

# 2026-06-05 - Prior-day-low VP short confluence isolation

1. Trader question:
   - When price breaks the prior-day low with VP/pressure support and bearish volume agrees, does downside continuation work?
2. Why this was tested:
   - The first per-rule confluence report said prior-day-low shorts improved when volume agreed with the short.
   - The earlier combined confluence strategy mixed this with a sparse rectangle breakout long, so the short needed its own validation.
3. Signal files:
   - Strict volume-plus-structure:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_prior_day_low_vp_short_strong_volume_structure.parquet`
   - Broader volume-agrees:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_prior_day_low_vp_short_volume_agrees.parquet`
4. Strategies:
   - `TraderRuleBlockPriorDayLowVpShortVolumeStructureStrategy`
   - `TraderRuleBlockPriorDayLowVpShortVolumeAgreesStrategy`
5. Strict volume-plus-structure result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-33-32.zip`
   - `11` trades.
   - `45.5%` win rate.
   - `-3.57%` return.
   - `6.03%` drawdown.
   - `0.71` profit factor.
   - Year split:
     - 2025 slightly positive.
     - 2026 clearly negative.
6. Broader volume-agrees result:
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-33-45.zip`
   - `25` trades.
   - `60.0%` win rate.
   - `9.35%` return.
   - `8.62%` drawdown.
   - `1.44` profit factor.
   - Year split:
     - 2025 positive.
     - 2026 positive.
7. Plain-English interpretation:
   - The broad version worked because it kept enough valid downside continuation trades.
   - The stricter version probably over-filtered and kept a worse subset; it had more stop losses than winners.
8. Verdict:
   - Promote `prior_day_low_vp_short_volume_agrees` as a validated short lead.
   - Reject/rework the stricter volume-plus-structure version as currently defined.
   - Run invalidation/risk-reduction research next because the current `2%` stop created `9` stop-loss exits.
9. Exit research follow-up:
   - Fast-harness report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260605_prior_day_low_vp_short_volume_agrees_exit_research.md`
   - Direct sweep selected:
     - `48h` hold.
     - `2.5%` stop.
     - `5.0%` target.
   - Direct-harness selected result:
     - `25` trades.
     - `52.0%` win rate.
     - `13.46%` return.
     - `6.30%` drawdown.
     - `1.51` profit factor.
10. Tailored-exit Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockPriorDayLowVpShortVolumeAgreesTailoredExitStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-36-35.zip`
   - Result:
     - `25` trades.
     - `52.0%` win rate.
     - `2.19%` return.
     - `10.40%` drawdown.
     - `1.08` profit factor.
11. Exit verdict:
   - Reject the wider-target tailored exit.
   - Keep the original `48h` hold, `2.0%` stop, `2.0%` target as the current validated exit.
   - Next rework should be invalidation/risk reduction, not simply a wider target.

# 2026-06-05 - First validated multi-scenario trading block

1. Trader question:
   - If the currently validated short and long leads are allowed to compete for the same BTC slot, do they combine into a stronger multi-scenario rule block?
2. What was combined:
   - Core orderbook short block:
     - support cleared after breakdown.
     - strict VAL/vacuum no-support-reclaim warning short.
   - Prior-day-low VP short:
     - prior-day low breaks with broad bearish volume agreement.
   - Rectangle/VP long:
     - rectangle breakout long with VP-context support.
3. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_validated_multi_scenario_block.parquet`
4. Overlap handling:
   - Raw signals: `61`.
   - Unique decision rows after de-duplication: `52`.
   - Exact overlaps: `9`.
   - When prior-day-low short and support-cleared orderbook short fired on the same hour, the orderbook lead was kept because it has the stronger isolated result.
   - Overlap report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260605_validated_multi_scenario_block_overlaps.csv`
5. Strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockValidatedMultiScenarioStrategy.py`
6. Freqtrade artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-41-53.zip`
7. Result:
   - `44` trades.
   - `63.6%` win rate.
   - `41.18%` return.
   - `9.81%` max drawdown.
   - `2.26` profit factor.
   - Market change over the same window:
     - `-14.54%`.
8. Per-leg result inside the block:
   - Support-cleared orderbook short:
     - `15` trades, `66.7%` win rate, `15.13%` return.
   - Strict no-reclaim VAL/vacuum short:
     - `9` trades, `77.8%` win rate, `9.07%` return.
   - Prior-day-low VP short:
     - `15` trades, `60.0%` win rate, `8.24%` return.
   - Rectangle/VP long:
     - `5` trades, `40.0%` win rate, `8.73%` return.
9. Plain-English interpretation:
   - The separate leads did not destroy each other when combined.
   - The block now catches several different market stories:
     - support clears and price continues down,
     - liquidity vacuum/no-reclaim short,
     - prior-day low break with bearish volume agreement,
     - sparse rectangle breakout long.
   - This is a useful step toward the real goal of multiple scenario-specific entry rules.
10. Caution:
   - Drawdown rose versus the orderbook-only core block.
   - The weaker prior-day-low and rectangle legs need risk/conflict filters before adding more leads.
   - This is not yet a live strategy; it is a promoted research block.
11. Next action:
   - Add conflict/risk checks around the weaker legs.
   - Keep building toward `50-100` trader-readable leads, but promote only those that improve the block or fill a genuinely different market scenario.

# 2026-06-06 - Rectangle/VP expansion and support-break validation cycle

1. Trader question:
   - Can the current lead pool be expanded with more distinct long/short entries without blindly adding weak variants?
2. Candidate expansion scripts:
   - `trading_lead_rectangle_vp_expansion.py`
   - `trading_lead_support_break_short_expansion.py`
   - `trading_lead_orderbook_support_removal_short_refinement.py`
3. Direct-harness result:
   - Rectangle/VP selected block:
     - `221` variants tested.
     - `9` selected.
     - `66` selected-block trades.
     - `16.27%` direct-harness return.
   - Support-break selected block:
     - `160` variants tested.
     - `4` selected.
     - `92` selected-block trades.
     - `4.74%` direct-harness return.
   - Orderbook support-removal selected block:
     - `118` variants tested.
     - `2` selected.
     - `43` selected-block trades.
     - `26.16%` direct-harness return.
4. Freqtrade validation:
   - Rectangle/VP selected block:
     - Strategy:
       - `TraderRuleBlockRectangleVpGoalCycleStrategy`
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-48-03.zip`
     - Result:
       - `66` trades.
       - `56.1%` win rate.
       - `23.31%` return.
       - `9.80%` drawdown.
       - `1.51` profit factor.
   - Support-break selected block:
     - Strategy:
       - `TraderRuleBlockSupportBreakGoalCycleStrategy`
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-48-04.zip`
     - Result:
       - `92` trades.
       - `39.1%` win rate.
       - `-9.99%` return.
       - `16.69%` drawdown.
       - `0.87` profit factor.
5. Plain-English support-break interpretation:
   - The direct harness made the TLV2/support-break shorts look mildly useful.
   - In Freqtrade, the same selected block lost money and had too many stop losses.
   - Verdict:
     - Reject/rework the current support-break batch.
     - Do not promote it until the trader story is redesigned around cleaner bearish acceptance, better invalidation, or a narrower regime.
6. Isolated rectangle lead validation:
   - Created isolated signal files and strategy classes so overlapping rectangle variants could be judged separately.
7. Promoted isolated rectangle leads:
   - Rectangle break plus orderbook acceptance long:
     - Trader question:
       - When rectangle resistance breaks upward and the orderbook accepts the move, does the breakout continue?
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-50-02.zip`
     - Result:
       - `11` trades.
       - `72.7%` win rate.
       - `9.84%` return.
       - `1.30%` drawdown.
       - `4.41` profit factor.
     - Verdict:
       - Promote as sparse validated long lead.
   - Rectangle squeeze upward release long:
     - Trader question:
       - When a rectangle/compression squeeze breaks upward with volume expansion, does the release continue?
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-50-14.zip`
     - Result:
       - `10` trades.
       - `80.0%` win rate.
       - `9.42%` return.
       - `1.31%` drawdown.
       - `5.65` profit factor.
     - Verdict:
       - Promote as sparse validated long lead.
   - Rectangle break above VAH/value acceptance long:
     - Trader question:
       - When rectangle resistance breaks while price accepts above VAH/value area, does the upside continue?
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-50-26.zip`
     - Result:
       - `20` trades.
       - `60.0%` win rate.
       - `8.95%` return.
       - `1.69%` drawdown.
       - `2.78` profit factor.
     - Verdict:
       - Promote as validated long lead with better trade count.
8. Watchlist/rework lead:
   - Rectangle break below VAL short:
     - Artifact:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-50-38.zip`
     - Result:
       - `10` trades.
       - `40.0%` win rate.
       - `7.60%` return.
       - `4.63%` drawdown.
       - `2.13` profit factor.
     - Problem:
       - 2025 was negative and 2026 carried the result.
     - Verdict:
       - Park as rework/watchlist; test bearish orderbook acceptance or regime filter before promotion.
9. Important process note:
   - Freqtrade runs must not be launched in parallel when relying on default result filenames because artifacts can collide. Sequential reruns produced clean evidence files.

# 2026-06-06 - Expanded multi-scenario block with cleaner rectangle longs

1. Trader question:
   - If the old sparse rectangle long is replaced by the cleaner isolated rectangle/VP long leads, does the mixed BTC rule block improve?
2. What was combined:
   - Support-cleared orderbook short.
   - Strict no-reclaim VAL/vacuum short.
   - Prior-day-low VP short with bearish volume agreement.
   - Rectangle break plus orderbook acceptance long.
   - Rectangle squeeze upward release long.
   - Rectangle break above VAH/value acceptance long.
3. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_expanded_multi_scenario_block.parquet`
4. Overlap handling:
   - Raw signals: `93`.
   - Unique decision rows after de-duplication: `75`.
   - Overlap rows: `16`.
   - Overlap report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_expanded_multi_scenario_block_overlaps.csv`
5. Strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockExpandedMultiScenarioStrategy.py`
6. Artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-54-12.zip`
7. Freqtrade result:
   - `53` trades.
   - `66.0%` win rate.
   - `40.71%` return.
   - `7.67%` max drawdown.
   - `2.33` profit factor.
   - Positive in both 2025 and 2026.
8. Plain-English interpretation:
   - The expanded block is better quality than the previous mixed block because it added trades and reduced drawdown while keeping almost the same return.
   - The squeeze-release long contributed well inside the block.
   - The orderbook-acceptance and VAH-acceptance rectangle longs were weaker inside the block than standalone because overlap priority changed which rows actually traded.
9. Comparison:
   - Previous mixed block:
     - `44` trades, `63.6%` win rate, `41.18%` return, `9.81%` drawdown, `2.26` profit factor.
   - Expanded mixed block:
     - `53` trades, `66.0%` win rate, `40.71%` return, `7.67%` drawdown, `2.33` profit factor.
10. Verdict:
   - Promote as current best quality mixed block candidate.
   - Next refinement should test:
     - remove or reprioritize weak VAH-overlap rows,
     - preserve squeeze-release long,
     - keep prior-day-low short but add risk reduction around its stop-loss rows,
     - compare final block against the previous raw-return winner.

# 2026-06-06 - Refined squeeze-priority mixed block

1. Trader question:
   - If weak VAH-overlap rectangle rows are removed and rectangle squeeze-release rows get priority, does the mixed BTC rule block improve?
2. What was combined:
   - Support-cleared orderbook breakdown short.
   - Strict no-reclaim VAL/vacuum short.
   - Prior-day-low VP short with bearish volume agreement.
   - Rectangle squeeze upward release long.
   - Non-overlapping rectangle/orderbook-acceptance long.
3. What was deliberately removed:
   - `rect_break_above_vah_long` from the merged block.
   - Reason: it was useful standalone but weak inside the expanded block after overlap priority changed which rows actually traded.
4. Signal file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah.parquet`
5. Overlap report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah_overlaps.csv`
6. Strategy:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRefinedMultiScenarioStrategies.py`
   - Class: `TraderRuleBlockRefinedSqueezePriorityMultiScenarioStrategy`
7. Clean artifact:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-05_23-59-39.zip`
8. Extracted trades:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_refined_squeeze_priority_block.csv`
9. Freqtrade result:
   - `50` trades.
   - `70.0%` win rate.
   - `43.94%` return.
   - `7.67%` max drawdown.
   - `2.56` profit factor.
   - Positive in both annual buckets:
     - 2025: `27` trades, `66.7%` win rate, `2.07` profit factor.
     - 2026: `23` trades, `73.9%` win rate, `3.13` profit factor.
10. Per-leg result:
    - `lead__validated__support_break_support_cleared_short`: `15` trades, `14.38%` return, `66.7%` win rate.
    - `rect_squeeze_upper_release_long`: `10` trades, `11.25%` return, `80.0%` win rate.
    - `lead__validated__val_vacuum_no_support_reclaim_warning_short`: `9` trades, `8.70%` return, `77.8%` win rate.
    - `sieve_prior_day_low_break_vp_short`: `15` trades, `8.23%` return, `60.0%` win rate.
    - `rect_break_orderbook_accept_long`: `1` trade, `1.37%` return.
11. Comparison:
    - Previous mixed block:
      - `44` trades, `63.6%` win rate, `41.18%` return, `9.81%` drawdown, `2.26` profit factor.
    - Expanded mixed block:
      - `53` trades, `66.0%` win rate, `40.71%` return, `7.67%` drawdown, `2.33` profit factor.
    - Refined squeeze-priority block:
      - `50` trades, `70.0%` win rate, `43.94%` return, `7.67%` drawdown, `2.56` profit factor.
12. Plain-English interpretation:
    - The merged block improved when the weak VAH rectangle overlap was removed.
    - The squeeze-release long is currently the cleaner long-side rectangle behaviour.
    - The orderbook short legs remain useful and complementary.
    - The prior-day-low VP short still contributes profit, but it is the most obvious place to reduce stop-loss damage.
13. Verdict:
    - Promote as current best mixed BTC rule-block candidate.
    - Park it as a working candidate while broader lead discovery continues.
14. Next research action:
    - Continue toward `50` to `100` useful trader-readable leads.
    - Run confluence only per entry family, not as one global filter.
    - Develop tailored exits only after a lead survives Freqtrade validation.
    - Add risk/leverage overlays only after entry and exit behaviour are stable.

# 2026-06-06 - MTF custom-indicator lead mining and first validation

1. Trader question:
   - Can custom multi-timeframe indicators produce more independent BTC trading leads, instead of only expanding the current mixed block?
2. What was tested:
   - `96` rule families across:
     - VP levels and value-area behaviour.
     - TLV2 support/resistance breaks and retests.
     - BOS/CHoCH market structure.
     - triangle, wedge, compression, and rectangle pattern geometry.
   - Timeframes:
     - 1h.
     - 4h.
     - 1d.
3. Script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_mtf_indicator_mining.py`
4. Direct-mining output:
   - `480` threshold variants tested.
   - `41` selected direct candidates.
   - Direct combined selected block:
     - `372` trades.
     - `48.7%` win rate.
     - `58.3%` return.
     - `16.7%` drawdown.
     - `1.257` profit factor.
   - Verdict:
     - Do not promote the full mined block.
     - Use it as a lead source because it is too broad/noisy as a strategy.
5. Direct-mining files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining_summary.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining_selected.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_20260606_mtf_indicator_mining.md`
6. Strong direct candidates:
   - 4h TLV2 resistance break long.
   - 1d TLV2 resistance retest long.
   - 4h triangle upper-break long.
   - 4h VAL acceptance short.
   - 1d rectangle lower-break short.
   - 1d TLV2 support retest short.
7. Registry update:
   - `trading_lead_registry.py` now loads the MTF selected candidates.
   - Latest registry:
     - `100` leads.
     - `572` confluence checks.
     - `26` direct MTF mining candidates still present after one was promoted.
8. First promoted MTF lead:
   - `lead__validated__mtf_4h_tlv2_res_break_long`
9. Trader question:
   - When price breaks 4h TLV2 resistance with bullish pressure, does the breakout continue?
10. What a trader saw:
    - Price crossed a scored 4h TLV2 resistance line.
    - Bullish pressure and structure/VP context were not fighting the move.
11. Signal:
    - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_mtf_4h_tlv2_res_break_long_thr_045.parquet`
12. Strategy:
    - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockMtfIndicatorMiningStrategies.py`
    - Class: `TraderRuleBlockMtf4hTlv2ResBreakLongStrategy`
13. Freqtrade artifact:
    - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-12-18.zip`
14. Extracted trades:
    - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_4h_tlv2_res_break_long.csv`
15. Freqtrade result:
    - `50` trades.
    - `68.0%` win rate.
    - `29.24%` return.
    - `4.57%` max drawdown.
    - `2.99` profit factor.
    - 2025: `34` trades, `64.7%` win rate, `2.70` profit factor.
    - 2026: `16` trades, `75.0%` win rate, `3.64` profit factor.
16. Plain-English interpretation:
    - A strong 4h TLV2 resistance break is now a validated long-side structure lead.
    - It is not just a sparse pattern result: it had `50` trades and worked in both annual buckets.
17. Verdict:
    - Promote as a strong long-side candidate for confluence, exit, and risk work.
18. Next research action:
    - Validate the next MTF candidates one at a time.
    - Add orderbook confluence only if it answers a trader question, such as:
      - Was resistance removed above price?
      - Did support rebuild after the break?
      - Did spread/fragility warn against the long?
    - Test exit improvements after confluence, not before the lead is stable.

# 2026-06-06 - MTF candidate validation batch

1. Purpose:
   - Continue broadening the lead pool by validating direct-mined MTF candidates through Freqtrade.
2. Strategy file:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockMtfIndicatorMiningStrategies.py`
3. Validated lead: 1d TLV2 resistance retest long.
   - Lead id:
     - `lead__validated__mtf_1d_tlv2_res_retest_long`
   - Trader question:
     - After breaking daily TLV2 resistance, does a retest that holds continue higher?
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-16-13.zip`
   - Trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_1d_tlv2_res_retest_long.csv`
   - Result:
     - `20` trades.
     - `75.0%` win rate.
     - `18.72%` return.
     - `2.05%` max drawdown.
     - `4.11` profit factor.
   - Verdict:
     - Promote as a clean long-side continuation/retest lead.
4. Validated sparse lead: 4h triangle upper-break long.
   - Lead id:
     - `lead__validated__mtf_4h_triangle_upper_break_long`
   - Trader question:
     - When a 4h triangle breaks upward with volume, does it follow through?
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-16-28.zip`
   - Trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_4h_triangle_upper_break_long.csv`
   - Result:
     - `12` trades.
     - `83.3%` win rate.
     - `13.21%` return.
     - `1.97%` max drawdown.
     - `5.30` profit factor.
   - Verdict:
     - Promote as sparse long-side lead; check overlap with rectangle/squeeze breakouts before merging.
5. Filter-needed lead: 4h VP VAL acceptance short.
   - Lead id:
     - `lead__validated__mtf_4h_vp_val_accept_short`
   - Trader question:
     - When price accepts below 4h VAL with bearish pressure, does it continue toward lower value?
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-16-41.zip`
   - Trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_4h_vp_val_accept_short.csv`
   - Result:
     - `49` trades.
     - `55.1%` win rate.
     - `20.40%` return.
     - `7.02%` max drawdown.
     - `1.66` profit factor.
   - Verdict:
     - Keep as a useful short candidate, but require confluence filtering before mixed-block use.
     - Main concern is stop-loss damage: `9` stop-loss exits.
6. Parked watchlist: 1d rectangle lower-break short.
   - Lead id:
     - `lead__watchlist__mtf_1d_rectangle_lower_break_short_unstable`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-16-54.zip`
   - Trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_1d_rectangle_lower_break_short.csv`
   - Result:
     - `11` trades.
     - `54.5%` win rate.
     - `9.33%` return.
     - `2.18%` drawdown.
     - `2.81` profit factor.
   - Problem:
     - 2025 was negative and 2026 carried the result.
   - Verdict:
     - Watchlist/rework only.
7. Rework lead: 1d TLV2 support retest short.
   - Lead id:
     - `lead__rework__mtf_1d_tlv2_sup_retest_short_weak`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-17-06.zip`
   - Trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_freqtrade_trades_20260606_mtf_1d_tlv2_sup_retest_short.csv`
   - Result:
     - `37` trades.
     - `45.9%` win rate.
     - `5.94%` return.
     - `8.64%` drawdown.
     - `1.25` profit factor.
   - Problem:
     - 2025 was negative.
     - Drawdown is too high for the return.
   - Verdict:
     - Rework or reject current definition.
8. Registry state after batch:
   - `100` leads in latest registry.
   - `20` validated/rework entry records.
   - Top MTF additions:
     - 4h TLV2 resistance break long.
     - 1d TLV2 resistance retest long.
     - 4h triangle upper-break long.
     - 4h VAL acceptance short.
9. Next action:
   - Run overlap and confluence checks for validated MTF leads.
   - Prioritize orderbook/VP/volume confluence that answers a concrete trader question.
   - Do not add parked/rework shorts to the mixed block until repaired.

# 2026-06-06 - MTF per-rule confluence validation

1. Purpose:
   - Test whether newly validated MTF structure leads get cleaner when filtered by matching structure, volume, or orderbook confluence.
2. Important bug fixed:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_per_rule_confluence.py`
   - `load_signals()` previously dropped duplicate timestamps by `date` only.
   - Combined MTF files can legitimately contain different rules on the same hour.
   - Example duplicate same-hour pairs included:
     - `mtf_1d_tlv2_res_retest_long` and `mtf_4h_tlv2_res_break_long`.
     - `mtf_4h_tlv2_res_break_long` and `mtf_4h_triangle_upper_break_long`.
   - Fix:
     - Deduplicate by `date + rule_id` when `rule_id` exists.
   - Result:
     - Combined MTF signal input remained `179` rows instead of collapsing to `171`.
3. Confluence screen artifacts:
   - Summary:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260606_mtf_validated_confluence_dedup_fix_summary.csv`
   - Selected:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_per_rule_confluence_20260606_mtf_validated_confluence_dedup_fix_selected.csv`
   - Best-per-rule signal export:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_mtf_validated_confluence_dedup_fix_best_per_rule_selected_confluence.parquet`
   - Broader selected-confluence signal export:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_mtf_validated_confluence_dedup_fix_validated_exit_selected_confluence.parquet`
4. Strongest confluence findings:
   - `mtf_4h_tlv2_res_break_long`:
     - Best filter: orderbook direction and custom structure agreed.
     - Screen result: `11` filtered trades, `72.7%` win rate, `9.33` profit factor.
     - Plain English: the 4h resistance-break long is cleaner when the book and structure point the same way.
   - `mtf_4h_tlv2_res_break_long`:
     - Strong-volume plus structure filter also helped.
     - Screen result: `7` filtered trades, `57.1%` win rate, `7.60` profit factor.
     - Plain English: volume plus structure helps, but the sample is small.
   - `mtf_4h_vp_val_accept_short`:
     - Custom structure agreement improved the short.
     - Screen result: `19` filtered trades, `63.2%` win rate, `3.67` profit factor.
     - Plain English: accepting below VAL works better when broader structure agrees with the downside.
   - `mtf_1d_rectangle_lower_break_short`:
     - No support-reclaim warning improved the short.
     - Screen result: `7` filtered trades, `85.7%` win rate, `6.48` profit factor.
     - Plain English: a breakdown is more believable if broken support is not immediately being reclaimed.
5. Freqtrade validation strategies:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockMtfIndicatorMiningStrategies.py`
   - Added:
     - `TraderRuleBlockMtfValidatedConfluenceStrategy`
     - `TraderRuleBlockMtfBestPerRuleConfluenceStrategy`
6. Broader selected-confluence backtest:
   - Strategy:
     - `TraderRuleBlockMtfValidatedConfluenceStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-25-35.zip`
   - Result:
     - `59` trades.
     - `59.3%` win rate.
     - `3.26%` return.
     - `0.42%` max account drawdown.
     - `2.25` profit factor.
   - Plain English:
     - The broader filtered MTF block was positive and low-drawdown, but still includes some lower-quality filtered rows.
7. Best-filter-per-rule backtest:
   - Strategy:
     - `TraderRuleBlockMtfBestPerRuleConfluenceStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_00-26-33.zip`
   - Result:
     - `50` trades.
     - `62.0%` win rate.
     - `3.17%` return.
     - `0.39%` max account drawdown.
     - `2.66` profit factor.
   - Plain English:
     - Taking only the best confluence filter for each MTF rule reduced trade count, kept almost the same return, improved win rate, improved profit factor, and reduced drawdown.
8. Registry update:
   - Added:
     - `lead__validated__mtf_best_per_rule_confluence_block`
   - Validated/rework records:
     - `21`.
   - Latest registry:
     - `100` leads.
   - Latest confluence plan:
     - `572` checks.
9. Verdict:
   - Promote the best-filter MTF confluence block as a clean candidate for comparison and exit research.
   - Do not treat it as final trading logic yet.
10. Next action:
   - Compare this block against the refined squeeze-priority mixed block.
   - Build exit tests per lead:
     - structure invalidation exits.
     - next-level exits.
     - pressure-fade exits.
     - partial target exits.
   - Only after entry and exit behaviour are stable, test risk/leverage overlays from orderbook, regime, context/news, and volatility.

# 2026-06-06 - Tailored exit validation

1. Purpose:
   - Move from entry/confluence testing into the next objective stage: exits tailored to each successful entry family.
2. Code changes:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_exit_research.py`
     - Fixed combined-signal deduplication to keep `date + rule_id`.
     - Added MTF-specific exit grids so TLV2, triangle, VAL, and rectangle leads are not treated as generic prior-day breaks.
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockMtfIndicatorMiningStrategies.py`
     - Added per-rule custom stop support.
     - Added `TraderRuleBlockMtfBestPerRuleTailoredExitStrategy`.
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRefinedMultiScenarioStrategies.py`
     - Added `TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy`.
3. Exit-sweep artifacts:
   - MTF selected exits:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_mtf_best_confluence_exit_sweep_selected.csv`
   - MTF combined trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_mtf_best_confluence_exit_sweep_combined_trades.csv`
   - Refined selected exits:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_refined_squeeze_priority_exit_sweep_selected.csv`
   - Refined combined trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_refined_squeeze_priority_exit_sweep_combined_trades.csv`
4. MTF selected exit settings:
   - `mtf_4h_tlv2_res_break_long`:
     - hold `36h`, stop `1.4%`, target `5.0%`.
   - `mtf_4h_triangle_upper_break_long`:
     - hold `12h`, stop `2.0%`, target `2.0%`.
   - `mtf_4h_vp_val_accept_short`:
     - hold `36h`, stop `1.0%`, target `5.0%`.
   - `mtf_1d_rectangle_lower_break_short`:
     - hold `36h`, stop `1.0%`, target `7.5%`.
   - `mtf_1d_tlv2_sup_retest_short`:
     - hold `72h`, stop `3.5%`, target `5.0%`.
5. MTF tailored-exit Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockMtfBestPerRuleTailoredExitStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\mtf_tailored_20260606\backtest-result-2026-06-06_00-33-09.zip`
   - Result:
     - `38` trades.
     - `52.6%` win rate.
     - `3.49%` return.
     - `0.48%` max account drawdown.
     - `3.08` profit factor.
   - Plain English:
     - The MTF block became more selective and higher profit-factor, but the 4h VAL acceptance short is a problem under this exit: `15` trades, roughly flat/negative, only `26.7%` win rate.
   - Verdict:
     - Keep the MTF tailored block as a useful exit-research result.
     - Do not merge it unchanged until VAL acceptance short is reworked or filtered.
6. Refined squeeze-priority tailored-exit Freqtrade validation:
   - Strategy:
     - `TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy`
   - Artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\refined_tailored_20260606\backtest-result-2026-06-06_00-33-22.zip`
   - Result:
     - `44` trades.
     - `75.0%` win rate.
     - `4.56%` return.
     - `0.58%` max account drawdown.
     - `3.24` profit factor.
   - Plain English:
     - This is now the cleanest tested mixed BTC rule block by win rate/profit-factor balance.
     - Keeping entries mostly unchanged but tailoring exits improved the previous refined block.
7. Comparison against previous blocks:
   - Previous refined squeeze-priority block:
     - `50` trades, `70.0%` win rate, `43.94%` return in earlier recorded metric context, `7.67%` drawdown, `2.56` profit factor.
   - Current refined squeeze-priority tailored-exit backtest in the current Freqtrade 2025-01-01 to 2026-05-14 window:
     - `44` trades, `75.0%` win rate, `4.56%` return, `0.58%` account drawdown, `3.24` profit factor.
   - Important caution:
     - Some earlier percent-return figures came from different harness/stake/report contexts.
     - For future comparisons, prefer same-command Freqtrade runs over the same timerange and config.
8. Registry update:
   - Added:
     - `lead__validated__mtf_best_per_rule_tailored_exit_block`
     - `lead__validated__refined_squeeze_priority_tailored_exit_block`
   - Validated/rework records:
     - `23`.
   - Latest registry:
     - `100` leads.
9. Next action:
   - Use `TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy` as the current quality baseline for risk-overlay tests.
   - Rework/filter `mtf_4h_vp_val_accept_short`.
   - Begin risk/leverage overlays only as modifiers over stable entry+exit blocks:
     - orderbook agreement.
     - support reclaim/rebuild warning.
     - volatility expansion.
     - distance to invalidation/next level.
     - context/news only after source coverage is proven safe.

# 2026-06-06 - First risk overlay screen on tailored-exit blocks

1. Purpose:
   - Begin testing the final objective layer: risk/leverage sizing based on additional metrics.
   - This pass keeps entries and exits fixed, then tests whether simple size multipliers improve the trade block.
2. Scope/caution:
   - The input was the exit-research combined trade CSVs, not raw Freqtrade trade exports.
   - These are first-pass sizing screens, not production leverage rules.
   - Any useful overlay must be validated closer to Freqtrade execution before promotion.
3. Script:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
4. Refined squeeze-priority tailored-exit risk screen:
   - Input trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_refined_squeeze_priority_exit_sweep_combined_trades.csv`
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_refined_tailored_risk_overlay.csv`
   - Baseline equal-size simulated result:
     - `40` trades.
     - `70.0%` win rate.
     - `63.57%` compounded trade-return in this simulator.
     - `5.77%` simulated max drawdown.
     - `3.32` profit factor.
   - Watchlist overlays:
     - `score_confidence_0p5_to_1p5`
       - Return improved to `85.31%`, but drawdown worsened to `6.92%`.
       - Plain English: stronger signal score may justify more size, but it adds risk.
     - `directional_volume_size`
       - Return improved to `76.50%`, drawdown worsened to `7.20%`.
       - Plain English: sizing up when volume agrees may help, but needs drawdown control.
     - `structure_agreement_size`
       - Return improved to `68.80%`, small drawdown increase.
       - Plain English: structure agreement is a modest useful sizing clue.
     - `combined_conservative_size`
       - Return improved to `73.32%`, drawdown worsened to `6.47%`.
       - Plain English: combined sizing is worth more testing, but not yet final.
   - Rejected for now:
     - `orderbook_warning_size`
     - `core_orderbook_quality_size`
   - Reason:
     - Both reduced return; generic orderbook sizing is too blunt for this block.
5. MTF tailored-exit risk screen:
   - Input trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_mtf_best_confluence_exit_sweep_combined_trades.csv`
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_mtf_tailored_risk_overlay.csv`
   - Baseline equal-size simulated result:
     - `35` trades.
     - `68.6%` win rate.
     - `94.02%` compounded trade-return in this simulator.
     - `2.61%` simulated max drawdown.
     - `6.83` profit factor.
   - Watchlist overlays:
     - `score_confidence_0p5_to_1p5`
       - Return improved to `116.96%`, drawdown worsened to `3.39%`.
     - `directional_volume_size`
       - Return improved to `110.73%`, drawdown worsened to `3.26%`.
     - `structure_agreement_size`
       - Return improved to `110.04%`, drawdown worsened to `3.13%`.
     - `volatility_risk_off_size`
       - Return improved slightly to `98.08%`, drawdown unchanged, profit factor improved to `7.30`.
     - `combined_conservative_size`
       - Return improved to `107.16%`, drawdown worsened to `3.04%`.
   - Rejected for now:
     - `orderbook_warning_size`
     - `core_orderbook_quality_size`
   - Reason:
     - Both reduced return despite slightly lower drawdown.
6. Plain-English conclusion:
   - The first risk lesson is not "use orderbook for leverage" yet.
   - The useful early clues are:
     - stronger entry score,
     - volume agreeing with the trade direction,
     - structure agreeing with the trade direction,
     - volatility risk-off as a possible conservative helper.
   - Orderbook risk must be redesigned by entry story:
     - support-reclaim warning should affect support-break shorts,
     - resistance rebuild should affect breakout longs,
     - liquidity vacuum should affect continuation sizing,
     - generic orderbook multipliers are too blunt.
7. Next action:
   - Build per-entry-story risk overlays instead of broad generic overlays.
   - Validate candidate overlays against Freqtrade-style outputs before promoting to leverage/risk logic.
   - Do not use news/context sizing until source-specific coverage and extraction are trustworthy.

# 2026-06-06 - Story-specific risk overlay screen

1. Purpose:
   - Rework the first risk screen so orderbook, structure, and volume sizing are tied to the specific entry story.
   - This responds to the earlier finding that generic orderbook sizing was too blunt.
2. Code:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
3. Added overlays:
   - `story_specific_orderbook_size`
     - Support-break shorts:
       - size up only when support removal/downside vacuum/bearish pressure are strong and support-reclaim warning is low.
       - size down when support reclaim/rebuild/bid absorption appears.
     - Breakout longs:
       - size up only when resistance removal/upside vacuum/bullish pressure are strong and resistance-rejection warning is low.
       - size down when resistance rejection/ask absorption/breakout failure appears.
     - 4h VAL acceptance short:
       - stricter reduction when support reclaim appears.
   - `story_specific_structure_volume_size`
     - Breakout longs:
       - size up when bullish structure and bullish volume agree.
       - size down when bearish structure conflicts.
     - Squeeze/triangle longs:
       - size up when compression release and bullish volume agree.
     - Breakdown shorts:
       - size up when bearish structure and bearish volume agree.
       - size down when bullish structure conflicts.
   - `story_specific_conservative_size`
     - Blend of entry confidence, story-specific structure/volume, story-specific orderbook, and volatility.
4. Refined tailored block output:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_refined_tailored_story_risk_overlay.csv`
   - Baseline:
     - `40` trades.
     - `63.57%` simulated compounded trade-return.
     - `5.77%` simulated max drawdown.
     - `3.32` profit factor.
   - `story_specific_structure_volume_size`:
     - `73.75%` return.
     - `6.92%` drawdown.
     - `3.20` profit factor.
     - Verdict:
       - Watchlist. Better return, worse drawdown and slightly worse profit factor.
   - `story_specific_conservative_size`:
     - `69.19%` return.
     - `6.02%` drawdown.
     - `3.36` profit factor.
     - Verdict:
       - Watchlist. Modest improvement, less aggressive.
   - `story_specific_orderbook_size`:
     - `50.90%` return.
     - `4.53%` drawdown.
     - `3.44` profit factor.
     - Verdict:
       - Reject for now. It reduced drawdown but gave away too much return.
5. MTF tailored block output:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_mtf_tailored_story_risk_overlay.csv`
   - Baseline:
     - `35` trades.
     - `94.02%` simulated compounded trade-return.
     - `2.61%` simulated max drawdown.
     - `6.83` profit factor.
   - `story_specific_structure_volume_size`:
     - `115.87%` return.
     - `3.21%` drawdown.
     - `7.30` profit factor.
     - Verdict:
       - Strong watchlist. Better return and profit factor, with a small drawdown increase.
   - `story_specific_conservative_size`:
     - `97.31%` return.
     - `2.69%` drawdown.
     - `7.07` profit factor.
     - Verdict:
       - Watchlist. Modest improvement, lower risk than aggressive sizing.
   - `story_specific_orderbook_size`:
     - `59.77%` return.
     - `1.41%` drawdown.
     - `7.35` profit factor.
     - Verdict:
       - Reject for now. Drawdown reduced, but return reduction is too large.
6. Plain-English conclusion:
   - The risk layer is starting to separate useful and unhelpful ideas:
     - Useful: size up when structure and volume agree with the specific trade story.
     - Possibly useful: conservative blend for modest improvement.
     - Not useful yet: broad or story-specific orderbook size reduction, because it cuts too many good trades.
   - Orderbook should probably be a sharp invalidation/skip rule, not a smooth generic size reducer.
7. Next action:
   - Build orderbook warning overlays that only trigger on extreme thesis invalidation.
   - Test rule-specific skip/reduce logic:
     - support-break short plus strong support reclaim.
     - breakout long plus strong resistance rejection.
     - VAL acceptance short plus immediate support rebuild.
   - Keep `story_specific_structure_volume_size` as the best current risk-sizing candidate for the MTF branch.

# 2026-06-06 - Extreme-only orderbook invalidation guard screen

1. Purpose:
   - Test whether orderbook works better as a sharp safety brake than as a broad size reducer.
2. Code:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_overlay.py`
3. Added overlays:
   - `extreme_orderbook_invalidation_guard`
     - Support-break shorts:
       - reduce size only when support reclaim/rebuild/bid absorption/breakdown failure is strong.
     - Breakout longs:
       - reduce size only when resistance rejection/ask absorption/breakout failure is strong.
   - `structure_volume_with_extreme_ob_guard`
     - Apply story-specific structure/volume sizing first.
     - Then apply the extreme orderbook invalidation guard.
4. Refined tailored block:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_refined_tailored_extreme_ob_risk_overlay.csv`
   - Baseline:
     - `63.57%` simulated compounded return.
     - `5.77%` simulated drawdown.
     - `3.32` profit factor.
   - `extreme_orderbook_invalidation_guard`:
     - `62.36%` return.
     - `5.77%` drawdown.
     - `3.29` profit factor.
     - Verdict:
       - Reject for now. It did not improve the refined block.
   - `structure_volume_with_extreme_ob_guard`:
     - `72.16%` return.
     - `6.92%` drawdown.
     - `3.16` profit factor.
     - Verdict:
       - Watchlist, but less attractive than structure/volume sizing alone on this block.
5. MTF tailored block:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_risk_overlay_20260606_mtf_tailored_extreme_ob_risk_overlay.csv`
   - Baseline:
     - `94.02%` simulated compounded return.
     - `2.61%` simulated drawdown.
     - `6.83` profit factor.
   - `extreme_orderbook_invalidation_guard`:
     - `84.07%` return.
     - `2.32%` drawdown.
     - `6.98` profit factor.
     - Verdict:
       - Reject for now. Drawdown improved but return loss is too high.
   - `structure_volume_with_extreme_ob_guard`:
     - `102.35%` return.
     - `2.78%` drawdown.
     - `7.55` profit factor.
     - Verdict:
       - Watchlist. This is the best risk-balanced MTF overlay so far.
6. Plain-English conclusion:
   - Orderbook risk logic is still not a standalone leverage tool.
   - The most useful role so far is a safety brake layered on top of a stronger structure/volume sizing rule.
   - On the MTF block, this gives up some return versus pure structure/volume sizing, but improves profit factor and keeps drawdown closer to baseline.
7. Next action:
   - Promote two risk overlay candidates for closer validation:
     - return-seeking: `story_specific_structure_volume_size`.
     - risk-balanced: `structure_volume_with_extreme_ob_guard`.
   - Validate them in a strategy-level or closer Freqtrade-equivalent path before any live-risk interpretation.

# 2026-06-06 - Strategy-level risk sizing validation

1. Purpose:
   - Validate whether risk overlays still look useful when implemented as Freqtrade strategy stake sizing instead of only CSV trade-return simulation.
2. Code/files:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_risk_signal_exports.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockResearchStrategy.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockRefinedMultiScenarioStrategies.py`
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockMtfIndicatorMiningStrategies.py`
3. Exported risk signal files:
   - `trading_lead_signals_20260606_refined_squeeze_priority_story_sv_risk.parquet`
   - `trading_lead_signals_20260606_refined_squeeze_priority_story_sv_extreme_ob_risk.parquet`
   - `trading_lead_signals_20260606_mtf_best_story_sv_risk.parquet`
   - `trading_lead_signals_20260606_mtf_best_story_sv_extreme_ob_risk.parquet`
4. Validation lesson:
   - With small BTC futures stakes, Freqtrade/Binance amount precision rounds entries to roughly 0.001 BTC steps.
   - This can hide stake multipliers. Example: 65, 100, and 120 USDT requested sizes can all collapse to nearly the same actual BTC futures stake near 100k BTC.
   - For risk-sizing validation, use larger fixed stakes and wallet, or interpret small-stake tests as invalid for sizing.
5. Fixed-size validation command shape:
   - `--stake-amount 1000`
   - `--dry-run-wallet 10000`
   - `--cache none`
   - `--pairs BTC/USDT:USDT`
   - `--timerange 20250501-20260523`
6. Refined block results:
   - Baseline artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\refined_tailored_fixed1000_20260606-2026-06-06_00-54-09.zip`
   - Baseline:
     - `44` trades.
     - `75.0%` win.
     - `4.51%` return.
     - `0.86%` drawdown.
     - `2.87` profit factor.
   - Story-specific structure/volume artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\refined_story_sv_risk_fixed1000_20260606-2026-06-06_00-54-09.zip`
   - Story-specific structure/volume:
     - `44` trades.
     - `75.0%` win.
     - `4.85%` return.
     - `1.04%` drawdown.
     - `2.72` profit factor.
     - Plain English:
       - It made more money by sizing up some winners, but it also increased drawdown and reduced profit factor.
       - Keep as a return-seeking watchlist, not a cleaner risk rule.
   - Structure/volume plus extreme orderbook guard artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\refined_story_sv_extreme_ob_risk_fixed1000_20260606-2026-06-06_00-54-09.zip`
   - Structure/volume plus extreme orderbook guard:
     - `44` trades.
     - `75.0%` win.
     - `4.58%` return.
     - `1.04%` drawdown.
     - `2.63` profit factor.
     - Plain English:
       - The orderbook guard did not protect enough to justify the lost return and worse profit factor.
7. MTF block results:
   - Baseline artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\mtf_tailored_fixed1000_20260606-2026-06-06_00-54-08.zip`
   - Baseline:
     - `38` trades.
     - `52.6%` win.
     - `3.89%` return.
     - `0.38%` drawdown.
     - `3.34` profit factor.
   - Story-specific structure/volume artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\mtf_story_sv_risk_fixed1000_20260606-2026-06-06_00-54-27.zip`
   - Story-specific structure/volume:
     - `38` trades.
     - `52.6%` win.
     - `4.18%` return.
     - `0.36%` drawdown.
     - `3.39` profit factor.
     - Plain English:
       - This is the strongest risk-sizing candidate so far.
       - It made more money and slightly reduced drawdown without changing entries/exits.
   - Structure/volume plus extreme orderbook guard artifact:
     - `C:\FreqTradeStuff\user_data\backtest_results\mtf_story_sv_extreme_ob_risk_fixed1000_rerun_20260606-2026-06-06_00-55-07.zip`
   - Structure/volume plus extreme orderbook guard:
     - `9` trades.
     - `33.3%` win.
     - `-0.02%` return.
     - `0.36%` drawdown.
     - `0.95` profit factor.
     - Plain English:
       - Invalid/parked. The signal file contains 53 rows, so the 9-trade output is not comparable to the 38-trade baseline.
       - Diagnose missing-entry behaviour before using this variant.
8. Current risk-sizing conclusion:
   - Use story-specific structure/volume confirmation as the main risk-sizing branch.
   - Refined block: return-seeking only, not yet cleaner.
   - MTF block: validated improvement.
   - Orderbook guard: not ready. It needs a more precise invalidation rule before becoming leverage/risk logic.

# 2026-06-06 - Entry-sieve runtime candidate leads added

1. Objective:
   - Borrow plausible good trades from existing entry-sieve runs and move them into the same trader-lead/confluence/exits/risk pipeline.
   - This supports the user's target of `50-100` trader-readable BTC leads before confluence, exits, and risk/leverage work.
2. New code:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_sieve_runtime_candidate_leads.py`
   - Updated:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_registry.py`
3. Scanner input:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\launcher_v2\runtime\entry_sieve\results\*.jsonl`
4. Scanner filters:
   - completed `ok` rows only
   - at least `5` trades
   - positive return
   - profit factor at least `1.20`, or win rate at least `55%`
5. Scanner outputs:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606.md`
6. Scanner result:
   - `400` usable candidate rows after strategy-level dedupe.
   - Latest sieve3 batch `20260605T221118_entry_sieve3_novel_mtf_batch_01` contained only hyperopt errors, so it did not contribute candidates.
   - Older completed sieve runs contributed the candidates.
7. Registry update:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trading_lead_registry.py --tag 20260606_sieve_runtime_update --max-leads 100 --min-leads 50`
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260606_sieve_runtime_update.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260606_sieve_runtime_update.md`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_latest.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_plan_20260606_sieve_runtime_update.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_plan_20260606_sieve_runtime_update.md`
8. Registry result:
   - `100` total leads.
   - `572` planned confluence checks.
   - Branch mix:
     - `structure_volume`: `28`
     - `sieve_borrowed`: `25`
     - `downside_risk`: `18`
     - `orderbook_state`: `18`
     - `multi_scenario_block`: `6`
     - `regime_confluence`: `5`
   - Stage mix:
     - `sieve_runtime_candidate`: `25`
     - `lead_family`: `17`
     - `direct_rule_mining_candidate`: `17`
     - `next_cycle_candidate`: `11`
     - `concept_evidence`: `10`
     - validated Freqtrade/confluence/block stages make up the remaining rows.
9. Important interpretation:
   - These sieve-runtime candidates are seed leads, not final trading rules.
   - Several high-return candidates have low win rate but positive profit factor. That can still be useful for trend/continuation logic, but each must be retested in trader-rule form before confluence or exit promotion.
   - Borrowed sieve leads should not be merged directly into the best block until overlap, confluence, and tailored exits are checked.
10. Next candidate groups to work:
   - BOS bullish continuation.
   - TLV2/VP resistance break longs.
   - TLV2 support breakdown shorts.
   - Rectangle breakdown shorts.
   - Triangle squeeze breakdown/upper-break leads.
   - Prior-week/month high break VP longs.
11. Validation:
   - `py_compile` passed for:
     - `trader_sieve_runtime_candidate_leads.py`
     - `trading_lead_registry.py`

# 2026-06-06 - BTC sieve-runtime confluence pass

1. Objective:
   - Start testing confluence on borrowed sieve leads using actual BTC trades.
   - Avoid relying on multi-pair headline results where BTC may not be the source of the edge.
2. Code changes:
   - `trader_sieve_runtime_candidate_leads.py` now supports:
     - `--pair BTC/USDT:USDT`
     - `--pair-required`
     - BTC-specific trade count, win rate, compounded return, profit factor, and drawdown from backtest ZIP contents.
   - Added:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trader_sieve_runtime_signal_exports.py`
   - Fixed:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\trading_lead_confluence_checks.py`
     - Bug: `worth_more_testing` verdicts were being overwritten to `watchlist`.
3. BTC-only candidate scan:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trader_sieve_runtime_candidate_leads.py --tag 20260606_btc_only --pair BTC/USDT:USDT --pair-required --min-trades 5 --min-profit-factor 1.2 --min-return 0.0`
   - Result:
     - `141` BTC-specific candidate strategies.
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606_btc_only.md`
4. Registry rebuild:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trading_lead_registry.py --tag 20260606_btc_sieve_runtime_update --max-leads 100 --min-leads 50`
   - Result:
     - `100` total leads.
     - `25` BTC-specific sieve-runtime candidates.
     - `572` planned confluence checks.
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_registry_20260606_btc_sieve_runtime_update.md`
5. BTC signal export:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trader_sieve_runtime_signal_exports.py --tag 20260606_btc_top30 --top 30 --min-btc-trades 5 --pair BTC/USDT:USDT`
   - Result:
     - `30` selected BTC sieve candidate strategies.
     - `1,459` BTC trade rows.
     - `1,459` signal rows.
   - Alignment:
     - Signal `date`/`signal_date` is `open_date - 1h`.
     - Plain English: confluence features are evaluated at the hour before the actual backtest entry.
   - Outputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_btc_top30_trades.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_btc_top30_signals.parquet`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_btc_top30.md`
6. Confluence test:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trading_lead_confluence_checks.py --trades user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_btc_top30_trades.csv --features user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet --tag 20260606_btc_sieve_top30_confluence_fixed --min-trades 8`
   - Result:
     - `1,459` input trades.
     - `868` result rows.
     - `86` worth more testing.
     - `47` watchlist.
     - `276` too few trades.
     - `428` reject for now.
     - `31` baselines.
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260606_btc_sieve_top30_confluence_fixed.md`
7. Broad all-entry findings:
   - Baseline borrowed-sieve BTC set:
     - `1,459` trades.
     - `50.79%` win rate.
     - `0.788%` average return per trade.
     - `1.7003` profit factor.
   - Structure trigger agrees:
     - `28` trades.
     - `64.29%` win rate.
     - `1.447%` average return.
     - `2.7255` profit factor.
     - Plain English: strong but selective; it greatly cuts trade count.
   - Strong volume plus structure:
     - `12` trades.
     - `66.67%` win rate.
     - `1.369%` average return.
     - `2.7153` profit factor.
     - Plain English: good but sparse.
   - Orderbook plus structure:
     - `16` trades.
     - `62.50%` win rate.
     - `1.269%` average return.
     - `2.4817` profit factor.
     - Plain English: orderbook helps when paired with structure, not broadly alone.
   - Compression plus range break:
     - `683` trades.
     - `54.76%` win rate.
     - `0.912%` average return.
     - `1.8680` profit factor.
     - Plain English: this is the best broad filter because it keeps many trades and improves quality.
   - Regime plus range break:
     - `921` trades.
     - `53.96%` win rate.
     - `0.864%` average return.
     - `1.8079` profit factor.
   - Range break agrees:
     - `922` trades.
     - `53.90%` win rate.
     - `0.859%` average return.
     - `1.8022` profit factor.
   - VP plus range break:
     - `898` trades.
     - `53.79%` win rate.
     - `0.855%` average return.
     - `1.7970` profit factor.
8. Important caveats:
   - Some individual-rule rows have huge profit factor because filtered rows had no losses; these are promising but not proven.
   - Generic orderbook-only confirmation was bad across all borrowed entries:
     - `147` trades.
     - `47.62%` win rate.
     - `1.4587` profit factor.
   - Orderbook should remain story-specific, not a universal filter.
9. Next-stage queue:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trading_lead_next_stage_queue.py --registry user_data\research_news_data\context_features\reports\trading_lead_registry_latest.csv --confluence user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260606_btc_sieve_top30_confluence_fixed.csv --tag 20260606_btc_sieve_confluence_fixed --max-tasks 80`
   - Result:
     - `64` tasks.
     - `12` tailored confluence retests.
     - `16` exit research tasks.
     - `36` per-lead confluence tasks.
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_next_stage_queue_20260606_btc_sieve_confluence_fixed.md`
10. Next:
   - Export filtered signals for the strongest broad filters:
     - `compression_plus_range_break`
     - `regime_plus_range_break`
     - `range_break_agrees`
     - `vp_plus_range_break`
     - `structure_trigger_agrees`
   - Run exit research on the strongest filtered families.

# 2026-06-06 - BTC sieve compression/range branch tailored exits and Freqtrade validation

1. Trader question:
   - If borrowed BTC sieve entries only fire when compression/range-break context agrees, and each rule family gets its own stop/target/hold exit, does the block produce usable BTC trades across 2020-2025?
2. What was looked at:
   - BTC/USDT futures 1h entries from top BTC sieve candidates.
   - Confluence filter:
     - `compression_plus_range_break`.
   - Exit families:
     - Per-rule stop loss.
     - Per-rule take profit.
     - Per-rule maximum hold time.
3. Signal exports:
   - Initial filtered signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break.parquet`
     - `683` rows.
     - `30` rule families.
   - Selected-exit-rules-only signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_selected_exit_rules.parquet`
     - `653` rows.
     - `21` rule families.
4. Exit research direct harness:
   - Command:
     - `python user_data\Custom_Launcher\research\context_features\trading_lead_exit_research.py --signals user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break.parquet --features user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet --tag 20260606_btc_sieve_compression_range_exit --min-trades 8`
   - Output files:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_btc_sieve_compression_range_exit.md`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_btc_sieve_compression_range_exit_selected.csv`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_btc_sieve_compression_range_exit_combined_trades.csv`
   - Direct-harness combined result:
     - `237` trades.
     - `66.67%` win rate.
     - `1.571%` average return per trade.
     - `3.6903` profit factor.
     - `-8.34%` max drawdown.
   - Caveat:
     - The shell command timed out after output generation, but all files were written.
     - Direct harness is useful for selecting exits, not final validation.
5. Freqtrade validation issue:
   - First strategy version used all `683` signals.
   - Only `21` rule families had tailored exits.
   - One unhandled family opened a long trade and held until forced exit, making the first Freqtrade run invalid.
   - Fix:
     - Filter signals to selected exit-rule families before strategy validation.
6. Strategy validation:
   - Added:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveCompressionRangeStrategy.py`
   - Command:
     - `.venv\Scripts\python.exe -m freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockBtcSieveCompressionRangeStrategy --strategy-path user_data\strategies --timeframe 1h --timerange 20200101-20260115 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-24-54.zip`
   - Corrected result:
     - `270` trades.
     - `44.8%` win rate.
     - `0.52%` average profit per trade.
     - `264.54%` total return.
     - `1.40` profit factor.
     - `27.55%` max drawdown.
     - `230` long trades.
     - `40` short trades.
7. Stronger sub-families in Freqtrade output:
   - `sieve1_bos_bull_continuation_long_1d`:
     - `7` entries, `71.4%` win rate, `36.11%` total contribution.
   - `sieve2_reversal_double_top_present_short_1h`:
     - `14` entries, `64.3%` win rate, `31.72%` total contribution.
   - `sieve1_vp_lvn_fast_traverse_long_1h`:
     - `11` entries, `63.6%` win rate, `19.57%` total contribution.
   - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_1d`:
     - `6` entries, `66.7%` win rate, `12.62%` total contribution.
8. Weak/dragging sub-families in Freqtrade output:
   - `sieve1_bos_bull_continuation_long_1h`:
     - `32` entries, `28.1%` win rate, `-2.06%` total contribution.
   - `sieve2_multi2_tlv2_vp_res_break_vp_bullctx_long_1h`:
     - `5` entries, `0%` win rate, `-12.08%` total contribution.
   - `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h`:
     - `14` entries, `57.1%` win rate, `-18.36%` total contribution.
   - `sieve2_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard`:
     - `16` entries, `25.0%` win rate, `-20.47%` total contribution.
9. Plain-English result:
   - The overall branch works, but it is not clean enough.
   - The next refinement should remove or re-filter the dragging rule families rather than trying to polish the whole block blindly.
   - This validates the user goal path: borrowed sieve lead -> confluence -> tailored exits -> Freqtrade validation.
10. Next:
   - Run the same tailored-exit and validation sequence for:
     - `regime_plus_range_break`
     - `range_break_agrees`
     - `vp_plus_range_break`
   - Build a cleaner selected-family strategy variant from the strongest sub-families.

# 2026-06-06 - Broad BTC sieve branch comparison and quality-family refinement

1. Trader question:
   - Which broad confluence filter gives the best Freqtrade-valid BTC lead block after tailored exits?
2. What was looked at:
   - Same top BTC sieve candidate pool.
   - Same BTC futures 1h validation window:
     - `2020-01-01` to `2026-01-15`.
   - Same validation settings:
     - BTC/USDT futures.
     - `max-open-trades 1`.
     - per-rule stop/target/hold exits selected by exit research.
3. Reusable strategy infrastructure added:
   - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
   - Classes:
     - `TraderRuleBlockBtcSieveRegimeRangeStrategy`
     - `TraderRuleBlockBtcSieveRangeBreakAgreesStrategy`
     - `TraderRuleBlockBtcSieveVpRangeStrategy`
     - `TraderRuleBlockBtcSieveCompressionRangeQualityFamiliesStrategy`
   - Purpose:
     - Avoid duplicating large per-rule exit dictionaries.
     - Load selected exit settings from the exit-research CSV.
4. Direct harness branch comparison:
   - Compression/range:
     - `237` trades, `66.67%` win rate, `1.571%` average return, `3.690` profit factor, `-8.34%` drawdown.
   - Regime/range:
     - `280` trades, `68.93%` win rate, `1.450%` average return, `2.889` profit factor, `-10.33%` drawdown.
   - Range-break-agrees:
     - `280` trades, `68.93%` win rate, `1.448%` average return, `2.887` profit factor, `-10.33%` drawdown.
   - VP/range:
     - `270` trades, `69.26%` win rate, `1.533%` average return, `3.076` profit factor, `-10.33%` drawdown.
5. Freqtrade branch comparison:
   - Compression/range:
     - `270` trades, `264.54%` return, `44.8%` win rate, `1.40` profit factor, `27.55%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-24-54.zip`
   - Regime/range:
     - `329` trades, `269.76%` return, `48.6%` win rate, `1.31` profit factor, `29.06%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-38-18.zip`
   - Range-break-agrees:
     - `330` trades, `245.46%` return, `48.2%` win rate, `1.28` profit factor, `29.32%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-38-41.zip`
   - VP/range:
     - `317` trades, `247.60%` return, `47.6%` win rate, `1.30` profit factor, `26.14%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-39-02.zip`
6. Why direct harness and Freqtrade disagree:
   - The direct harness tests selected entries in a simplified one-at-a-time way.
   - Freqtrade applies actual strategy mechanics:
     - max-open-trades behaviour,
     - exact stoploss handling,
     - trailing stop naming through custom stoploss,
     - signal collision/overlap,
     - stake compounding.
   - Therefore direct harness is useful for finding exit candidates, but Freqtrade decides whether a branch is truly usable.
7. Quality-family refinement:
   - Built from compression/range Freqtrade output.
   - Kept only rule families with:
     - at least `5` Freqtrade trades,
     - positive total contribution,
     - positive average return.
   - Kept `11` rule families.
   - Signal file:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_positive_families.parquet`
   - Rule-family audit:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_quality_families_20260606_btc_sieve_compression_range.csv`
8. Quality-family Freqtrade result:
   - Strategy:
     - `TraderRuleBlockBtcSieveCompressionRangeQualityFamiliesStrategy`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-40-38.zip`
   - Result:
     - `227` trades.
     - `371.08%` total return.
     - `46.7%` win rate.
     - `0.74%` average profit per trade.
     - `1.56` profit factor.
     - `20.46%` max drawdown.
     - `187` long trades, `40` short trades.
   - Year breakdown:
     - 2020: `44` trades, `1042.646` USDT, profit factor `3.25`.
     - 2021: `45` trades, `608.978` USDT, profit factor `1.45`.
     - 2022: `25` trades, `82.305` USDT, profit factor `1.09`.
     - 2023: `26` trades, `1200.685` USDT, profit factor `2.96`.
     - 2024: `40` trades, `-73.974` USDT, profit factor `0.96`.
     - 2025: `46` trades, `908.581` USDT, profit factor `1.62`.
9. Plain-English interpretation:
   - The quality-family branch is the current best lead block.
   - It still has weak periods, especially 2024, but it is positive in most years and has substantially better drawdown than the broader branches.
   - It is not final because it still depends heavily on a few continuation/breakout families and needs risk controls.
10. Next:
   - Add risk/leverage overlays to the quality-family branch:
     - size up when structure/volume confirms the entry story,
     - size down or skip when story-specific orderbook invalidates the setup,
     - do not use generic orderbook caution as a broad size reducer.
   - Then inspect exits per retained rule family to see if a few families need different hold/stop/target logic.

# 2026-06-06 - Storage check after BTC sieve validation pass

1. Storage sizes:
   - `C:\FreqTradeStuff\user_data\backtest_results`: about `16.7 MB`.
   - `C:\FreqTradeStuff\user_data\models`: about `28.34 GB`.
   - `C:\FreqTradeStuff\user_data\freqaimodels`: effectively empty.
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports`: about `395 MB`.
2. Largest model folders:
   - `context-freqai-structure-lightgbm-20260520`: about `3.526 GB`.
   - `context-freqai-event-context-lgbm-20260521`: about `3.310 GB`.
   - `context-freqai-event-combined-lgbm-20260521`: about `3.310 GB`.
   - `context-freqai-event-price-lgbm-20260521`: about `3.309 GB`.
3. Action:
   - No deletion performed.
   - If disk cleanup is needed, old `user_data\models` FreqAI/model artifacts are the main candidate, not current backtest results.

# 2026-06-06 - BTC sieve quality-family risk overlay validation

1. Trader question:
   - We already have a promising BTC lead block. Can extra information decide when to trade larger or smaller?
2. What was looked at:
   - Same quality-filtered compression/range BTC sieve entries.
   - Same tailored per-rule exits.
   - Two sizing overlays:
     - story-specific structure/volume sizing,
     - story-specific structure/volume sizing with an orderbook invalidation guard.
3. What the trader would have seen:
   - For resistance-breakout longs:
     - bullish structure,
     - volume agreement,
     - price above/through value areas,
     - optional compression release.
   - For reclaim/continuation/LVN-traverse longs:
     - support reclaim or bullish continuation context,
     - volume agreement,
     - thin upside volume-profile travel where available.
   - For POC rejection and double-top shorts:
     - bearish structure/volume agreement,
     - price below value area or stacked resistance.
   - For orderbook guard:
     - resistance removed/upside vacuum/bullish pressure for longs,
     - support removed/downside vacuum/bearish pressure for shorts,
     - size reduction when the book strongly suggested rejection or support rebuild against the trade.
4. Generated signal files:
   - Story-only sizing:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_quality_story_risk.parquet`
     - `361` rows, `36` size changes, average multiplier `1.0105`.
   - Story plus orderbook guard:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_sieve_quality_story_ob_guard_risk.parquet`
     - `361` rows, `77` size changes, average multiplier `0.9534`.
5. Fixed-stake Freqtrade comparison:
   - Equal fixed stake:
     - `227` trades, `16.35%` return, `46.7%` win rate, `1.76` profit factor, `1.83%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-47-42.zip`
   - Story-only sizing:
     - `227` trades, `16.02%` return, `46.7%` win rate, `1.74` profit factor, `1.81%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-48-02.zip`
   - Story plus orderbook guard:
     - `227` trades, `14.98%` return, `46.7%` win rate, `1.73` profit factor, `2.02%` drawdown.
     - Output:
       - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_01-48-21.zip`
6. Result in plain English:
   - The current sizing rules did not know enough to improve the already-profitable lead block.
   - Story-only sizing was close but not better.
   - The orderbook guard was harmful in this version because it reduced exposure in places where the lead block still made money, or failed to reduce the right losers.
7. Verdict:
   - Entry/exit block:
     - keep and continue refining.
   - Current story-only risk sizing:
     - park for rework.
   - Current orderbook guard:
     - reject for this block as currently defined.
8. Next step:
   - Build a loser/winner audit by retained rule family and by exact guard reason.
   - Test skip/reduce rules before leverage:
     - "do not take this family when condition X is present",
     - "halve size only for this family when condition Y is present",
     - "size up only where the rule family has proven positive with the confirming state".

# 2026-06-06 - Trade-management drawdown reduction: orderbook crash exit

1. Trader question:
   - If a long trade is already open, can the orderbook warn that a crash or downside escalation is developing so we exit earlier?
2. What was looked at:
   - Same quality-filtered compression/range BTC sieve entries.
   - Same selected per-rule exits.
   - Frozen 1h confluence feature cache.
   - Early-exit concepts:
     - orderbook invalidation,
     - orderbook crash/downturn escalation,
     - volume failure,
     - structure level failure,
     - volatility shock,
     - time-to-confirm,
     - MTF disagreement skip,
     - chop/range skip,
     - post-break acceptance skip,
     - support/resistance proximity skip.
3. Direct-harness output:
   - Summary:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_trade_management_20260606_btc_quality_first_trade_management_v3_summary.csv`
   - Trades:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_trade_management_20260606_btc_quality_first_trade_management_v3_trades.csv`
   - Report:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_trade_management_20260606_btc_quality_first_trade_management_v3.md`
4. Direct-harness result:
   - Baseline selected exits:
     - `221` trades, `3606.99%` return, `-8.90%` drawdown, `3.789` profit factor.
   - Orderbook crash/downturn exit:
     - `242` trades, `3128.28%` return, `-8.60%` drawdown, `4.155` profit factor.
   - Other early exits:
     - reduced drawdown more,
     - but gave up too much return to promote immediately.
5. Freqtrade implementation:
   - Strategy:
     - `TraderRuleBlockBtcSieveQualityOrderbookCrashExitStrategy`
   - File:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
   - Logic:
     - same signal file as the quality-family baseline,
     - same exit CSV,
     - merges only needed frozen feature columns,
     - exits long trades if at least two downside-escalation components are active:
       - bearish orderbook pressure,
       - support removed/cleared,
       - downside liquidity vacuum,
       - spread/single-venue fragility,
       - recent price breakdown.
6. Freqtrade result:
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_20-54-32.zip`
   - Baseline quality-family block:
     - `227` trades, `371.08%` return, `46.7%` win rate, `1.56` profit factor, `20.46%` drawdown.
   - Orderbook crash-exit block:
     - `247` trades, `301.98%` return, `42.9%` win rate, `1.59` profit factor, `15.69%` drawdown.
7. Result in plain English:
   - The orderbook crash exit did what the user suspected in one important way:
     - it helped reduce large drawdown.
   - It was not free:
     - it exited enough trades early that total return and win rate fell.
   - It is better than the earlier orderbook entry filter and position-size guard, but still too blunt.
8. Verdict:
   - Worth rework.
   - Not final.
9. Next step:
   - Split by rule family:
     - keep crash exit where it cuts losers,
     - remove it where it cuts profitable continuation,
     - test a stricter threshold for continuation-long families,
     - test a looser threshold for weak reclaim/LVN families.

# 2026-06-06 - Selective orderbook crash exit: promoted research variant

1. Trader question:
   - Does orderbook crash/downturn exit logic work better if applied only to entry families where it actually helped?
2. What was looked at:
   - Freqtrade baseline quality-family result.
   - Freqtrade broad orderbook crash-exit result.
   - Rule-family profit deltas between the two.
3. Family audit:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_orderbook_crash_exit_family_audit_20260606.csv`
   - Broad crash exit helped:
     - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`
     - `sieve1_tlv2_resistance_breakout_long_4h`
     - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h`
     - `sieve1_vp_lvn_fast_traverse_long_1h`
   - Broad crash exit hurt:
     - `sieve2_multi2_vp_prior_month_high_break_vp_node_long`
     - `sieve1_bos_bull_continuation_long_8h`
     - `sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard`
     - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_1d`
4. Freqtrade strategy:
   - `TraderRuleBlockBtcSieveQualitySelectiveOrderbookCrashExitStrategy`
5. Freqtrade output:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_20-56-44.zip`
6. Result:
   - Baseline quality-family:
     - `227` trades, `371.08%` return, `46.7%` win rate, `1.56` profit factor, `20.46%` drawdown.
   - Broad orderbook crash exit:
     - `247` trades, `301.98%` return, `42.9%` win rate, `1.59` profit factor, `15.69%` drawdown.
   - Selective orderbook crash exit:
     - `233` trades, `426.75%` return, `46.8%` win rate, `1.75` profit factor, `14.88%` drawdown.
7. Result in plain English:
   - The user’s idea was directionally right:
     - orderbook can help as a crash/downturn escalation detector.
   - It becomes useful when it is tied to the correct entry story.
   - It should not be slapped onto every long trade.
8. Verdict:
   - Promote as the current best BTC lead-block research variant.
   - Still not production-alpha.
9. Next step:
   - Apply the same family-specific method to:
     - volume failure,
     - time-to-confirm,
     - volatility shock,
     - structure failure.
   - Then combine only validated improvements into a candidate production-alpha research strategy.

# 2026-06-06 - Selective time-to-confirm exit refinement

1. Trader question:
   - After a good entry fires, can we reduce damage by exiting only selected rule families when the trade has not started working after about six hours?
2. What was looked at:
   - Same BTC quality-family signal file.
   - Same selected per-rule exits.
   - Frozen 1h confluence cache.
   - The previous best selective orderbook crash-exit logic.
   - Additional family-scoped early exits:
     - volume failure,
     - structure failure,
     - volatility shock,
     - time-to-confirm,
     - selected combinations.
3. Report artifact:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_selective_trade_management_freqtrade_comparison_20260606.csv`
4. Freqtrade comparison:
   - Baseline quality family:
     - `227` trades, `371.081%` return, `20.465%` drawdown, `1.5631` profit factor.
   - Selective orderbook crash:
     - `233` trades, `426.745%` return, `14.878%` drawdown, `1.7514` profit factor.
   - Selective orderbook plus all tested management exits:
     - `244` trades, `297.486%` return, `16.028%` drawdown, `1.7630` profit factor.
   - Selective orderbook plus volume:
     - `238` trades, `391.576%` return, `14.681%` drawdown, `1.7967` profit factor.
   - Selective orderbook plus structure:
     - `239` trades, `383.836%` return, `13.718%` drawdown, `1.8244` profit factor.
   - Selective orderbook plus volatility:
     - `236` trades, `320.641%` return, `14.190%` drawdown, `1.6582` profit factor.
   - Selective orderbook plus time-to-confirm:
     - `233` trades, `447.648%` return, `14.471%` drawdown, `1.7959` profit factor.
   - Selective orderbook plus time and structure:
     - `240` trades, `388.705%` return, `13.801%` drawdown, `1.8379` profit factor.
   - Selective orderbook plus time and volume:
     - `238` trades, `390.039%` return, `15.803%` drawdown, `1.8003` profit factor.
5. Result in plain English:
   - The time-to-confirm exit improved the best existing system.
   - The market story is:
     - this entry should begin showing signs of life within several hours,
     - if it does not, and confirmation is absent, the setup is probably stale,
     - leaving early protects the block without reducing trade count.
   - Structure-only and volume-only exits reduced drawdown, but gave back too much return relative to the time-to-confirm variant.
   - The broad all-exits stack was worse, so more layers are not automatically better.
6. Verdict:
   - Promote `TraderRuleBlockBtcSieveQualitySelectiveObTimeExitStrategy` as current best research variant.
   - Park structure, volume, volatility, and broad combined exits for rework.
7. Next step:
   - Use the same rule-family-specific discipline to mine rare complete-pattern/sunflower-style entries.
   - Focus next on adding additional high-win-rate rare lead families, then test whether the selective orderbook/time exit framework transfers.

# 2026-06-06 - Complete-pattern rare-entry source found, sunflower still unresolved

1. Trader question:
   - Are there rare high-win-rate sieve entries that should be included in the eventual final system?
2. What was found:
   - Filename/content search did not locate a literal `sunflower` artifact.
   - A complete-folder report was found:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\launcher_v2\runtime\entry_sieve\reports\sieve3_complete_folder_metrics_20260605.json`
3. Relevant summary:
   - `sieve3_candidates`:
     - `60` strategy files,
     - `79,645` total trades,
     - `50.56%` weighted average win rate.
   - `sieve2_complete_patterns`:
     - `22` strategy files,
     - `2,278` total trades,
     - `55.27%` weighted average win rate,
     - `56.06%` unweighted row average win rate.
4. Example complete-pattern candidates:
   - `sieve2_geometry_triangle_squeeze_breakout_long_1h`
   - `sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h`
   - `sieve2_multi2_tlv2_boschoch_fall_res_ride_bos_bear_short_1h`
   - `sieve2_geometry_ascending_channel_lower_bounce_long_8h`
   - `sieve2_geometry_wedge_breakout_long_1h`
   - `sieve2_geometry_wedge_breakdown_short_4h`
5. Result in plain English:
   - These look closer to the user's requested rare, high-win-rate entry families than the broad sieve3 batch.
   - They are not automatically production candidates because folder-level metrics are not enough.
6. Verdict:
   - Promote the complete-pattern folder into the next lead-mining queue.
   - Keep `complete sunflower` unresolved until a literal artifact or user pointer is found.
7. Candidate extraction:
   - Created:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_complete_pattern_candidates_20260606.csv`
   - Highest-ranked `sieve2_complete_patterns` candidates from the folder-level summary:
     - `sieve2_geometry_wedge_breakout_long_1h`
       - `76` trades, `63.16%` weighted win rate, best `3/2` TP/SL row.
     - `sieve2_geometry_ascending_channel_lower_bounce_long_8h`
       - `89` trades, `62.92%` weighted win rate, best `4/2` TP/SL row.
     - `sieve2_multi2_tlv2_boschoch_fall_res_ride_bos_bear_short_1h`
       - `102` trades, `62.75%` weighted win rate, best `4/2` TP/SL row.
     - `sieve2_multi2_tlv2_vp_res_break_vp_node_long_4h`
       - `72` trades, `62.50%` weighted win rate, best `4/2` TP/SL row.
     - `sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h`
       - `111` trades, `61.26%` weighted win rate, best `4/2` TP/SL row.
8. Next step:
   - Feed these complete-pattern candidates through the same lead pipeline used for the current BTC quality-family block:
     - signal export,
     - confluence check,
     - tailored exit research,
     - Freqtrade validation,
     - family-specific trade-management audit.

# 2026-06-06 - Rare complete-pattern entries validated and merged

1. Trader question:
   - Do rare high-win-rate complete-pattern sieve entries add useful BTC trades, and can they be included in the eventual multi-scenario rule block?
2. Candidate source:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_complete_pattern_candidates_20260606.csv`
   - Matched BTC backtest archives from:
     - `C:\FreqTradeStuff\user_data\Custom_Launcher\launcher_v2\runtime\entry_sieve\results\20260521T012740_entry_all_resume.jsonl`
3. Candidate lead output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_complete_pattern_candidate_leads_20260606.csv`
4. Exported signal output:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_signal_exports_20260606_complete_pattern_rare_entries_signals.parquet`
   - `4` candidate families.
   - `29` signal rows.
5. Rare-entry families exported:
   - `sieve2_geometry_descending_channel_lower_breakdown_short_1h`
   - `sieve2_multi2_tlv2_boschoch_res_break_bos_bull_long_8h`
   - `sieve2_geometry_wedge_breakout_long_1h`
   - `sieve2_geometry_triangle_squeeze_breakout_long_1h`
6. Confluence check:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_confluence_checks_20260606_complete_pattern_rare_entries_confluence.csv`
   - Base rare entries:
     - `29` trades, `72.41%` win rate, `1.86%` average return, `4.16` profit factor.
   - `compression_plus_range_break`:
     - `24` trades, `83.33%` win rate, `2.48%` average return, `7.97` profit factor.
   - Interpretation:
     - Compression plus range-break context appears to improve the rare entries, but the sample is still small.
7. Tailored exit research:
   - Output:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_complete_pattern_rare_entries_exit_selected.csv`
   - Direct harness combined result:
     - `27` trades, `88.89%` win rate, `182.13%` compounded return, `1.92%` drawdown, `24.32` profit factor.
8. Freqtrade standalone rare-entry validation:
   - Strategy:
     - `TraderRuleBlockBtcSieveCompletePatternRareEntriesStrategy`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_21-22-24.zip`
   - Result:
     - `27` trades, `73.20%` return, `59.3%` win rate, `4.54` profit factor, `6.30%` drawdown.
9. Merged quality plus rare-entry validation:
   - Strategy:
     - `TraderRuleBlockBtcSieveQualityPlusCompletePatternRareEntriesStrategy`
   - Inputs:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries.parquet`
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_exit_research_20260606_btc_quality_plus_complete_pattern_rare_entries_selected.csv`
   - Output:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_21-23-22.zip`
   - Result:
     - `248` trades, `555.26%` return, `45.2%` win rate, `1.80` profit factor, `17.45%` drawdown.
10. Comparison report:
    - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_complete_pattern_merge_freqtrade_comparison_20260606.csv`
11. Result in plain English:
    - The rare complete-pattern entries are useful by themselves.
    - Adding them to the current best quality block increases return materially.
    - The merged block also worsens drawdown, so it is a high-return research candidate, not a final promoted system.
12. Verdict:
    - Promote rare complete-pattern entries as a validated lead family.
    - Keep the quality-only selective orderbook plus time-to-confirm variant as the cleaner current best drawdown-managed block.
    - Rework the merged block with selective rare-entry confluence filters or drawdown controls before production-alpha promotion.

# 2026-06-06 - Rare complete-pattern confluence filtering and rare-priority merge follow-up

1. Trader question:
   - If rare complete-pattern entries are promising, do they become cleaner when the surrounding market context also looks like compression/range-break behaviour?
2. Issue found in merge mechanics:
   - `TraderRuleBlockResearchStrategy._load_signals()` keeps one signal per timestamp.
   - A simple concat can hide a rare signal if a quality-family signal occurs in the same hour.
   - New merged rare-priority files were created by removing same-hour quality rows before adding rare rows.
3. Filtered rare-entry files:
   - compression plus range-break rare-only:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_complete_pattern_rare_entries_compression_range.parquet`
   - range-break rare-only:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_complete_pattern_rare_entries_range_break.parquet`
   - compression-release rare-only:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_complete_pattern_rare_entries_compression_release.parquet`
4. Rare-priority merged files:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_compression_range_rare_priority.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_range_break_rare_priority.parquet`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260606_btc_quality_plus_complete_pattern_rare_entries_all_rare_priority_rare_priority.parquet`
5. Freqtrade validations:
   - `TraderRuleBlockBtcSieveCompletePatternRareCompressionRangeStrategy`
     - `22` trades, `77.66%` return, `68.2%` win rate, `7.05` profit factor, `4.08%` drawdown.
     - Plain English:
       - These are rare setup trades. When they also occur with compression/range-break context, the trade quality improves.
   - `TraderRuleBlockBtcSieveQualityPlusRareCompressionRangePriorityStrategy`
     - `244` trades, `589.17%` return, `45.5%` win rate, `1.86` profit factor, `18.26%` drawdown.
     - Plain English:
       - Highest return in this follow-up, but drawdown is worse than the current clean best.
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy`
     - `246` trades, `588.42%` return, `45.5%` win rate, `1.84` profit factor, `16.83%` drawdown.
     - Plain English:
       - Similar return to compression/range priority with lower drawdown, but still not as clean as the current best drawdown-managed quality-only block.
   - `TraderRuleBlockBtcSieveQualityPlusRareAllPriorityStrategy`
     - `248` trades, `571.38%` return, `45.2%` win rate, `1.81` profit factor, `16.83%` drawdown.
     - Plain English:
       - Better than the unfiltered duplicate-prone merge, but weaker than the filtered rare-priority variants.
6. Comparison artifact:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rare_priority_filter_merge_comparison_20260606.csv`
7. Verdict:
   - The rare complete-pattern compression/range subset is a strong small lead family.
   - The best full merge is not yet clean enough because drawdown remains above the current accepted quality-only best.
   - Next tests should focus on rare-family-specific exits, especially keeping winners alive while cutting failed rare breakouts/reversals earlier.

# 2026-06-06 - Rare-specific exit and cleanup filter follow-up

1. Trader question:
   - Can rare complete-pattern entries be included in the main BTC rule block if they get dedicated exit logic?
2. Code additions:
   - Added rare-specific exit strategy wrappers in:
     - `C:\FreqTradeStuff\user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
   - Compiled successfully with:
     - `.venv\Scripts\python.exe -m py_compile user_data\strategies\TraderRuleBlockBtcSieveBroadFilterStrategies.py`
3. Rare-specific exit results:
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityStrategy`
     - `246` trades, `588.42%` return, `16.83%` drawdown, `1.84` profit factor.
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareTimeExitStrategy`
     - `247` trades, `558.02%` return, `20.09%` drawdown, `1.82` profit factor.
     - Interpretation:
       - Rare time-to-confirm is not useful in this form.
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareStructureExitStrategy`
     - `250` trades, `499.71%` return, `12.03%` drawdown, `1.96` profit factor.
     - Interpretation:
       - Rare structure-failure exit is useful because it cuts drawdown while preserving most of the return advantage.
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakPriorityRareTimeStructureExitStrategy`
     - `251` trades, `465.73%` return, `16.38%` drawdown, `1.91` profit factor.
     - Interpretation:
       - Combining rare time and structure exits is worse than structure alone.
4. Reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_rare_specific_exit_comparison_20260606.csv`
5. Family audit of the rare-structure candidate:
   - Positive across 2020, 2021, 2022, 2023, 2024, and 2025 in the tested window.
   - Weak entry family:
     - `sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h`
     - `48` trades, `-2.33%` contribution, `0.98` profit factor.
   - Sunday was negative, but this needed validation because a weekday filter can overfit.
6. Audit reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_family_audit_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_year_audit_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_day_audit_20260606.csv`
7. Cleanup filter tests:
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoSundayRareStructureExitStrategy`
     - `214` trades, `337.35%` return, `13.47%` drawdown, `1.87` profit factor.
     - Interpretation:
       - Sunday filter is too blunt and removes too much good return.
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoWeakFamilyRareStructureExitStrategy`
     - `210` trades, `460.18%` return, `8.12%` drawdown, `2.14` profit factor.
     - Interpretation:
       - Removing the weak family gives the best risk-adjusted result so far.
   - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoSundayNoWeakFamilyRareStructureExitStrategy`
     - `181` trades, `330.66%` return, `7.82%` drawdown, `2.12` profit factor.
     - Interpretation:
       - Slightly lower drawdown than no-weak-family alone, but gives up too much return.
8. Cleanup comparison:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_best_candidate_cleanup_filter_comparison_20260606.csv`
9. Verdict:
   - Current best production-alpha research candidate:
     - `TraderRuleBlockBtcSieveQualityPlusRareRangeBreakNoWeakFamilyRareStructureExitStrategy`
   - It improves materially on the previous clean best:
     - previous clean best:
       - `233` trades, `447.65%` return, `14.47%` drawdown, `1.80` profit factor.
     - new best:
       - `210` trades, `460.18%` return, `8.12%` drawdown, `2.14` profit factor.
   - Keep:
     - rare complete-pattern entries,
     - range-break rare priority,
     - rare structure-failure exit,
     - weak-family removal.
   - Park:
     - Sunday filter,
     - rare time-to-confirm exit in this form.

# 2026-06-06 - Expansion pack 1 and merged current-best follow-up

1. Trader question:
   - Can the current best BTC block be expanded with more independent sieve-derived lead families without losing control of drawdown?
2. Process:
   - Built an expansion candidate pack excluding already-used current-best families and known weak families.
   - Exported decision-hour signal parquet for selected candidates.
   - Ran confluence checks and per-rule confluence selection.
   - Ran tailored exit research for the selected expansion signals.
   - Validated with real Freqtrade backtests, then merged the positive expansion families with the current best block.
3. Expansion pack result:
   - Raw best-per-rule block failed:
     - `177` trades, `-12.17%` return, `85.47%` drawdown, `0.94` profit factor.
   - Failure cause:
     - `sieve2_multi2_vp_prior_month_high_break_vp_bullctx_long` caused catastrophic historical damage.
   - Removing only the catastrophic family:
     - `226` trades, `1140.33%` return, `7.26%` drawdown, `2.38` profit factor.
   - Keeping only positive families:
     - `201` trades, `888.11%` return, `4.90%` drawdown, `2.57` profit factor.
4. Merged candidate result:
   - Current best before merge:
     - `210` trades, `460.18%` return, `8.12%` drawdown, `2.14` profit factor.
   - Current-priority merge:
     - `324` trades, `1693.90%` return, `11.70%` drawdown, `1.93` profit factor.
   - Expansion-priority merge:
     - `324` trades, `1767.20%` return, `11.71%` drawdown, `1.96` profit factor.
5. Plain-English result:
   - The expansion pack adds real useful entries.
   - The best merged block trades more often and makes much more in the backtest, but it gives back more during the worst drawdown than the cleaner smaller block.
   - This is a strong research candidate, not final production alpha.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_btc_expansion_pack1_freqtrade_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_current_best_plus_expansion_pack1_freqtrade_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-21-12.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-21-30.zip`
7. Complete/sunflower status:
   - Complete-pattern rare entries are already represented in the current best family.
   - Focused searches still found no literal `sunflower` artifact.
   - Keep sunflower unresolved until the exact artifact/rule name is identified.
8. Next action:
   - Run drawdown cleanup on the merged expansion-priority block.
   - Prioritise family-specific exits, crash/downturn invalidation, and weak-family audits.
   - Continue independent lead-pack mining toward the `50-100` lead objective.

# 2026-06-06 - Merged expansion weak-family cleanup

1. Trader question:
   - The merged expansion-priority block is high-return but has `11.71%` drawdown. Can we clean obvious weak families without overfitting or throwing away the edge?
2. Audit:
   - Family audit generated from:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-21-30.zip`
   - Negative families:
     - `sieve1_bos_bull_continuation_long_8h`
     - `sieve1_tlv2_resistance_breakout_long_4h`
     - `sieve2_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard`
3. Tests:
   - baseline merged expansion-priority:
     - `324` trades, `1767.20%` return, `11.71%` drawdown, `1.96` profit factor.
   - no worst family:
     - `317` trades, `1942.81%` return, `6.04%` drawdown, `2.03` profit factor.
   - no two weak families:
     - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor.
   - no three weak families:
     - `304` trades, `1753.60%` return, `6.05%` drawdown, `2.08` profit factor.
4. Plain-English result:
   - Removing one or two weak families made the system both more profitable and less painful during its worst slump.
   - Removing the third family was too much cleanup: it did not reduce drawdown but reduced return.
5. Verdict:
   - Promote `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy` as the current best BTC production-alpha research candidate.
   - Keep `no_worst_family` as a near-equivalent fallback.
   - Reject `no_three_weak_families` as over-cleaned.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_merged_expansion_priority_family_audit_20260606_current_best_plus_expansion_pack1_expansion_priority.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_current_best_plus_expansion_pack1_cleanup_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-25-35.zip`
7. Next action:
   - Audit remaining losses and drawdown period in `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy`.
   - Try targeted invalidation exits instead of broad additional filters.

# 2026-06-06 - Current-best trade-management follow-up

1. Trader question:
   - Can the current best merged BTC block be improved with existing selective trade-management wrappers or a narrow short-side invalidation layer?
2. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy`
   - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor, `55.7%` win rate.
3. Existing wrapper tests:
   - full orderbook/structure/volume/time management:
     - `315` trades, `1473.77%` return, `6.04%` drawdown, `2.04` profit factor.
   - orderbook plus time-to-confirm:
     - `311` trades, `1811.93%` return, `6.03%` drawdown, `2.04` profit factor.
   - orderbook plus structure:
     - `314` trades, `1784.91%` return, `6.04%` drawdown, `2.06` profit factor.
4. Targeted short-structure invalidation test:
   - Added:
     - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesShortStructureExitStrategy`
   - Applied structure-failure exits only to:
     - `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`
     - `sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h`
     - `sieve2_reversal_double_top_present_short_1h`
   - Result:
     - `319` trades, `1338.65%` return, `10.83%` drawdown, `2.06` profit factor, `51.4%` win rate.
   - Evidence:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-33-32.zip`
5. Plain-English interpretation:
   - The exit fired, but it did not fire in a useful way.
   - It cut too much profitable short exposure and shifted the worst drawdown to another period.
   - The current best remains better without these additions.
6. Verdict:
   - Reject these trade-management variants for the current best.
   - Keep the code only as research wrappers unless later evidence says otherwise.
   - Continue exit research with more precise failure-mode tests rather than broad generic exits.
7. Comparison report:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_no_two_weak_families_trade_management_comparison_20260606.csv`
8. Complete/sunflower note:
   - Complete-pattern rare entries remain included in the validated research path.
   - Literal `complete sunflower` remains unresolved until an exact artifact/rule name is found.

# 2026-06-06 - Expansion pack 2 balanced lead mining and merge

1. Trader question:
   - Can another independent pack of unused sieve-derived BTC lead families improve the current best multi-scenario block?
2. Selection:
   - Source:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads.csv`
   - Selected:
     - `32` unused BTC candidate families.
   - Excluded:
     - current-best rules,
     - expansion-pack-1 rules,
     - known weak/catastrophic families.
   - Selection constraints:
     - at least `5` BTC trades,
     - profit factor at least `1.25`,
     - drawdown no worse than `15%`,
     - balanced behaviour groups.
3. Generated data:
   - Candidate CSV:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trader_sieve_runtime_candidate_leads_20260606_btc_expansion_pack2_balanced.csv`
   - Raw exported signal rows:
     - `489`.
   - Per-rule confluence selected signal rows:
     - `154`.
   - Tailored exit rule rows:
     - `18`.
4. Standalone validation:
   - Strategy:
     - `TraderRuleBlockBtcSieveExpansionPack2BalancedBestPerRuleStrategy`
   - Result:
     - `131` trades, `421.99%` return, `7.06%` drawdown, `2.58` profit factor, `55.0%` win rate.
   - Evidence:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-40-36.zip`
5. Merge validation:
   - Current best baseline:
     - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack1NoTwoWeakFamiliesStrategy`
     - `309` trades, `1947.68%` return, `6.05%` drawdown, `2.06` profit factor.
   - Current-priority merge:
     - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy`
     - `357` trades, `3228.90%` return, `7.04%` drawdown, `2.01` profit factor, `54.9%` win rate.
   - Pack2-priority merge:
     - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2Pack2PriorityStrategy`
     - `358` trades, `3091.94%` return, `7.03%` drawdown, `2.01` profit factor, `54.5%` win rate.
6. Plain-English result:
   - Pack 2 adds useful trades.
   - Current-priority conflict handling is better than letting pack 2 override the current best on same-hour conflicts.
   - The new merged candidate makes much more in the historical test, but gives back slightly more during its worst slump.
7. Verdict:
   - Promote `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy` as the new high-return research candidate.
   - Do not treat it as final production alpha until weak-family and drawdown cleanup are audited.
8. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_btc_expansion_pack2_freqtrade_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-42-30.zip`

# 2026-06-06 - Expansion pack 2 weak-family cleanup

1. Trader question:
   - After expansion pack 2 produced a high-return merged BTC candidate, can weak-family cleanup improve it without throwing away the useful new entries?
2. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestPlusExpansionPack2CurrentPriorityStrategy`
   - `357` trades, `3228.90%` return, `7.04%` drawdown, `2.01` profit factor, `54.9%` win rate.
3. Cleanup variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoWorstOverallFamilyStrategy`
     - `352` trades, `2968.85%` return, `7.04%` drawdown, `2.04` profit factor, `55.1%` win rate.
     - Interpretation: removing only the single worst family cut return and did not improve the bad slump.
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
     - `346` trades, `3236.58%` return, `6.82%` drawdown, `2.08` profit factor, `55.5%` win rate.
     - Interpretation: this is the useful cleanup. It removed enough weak behaviour to improve the whole block without cutting the high-return effect.
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoPack2WeakFamiliesStrategy`
     - `342` trades, `2957.66%` return, `6.82%` drawdown, `2.15` profit factor, `55.3%` win rate.
     - Interpretation: cleaner, but gives up too much return versus the top-3 cleanup.
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoDrawdownLossFamiliesStrategy`
     - `275` trades, `1513.50%` return, `12.56%` drawdown, `2.25` profit factor, `57.8%` win rate.
     - Interpretation: too blunt. It removed many useful entries, cut total return by more than half, and worsened drawdown.
4. Promoted current research candidate:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
5. Complete/sunflower status:
   - Rare complete-pattern entries are still represented through the complete-pattern signal artifacts and dedicated exit handling already merged into the current-best path.
   - No literal `sunflower` rule name was found in the searched strategy and guidance paths. Treat this as unresolved naming unless the user points to the exact source.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_current_best_plus_pack2_cleanup_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_22-50-09.zip`
7. Next step:
   - Continue independent lead-pack mining toward `50-100` leads.
   - Use exact family-level failure audits for exits and drawdown controls; do not broadly delete every family that appears in one drawdown window.

# 2026-06-06 - Current best production-promotion and failure-mode audit

1. Trader question:
   - Does the current best BTC rule block justify further production-alpha work versus simply holding BTC, and where does it still fail?
2. Candidate:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
3. Promotion-gate evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_promotion_gate_current_best_years_20260606.csv`
4. Key promotion-gate result:
   - Strategy return:
     - `3236.58%`
   - Same-window market change / buy-and-hold proxy from Freqtrade artifact:
     - `1247.17%`
   - Return ratio:
     - `2.60x`
   - Drawdown:
     - `6.82%`
   - Trades:
     - `346`, about `57.3` per year.
   - Long and short both contributed:
     - long return `1878.02%`,
     - short return `1358.56%`.
   - Full-year stability:
     - `6/6` full historical years positive from 2020 through 2025.
5. Promotion-gate verdict:
   - Research promotion is justified.
   - Final production-alpha is not complete because targeted exits, risk validation, broader lead count, refreshed source coverage, and dry-run-specific preparation remain incomplete.
6. Failure-mode evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_failure_mode_audit_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_worst_trades_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_worst_families_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_worst_months_20260606.csv`
7. Failure-mode result:
   - The worst cluster is `2025-12`:
     - `5` trades,
     - `-2044.96` absolute profit,
     - worst trade `-887.13`.
   - Worst trade families to target next:
     - `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`
     - `sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h`
     - `sieve1_ladder_long_sup_hold`
     - `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h`
     - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`
8. Next tests:
   - December-2025 replay with orderbook crash/downturn and volatility risk states.
   - VP resistance-break long acceptance/volume-failure exits.
   - Support-break short support-reclaim/structure-failure exits.
   - Ladder support-hold time-to-confirm and volatility-shock exits.

# 2026-06-06 - Targeted current-best exit validation

1. Trader question:
   - Can the current best BTC candidate reduce failure risk by using targeted exits only on the rule families that caused the worst losses?
2. What was tested:
   - short structure-failure exit,
   - long volume/time confirmation failure exit,
   - long orderbook crash/downturn escalation exit,
   - combined targeted failure stack.
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestPlusPack2NoTop3OverallWeakFamiliesStrategy`
   - `346` trades, `3236.58%` return, `6.82%` drawdown, `2.08` profit factor, `55.5%` win rate.
4. Results:
   - `TraderRuleBlockBtcSieveCurrentBestPack2ShortStructureFailureExitStrategy`
     - `356` trades, `2280.07%` return, `10.09%` drawdown, `2.12` profit factor.
     - Rejected: worse return and worse drawdown.
   - `TraderRuleBlockBtcSieveCurrentBestPack2LongVolumeFailureExitStrategy`
     - `363` trades, `2126.85%` return, `6.99%` drawdown, `2.08` profit factor.
     - Rejected: cut too much return and slightly worsened drawdown.
   - `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
     - Promoted as new current best research candidate.
   - `TraderRuleBlockBtcSieveCurrentBestPack2TargetedFailureExitStackStrategy`
     - `376` trades, `1594.13%` return, `9.35%` drawdown, `2.36` profit factor.
     - Rejected: over-managed and cut too much edge.
5. Plain-English result:
   - Orderbook finally showed clear value when used in the way the user suggested: as a crash/downturn escalation exit for vulnerable long trades.
   - The useful behaviour was not broad orderbook filtering. It was targeted invalidation/risk management for specific long families.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_targeted_exit_comparison_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_targeted_exit_comparison_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_targeted_exit_years_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_targeted_exit_reasons_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-04-43.zip`
7. Next step:
   - Treat `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy` as the current best research candidate.
   - Rework short-side failure exits more carefully; current structure-failure logic is too blunt.
   - Keep building independent lead families and later retest the promoted orderbook exit on the broader block.

# 2026-06-06 - Strict current-best exit rework

1. Trader question:
   - Can the current best BTC rule block reduce remaining failure risk with stricter, family-specific exits instead of broad generic exits?
2. What was tested:
   - strict short reclaim exit on vulnerable short support-break/rejection families,
   - long acceptance-failure exit on vulnerable long continuation/reclaim families,
   - combined strict short plus long acceptance-failure exit,
   - all compared against the pack-2 cleanup baseline and the promoted long orderbook crash exit.
3. Results:
   - Pack-2 cleanup baseline:
     - `346` trades, `3236.58%` return, `6.82%` drawdown, `2.08` profit factor.
   - Current best long orderbook crash exit:
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
   - Strict short reclaim plus orderbook exit:
     - `357` trades, `3685.89%` return, `5.30%` drawdown, `2.47` profit factor.
   - Long acceptance-failure plus orderbook exit:
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
   - Strict short and long acceptance-failure combined:
     - `357` trades, `3685.89%` return, `5.30%` drawdown, `2.47` profit factor.
4. Plain-English result:
   - The stricter short reclaim exit did what the broad short-structure exit failed to do: it lowered drawdown without wrecking the system.
   - It gave back a small amount of return, so it should be parked as a lower-drawdown branch rather than replacing the max-return current best immediately.
   - The long acceptance-failure trigger had no material effect; the tested conditions did not catch the real remaining long failures.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_strict_exit_rework_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_strict_exit_rework_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-16-12.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-16-43.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-17-14.zip`
6. Next step:
   - Keep `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy` as the current best return branch.
   - Keep `TraderRuleBlockBtcSieveCurrentBestObExitStrictShortReclaimStrategy` as the current best drawdown-control branch.
   - Rework long exits using actual worst-trade replay, not broad acceptance-failure thresholds.

# 2026-06-06 - Long failure replay and replay-expanded orderbook exits

1. Trader question:
   - For the remaining worst long losses in the current best, what did the market show after entry that could have justified an earlier exit?
2. Replay result:
   - `30` worst remaining long losses were replayed against the frozen trader confluence feature cache.
   - `27` had orderbook-stress conditions after entry.
   - `3` remained unclear.
3. Follow-up test:
   - Expanded the current best orderbook crash exit to the long families that appeared in the orderbook-stress replay group.
4. Results:
   - `TraderRuleBlockBtcSieveCurrentBestReplayExpandedOrderbookStressExitStrategy`
     - `374` trades, `3374.76%` return, `6.04%` drawdown, `2.52` profit factor.
     - Rejected: return fell by about `347` percentage points without improving drawdown.
   - `TraderRuleBlockBtcSieveCurrentBestReplayExpandedObStressStrictShortStrategy`
     - `375` trades, `3342.79%` return, `5.29%` drawdown, `2.55` profit factor.
     - Rejected versus strict short reclaim branch: nearly same drawdown but much lower return.
5. Plain-English interpretation:
   - Orderbook stress is a real warning sign, but it is not the same as trade invalidation.
   - Some profitable or recoverable long trades also show orderbook stress.
   - The next long-exit rule needs a stricter story: orderbook stress plus failed support/reclaim plus negative trade progress.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_failure_replay_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_failure_replay_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_replay_expanded_orderbook_exit_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_replay_expanded_orderbook_exit_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-22-48.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-23-19.zip`
7. Next step:
   - Build a long invalidation exit requiring orderbook stress plus failed support/reclaim and negative trade progress.

# 2026-06-06 - Tighter long orderbook-stress invalidation still rejected

1. Trader question:
   - If orderbook stress alone is too broad, does it become useful when combined with a losing trade and failed support/structure?
2. What was tested:
   - `TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestLongObStressInvalidationStrictShortStrategy`
3. Result:
   - Long invalidation only:
     - `350` trades, `3311.53%` return, `6.04%` drawdown, `2.15` profit factor.
   - Long invalidation plus strict short reclaim:
     - `351` trades, `3281.12%` return, `5.30%` drawdown, `2.16` profit factor.
4. Plain-English interpretation:
   - Even after requiring the trade to be losing and structure/support to fail, a shared long-side invalidation rule still cut too much edge.
   - The market story is not uniform across all long families.
   - Long failures need per-family replay and exits; a shared long exit is currently too blunt.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_ob_stress_invalidation_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_ob_stress_invalidation_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-26-58.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-27-29.zip`
6. Next step:
   - Rework long exits by individual family, starting with the families that still generate large losses in the current best.

# 2026-06-06 - Current-best long-family diagnostics

1. Trader question:
   - After shared long exits failed, which individual long entry families show a distinct failure story that could support a tailored exit or removal test?
2. Tool added:
   - `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\current_best_family_diagnostics.py`
3. Inputs:
   - Current-best backtest archive:
     - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-04-43.zip`
   - Frozen confluence cache:
     - `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
4. Output reports:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606_signals.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_long_family_diagnostics_20260606_trades.csv`
5. Main family-level findings:
   - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`:
     - `34` trades, `23` losses, `23.60%` total profit.
     - Loser-skewed exit clues: support removed, spread fragility.
   - `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h`:
     - `24` trades, `10` losses, `23.61%` total profit.
     - Loser-skewed exit clues: bid absorption, local breakdown, support removal.
   - `sieve2_multi2_vp_prior_month_high_break_vp_node_long`:
     - `18` trades, `8` losses, `10.74%` total profit.
     - Loser-skewed exit clues: bid absorption, local breakdown.
   - `sieve1_geometry_triangle_squeeze_breakout_long_1h`:
     - `16` trades, `8` losses, `29.31%` total profit.
     - Loser-skewed exit clues: support cleared, breakout failure.
   - `sieve2_bos_bull_continuation_long_1h`:
     - `9` trades, `5` losses, `-3.52%` total profit.
     - Candidate for removal or strict filter before adding a complex exit.
   - `sieve1_ladder_long_sup_hold`:
     - `10` trades, `5` losses, `9.31%` total profit.
     - Loser-skewed exit clues: failed breakdown/breakout structure risk, support cleared.
   - `sieve1_vp_lvn_fast_traverse_long_1h`:
     - `13` trades, `5` losses, `18.86%` total profit.
     - Loser-skewed exit clues: support removal, downside vacuum, spread fragility.
6. Interpretation:
   - The long side is not one failure mode.
   - Some families should get stricter exits, while at least one net-negative family should be tested as removal/strict-filter first.
   - The next strategy pass should implement family-specific changes instead of broad shared long exits.

# 2026-06-06 - Current-best family-specific tests

1. Trader question:
   - Does the current-best branch improve by removing the one net-negative BOS long family, or by adding bundled family-specific long exits based on the diagnostic clues?
2. Variants added:
   - `TraderRuleBlockBtcSieveCurrentBestNoBosBullContinuationLong1hStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestFamilySpecificLongExitV1Strategy`
3. Results:
   - Prior current best:
     - `TraderRuleBlockBtcSieveCurrentBestPack2LongOrderbookCrashExitStrategy`
     - `356` trades, `3721.77%` return, `6.04%` drawdown, `2.44` profit factor.
   - Remove only `sieve2_bos_bull_continuation_long_1h`:
     - `TraderRuleBlockBtcSieveCurrentBestNoBosBullContinuationLong1hStrategy`
     - `352` trades, `3995.34%` return, `6.04%` drawdown, `2.45` profit factor.
   - Bundled family-specific long exits:
     - `TraderRuleBlockBtcSieveCurrentBestFamilySpecificLongExitV1Strategy`
     - `359` trades, `3606.94%` return, `6.05%` drawdown, `2.39` profit factor.
4. Plain-English interpretation:
   - Removing the weak BOS long family worked.
   - Trying to rescue several families with a bundled exit did not work; it cut too much useful edge.
   - The new max-return candidate is the BOS-removal branch, while strict short reclaim remains the lower-drawdown branch.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_family_specific_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_family_specific_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-37-49.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-38-20.zip`
6. Next step:
   - Promote `TraderRuleBlockBtcSieveCurrentBestNoBosBullContinuationLong1hStrategy` as the max-return branch.
   - Test one family-specific exit at a time, starting with triangle breakout and LVN fast-traverse families, instead of bundling multiple exit ideas together.

# 2026-06-06 - Current-best single-family long exit tests

1. Trader question:
   - After removing the weak BOS long family, can any single long entry family get a tailored exit that improves the BTC rule block?
2. Variants added:
   - `TraderRuleBlockBtcSieveCurrentBestNoBosTriangleSupportClearedExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoBosLvnOrderbookFragilityExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoBosLadderStructureFailureExitStrategy`
3. Results:
   - No-BOS baseline:
     - `352` trades, `3995.34%` return, `6.04%` drawdown, `2.45` profit factor.
   - Triangle support-cleared exit:
     - `352` trades, `4162.88%` return, `6.04%` drawdown, `2.52` profit factor.
   - LVN orderbook-fragility exit:
     - `352` trades, `3995.34%` return, `6.04%` drawdown, `2.45` profit factor.
   - Ladder structure-failure exit:
     - `352` trades, `3995.34%` return, `6.04%` drawdown, `2.45` profit factor.
4. Plain-English interpretation:
   - Triangle breakout trades improved when support-cleared / breakout-failure evidence could exit them early.
   - LVN and ladder triggers did not materially fire or did not change outcomes in this implementation.
   - The current max-return branch is now the no-BOS plus triangle-exit branch.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_single_family_exit_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_single_family_exit_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-42-45.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-43-16.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-43-48.zip`
6. Next step:
   - Test whether the triangle branch combines with strict short reclaim.
   - Rework LVN and ladder only after inspecting why their triggers did not alter trades.

# 2026-06-06 - Combined triangle and strict-short exit test

1. Objective:
   - Check whether the no-BOS triangle support-cleared max-return branch can also use the strict short-reclaim drawdown-control exit.
2. Strategy added:
   - `TraderRuleBlockBtcSieveCurrentBestNoBosTriangleStrictShortReclaimStrategy`
3. Trader-readable logic:
   - Keep the long-side rule that exits triangle squeeze breakout longs when support clears or breakout failure appears.
   - Add the short-side rule that exits vulnerable shorts when bullish reclaim evidence says the short idea is invalidated.
4. Backtest:
   - Pair: `BTC/USDT:USDT`.
   - Timeframe: `1h`.
   - Timerange: `20200101-20260115`.
   - Max open trades: `1`.
5. Result:
   - `353` trades.
   - `4125.17%` return.
   - `5.30%` drawdown.
   - `2.55` profit factor.
   - `54.4%` win rate.
   - Long / short return: `2475.97% / 1649.20%`.
6. Comparison:
   - Triangle-only max-return branch:
     - `4162.88%` return, `6.04%` drawdown, `2.52` profit factor.
   - Strict short-reclaim drawdown branch:
     - `3685.90%` return, `5.30%` drawdown, `2.47` profit factor.
7. Decision:
   - Promote the combined branch as the current best balanced production-alpha research candidate.
   - Keep the triangle-only branch as the highest-return candidate.
   - Continue one-family-at-a-time exit tests and complete-pattern/sunflower artifact resolution.
8. Evidence:
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-49-19.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_combined_exit_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_best_combined_exit_tests_20260606.csv`
9. Promotion-gate refresh:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_years.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_worst_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_worst_families.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_best_balanced_triangle_strict_short_20260606_worst_months.csv`

# 2026-06-07 - Balanced family removal tests

1. Objective:
   - Use the balanced branch's loss-burden audit to test whether low-efficiency families should be removed rather than given more exits.
2. Strategies added:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakLongStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoOvertradeResBreakAndPocShortStrategy`
3. Results:
   - Balanced baseline:
     - `353` trades, `4125.17%` return, `5.30%` drawdown, `2.55` profit factor.
   - Remove overtrade resistance-break long:
     - `327` trades, `3965.75%` return, `5.30%` drawdown, `2.70` profit factor, `57.2%` win rate.
   - Remove POC reject short:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor, `56.0%` win rate.
   - Remove both:
     - `310` trades, `4528.92%` return, `5.30%` drawdown, `2.91` profit factor, `59.0%` win rate.
4. Decision:
   - Promote `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy` as the current strongest production-alpha research candidate.
   - Keep the both-removed variant as a cleaner alternative if production gates prefer profit factor/win rate over max return.
   - Do not remove the overtrade resistance-break long alone. It needs a smarter family-specific filter or exit if revisited.
5. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_balanced_family_removal_tests_20260606.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_balanced_family_removal_tests_20260606.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_reject_short_20260607_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-57-17.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-57-18.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-06_23-58-05.zip`

# 2026-06-07 - Short volume-pressure exits rejected

1. Trader question:
   - For short entries that lose when volume pressure turns against the short, can an early pressure-based exit reduce damage?
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortVolumePressureExitExpandedStrategy`
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
4. Results:
   - Top short-family pressure exit:
     - `339` trades, `3062.21%` return, `10.77%` drawdown, `2.51` profit factor.
   - Expanded short-family pressure exit:
     - `341` trades, `2906.33%` return, `10.78%` drawdown, `2.50` profit factor.
5. Plain-English interpretation:
   - The diagnostic clue was real enough to test, but the implementation was too blunt.
   - A short should not be exited just because pressure turns positive; the market must also show that the short thesis is being invalidated.
   - Next short-side tests should require reclaim/acceptance against the short, bid absorption, breakdown failure, or similar trader-state evidence.
6. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_volume_pressure_exit_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_volume_pressure_exit_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-03-07.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-03-30.zip`

# 2026-06-07 - Contextual short-failure exits rejected

1. Trader question:
   - Can the failed volume-pressure short exit be made useful by requiring more trader-like evidence: breakdown failure, bid absorption, reclaim pressure, or an already losing trade?
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureContextExitExpandedStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocShortFailureLossOnlyExitExpandedStrategy`
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor.
4. Results:
   - Top-family contextual exit:
     - `343` trades, `3794.52%` return, `10.70%` drawdown, `2.75` profit factor.
   - Expanded contextual exit:
     - `348` trades, `3097.61%` return, `9.72%` drawdown, `2.75` profit factor.
   - Top-family loss-only contextual exit:
     - `338` trades, `4142.23%` return, `9.48%` drawdown, `2.52` profit factor.
   - Expanded loss-only contextual exit:
     - `338` trades, `4021.28%` return, `8.76%` drawdown, `2.54` profit factor.
5. Plain-English interpretation:
   - Requiring more context helped less than expected.
   - These exits still cut or alter too many useful shorts.
   - The top-family loss-only version was the least bad, but it still gave up return and almost doubled drawdown versus baseline.
6. Decision:
   - Reject all four variants.
   - Keep the current strongest no-POC branch unchanged.
   - Do not continue broad short-failure exits without exact worst-short replay evidence.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_failure_context_exit_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_short_failure_context_exit_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-10-47.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-11-16.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-12-42.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-13-14.zip`

# 2026-06-07 - Current strongest long-family exits rejected

1. Trader question:
   - Can the current strongest no-POC branch be improved by adding narrow exits for long families where diagnostics found orderbook/structure failure clues?
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocLadderSupportFailureExitStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocLvnFragilityLossExitStrategy`
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `13.24%` max underwater, `2.73` profit factor, `56.0%` win rate.
4. Results:
   - Ladder support-failure exit:
     - `336` trades, `4513.27%` return, `5.30%` drawdown, `13.24%` max underwater, `2.64` profit factor, `55.7%` win rate.
   - LVN fragility loss exit:
     - `336` trades, `4507.71%` return, `5.30%` drawdown, `13.24%` max underwater, `2.64` profit factor, `55.7%` win rate.
5. Plain-English interpretation:
   - These exits did not reduce the system's meaningful risk.
   - They kept drawdown basically unchanged but reduced return and profit factor.
   - The diagnostic clues may still help in exact loser replay, but these broad family rules are too blunt for promotion.
6. Decision:
   - Reject both variants for now.
   - Keep the current strongest no-POC branch unchanged.
   - Next expansion should validate rare high-win-rate entry families such as `complete sunflower` independently before merging.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_long_family_exit_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_long_family_exit_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-18-10.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-20-09.zip`

# 2026-06-07 - Complete-pattern rare entries tested against current strongest branch

1. Trader question:
   - Should rare complete-pattern sieve entries be included in the final multi-entry BTC system?
2. Source artifacts:
   - `trader_sieve_runtime_signal_exports_20260606_complete_pattern_rare_entries_signals.parquet`
   - `trading_lead_exit_research_20260606_complete_pattern_rare_entries_exit_selected.csv`
3. Existing standalone rare evidence:
   - `TraderRuleBlockBtcSieveCompletePatternRareEntriesStrategy`:
     - `27` trades, `73.20%` return, `6.30%` drawdown, `4.54` profit factor, `59.3%` win rate.
   - Tailored exit research:
     - `27` trades, `88.89%` win rate, `182.13%` simple total return, `-1.92%` max drawdown, `24.315` profit factor.
4. New current-strongest merge variants:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareCurrentPriorityStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareRarePriorityStrategy`
5. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `13.24%` max underwater, `2.73` profit factor, `56.0%` win rate.
6. Results:
   - Current-best priority merge:
     - `339` trades, `4489.73%` return, `5.30%` drawdown, `13.24%` max underwater, `2.67` profit factor, `55.5%` win rate.
   - Rare-priority merge:
     - `339` trades, `4489.73%` return, `5.30%` drawdown, `13.24%` max underwater, `2.67` profit factor, `55.5%` win rate.
7. Plain-English interpretation:
   - The rare entries are not bad as a standalone idea.
   - But bolting them onto the current strongest branch made the total system weaker.
   - That usually means either the rare entries overlap awkwardly with current signals, their exits need to be different inside the merged block, or they should replace specific weak families rather than simply add more trades.
8. Decision:
   - Reject simple merge for now.
   - Keep rare complete-pattern entries as a promising standalone lead family.
   - Literal `sunflower` naming was not found in the searched strategy/report files; if it is a separate named artifact, it still needs locating and validation.
9. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-26-01.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-26-31.zip`

# 2026-06-07 - Filtered complete-pattern rare merges rejected

1. Trader question:
   - Do complete-pattern rare entries improve the current strongest branch if only the higher-quality filtered rare states are allowed?
2. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareCompressionRangeStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocPlusCompletePatternRareRangeBreakStrategy`
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `13.24%` max underwater, `2.73` profit factor, `56.0%` win rate.
4. Results:
   - Compression/range rare-priority merge:
     - `338` trades, `4351.32%` return, `5.30%` drawdown, `13.21%` max underwater, `2.63` profit factor, `55.6%` win rate.
   - Range-break rare-priority merge:
     - `337` trades, `4593.56%` return, `5.30%` drawdown, `13.21%` max underwater, `2.72` profit factor, `55.8%` win rate.
5. Plain-English interpretation:
   - Filtering helped the rare family as a standalone idea, but not when merged into the current strongest branch.
   - Range-break filtering was much less damaging than compression/range filtering, but still not better than baseline.
   - The current strongest branch is already dense enough that adding rare entries can displace or alter better existing trades under max-open-trades `1`.
6. Decision:
   - Reject both filtered simple merges for now.
   - Keep complete-pattern rare as a standalone/replacement lead family with dedicated exits.
   - Do not claim literal `sunflower` inclusion; focused search found no artifact with that name.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-30-45.zip`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-31-14.zip`

# 2026-06-07 - Rare range-break replacement branch rejected

1. Trader question:
   - Can the range-break subset of complete-pattern rare entries help if it replaces weaker current-best families instead of being bolted on?
2. Setup:
   - Removed `sieve2_reframed_vp_poc_reject_short_1h`.
   - Removed `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`.
   - Added range-break complete-pattern rare entries with rare priority.
3. Signal validation:
   - Snapshot: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260607_current_strongest_no_poc_no_overtrade_plus_rare_range_break_replacement.parquet`
   - Rows: `439`.
   - Duplicate timestamps: `0`.
   - Current-best replacement-source rows: `427`.
   - Rare replacement rows: `12`.
4. Baselines:
   - Current strongest no-POC branch:
     - `336` trades, `4700.59%` return, `5.30%` drawdown, `2.73` profit factor, `56.0%` win rate.
   - Cleaner no-overtrade/no-POC branch:
     - `310` trades, `4528.92%` return, `5.30%` drawdown, `2.91` profit factor, `59.0%` win rate.
5. Result:
   - `311` trades, `4198.00%` return, `5.30%` drawdown, `14.21%` max underwater, `2.80` profit factor, `58.5%` win rate.
6. Plain-English interpretation:
   - This branch raised win rate versus the current strongest branch but reduced total return too much.
   - It also did not beat the cleaner no-overtrade/no-POC branch on return, profit factor, or win rate.
   - The rare entries may still have value as a separate market-state block, but this replacement shape is not a promotion.
7. Decision:
   - Reject this specific replacement branch for now.
   - Keep complete-pattern rare entries on the roadmap as a separate lead family with dedicated exits and market-state gating.
8. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_plus_complete_pattern_rare_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-34-39.zip`

# 2026-06-07 - Complete-pattern rare overlap audit

1. Trader question:
   - If complete-pattern rare entries are good standalone, why do they not improve the current strongest combined strategy?
2. What was checked:
   - Rare signal timestamps from `trader_sieve_runtime_signal_exports_20260606_complete_pattern_rare_entries_signals.parquet`.
   - Current strongest trade windows from `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`.
   - Whether the current strongest branch was already entering/holding a trade at the rare entry hour.
3. Result:
   - Rare signals: `29`.
   - Rare signals where the current strongest branch was active at the next-hour entry point: `23`.
   - Rare standalone average return when current branch was active: `2.68%`, win rate `86.96%`.
   - Current active trade average return on those same contested rows: `3.22%`.
   - Rare was better than the active current trade in only `7` of `23` contested rows.
4. Plain-English interpretation:
   - The rare entries are not useless.
   - They are mostly competing with already-good current-best trades.
   - This explains why blind add-on, priority, and replacement tests did not improve the whole strategy.
5. Decision:
   - Keep complete-pattern rare entries as a separate lead family.
   - Do not force them into the main current-best block unless a stricter market-state gate identifies the minority of cases where rare beats the active current trade.
   - Final production-alpha design should allow dedicated exits per entry family, but also needs entry priority logic because BTC can only hold one same-pair position at a time in these tests.
6. Formal audit file:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_complete_pattern_rare_overlap_audit_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_complete_pattern_rare_overlap_audit_20260607.csv`
7. Additional finding from formal audit:
   - Non-contested rare signals were weak in this sample:
     - `6` non-contested rows,
     - `-1.29%` average standalone return,
     - `16.67%` win rate.
   - This argues against a simple "add rare only when free" gate.

# 2026-06-07 - Tiny weak-family removal from current strongest branch

1. Trader question:
   - Can the current strongest no-POC branch be improved by removing tiny long families that are individually weak, instead of adding more entries or broad exits?
2. Family audit source:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_family_audit_20260607.csv`
3. Baseline:
   - `TraderRuleBlockBtcSieveCurrentBestBalancedNoPocRejectShortStrategy`
   - `336` trades, `4700.59%` return, `5.30%` drawdown, `13.24%` max underwater, `2.73` profit factor, `56.0%` win rate.
4. Variants tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTripleBottomLongStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy`
5. Results:
   - Remove only `sieve1_triple_bottom_confirmed_long_1h`:
     - `334` trades, `4679.24%` return, `5.29%` drawdown, `13.24%` max underwater, `2.75` profit factor, `56.3%` win rate.
   - Remove three tiny negative long families:
     - `332` trades, `4764.00%` return, `5.30%` drawdown, `12.69%` max underwater, `2.75` profit factor, `56.6%` win rate.
   - Remove all net-negative absolute families including overtrade resistance-break long:
     - `306` trades, `4607.96%` return, `5.30%` drawdown, `13.73%` max underwater, `2.94` profit factor, `59.8%` win rate.
6. Plain-English interpretation:
   - The best improvement came from pruning a few small weak long families.
   - This helped more than forcing rare entries into the current strongest block.
   - The high-profit-factor alternative is cleaner but sacrifices return, so it should be kept as a comparison branch rather than the main branch.
7. Decision:
   - Promote `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` as the new strongest return branch for research.
   - Keep `TraderRuleBlockBtcSieveCurrentBestNoPocNoNegativeAbsFamiliesStrategy` as a cleaner high-win-rate/high-profit-factor alternative.
   - Reject the triple-bottom-only removal as a standalone improvement.
8. Complete/sunflower implication:
   - The final production-alpha system should be able to include rare complete-pattern/sunflower-style entries with dedicated exits.
   - The current evidence says those entries also need an entry-priority gate, because they often compete with already-good current-best trades.
   - Literal `sunflower` is still unresolved by artifact name; complete-pattern rare entries are the current concrete representative.
9. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_tiny_family_removal_tests_20260607.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_tiny_family_removal_tests_20260607.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-46-44.zip`

# 2026-06-07 - Promotion gate for new strongest branch

1. Trader question:
   - Does `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` deserve to become the current production-alpha research baseline?
2. Audit tooling:
   - Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_backtest_audit.py`.
   - The script parses Freqtrade backtest ZIPs and writes promotion summary, year stability, worst month, worst trade, and worst family reports.
3. Baseline comparison:
   - Audited branch:
     - `332` trades, `4764.00%` return, `5.3003%` drawdown, `2.7548` profit factor, `56.6%` win rate.
   - Prior strongest baseline:
     - `336` trades, `4700.59%` return, `5.2975%` drawdown, `2.7254` profit factor, `56.0%` win rate.
4. Plain-English interpretation:
   - The new branch is better on return and profit factor.
   - Drawdown is practically unchanged but fractionally higher in exact value, so do not claim drawdown improvement.
   - It beat same-window market change by `3.82x`.
   - All completed yearly buckets from 2020 through 2025 were positive; the January 2026 edge bucket is partial and excluded from the full-year gate.
5. Failure audit:
   - Worst month: December 2025, `-2217.96`.
   - Worst family: `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`, `-68.40` across `34` trades.
   - Worst trade: `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`, opened `2025-12-17 08:00:00+00:00`, `-1282.86`.
6. Decision:
   - Promote `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` as the current research baseline.
   - Keep failure-mode work open; next drawdown work should target the December 2025 short-loss cluster and filtered handling of the overtrade resistance-break long family.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_years.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_worst_families.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_worst_months.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_no_poc_no_tiny_negative_longs_20260607_worst_trades.csv`

# 2026-06-07 - December short-failure diagnostic and rejected exit

1. Trader question:
   - Can the current research baseline reduce its worst short-loss cluster by exiting shorts when the breakdown story starts failing?
2. Diagnostic tooling:
   - Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_failure_cluster_audit.py`.
   - It joins a backtest trade cluster to the frozen confluence snapshot and compares decision-hour features against same-direction winners/losers.
3. Diagnostic finding:
   - December 2025 short cluster:
     - `5` short trades.
     - `1` win, `4` losses.
     - `-2219.03` total profit in the diagnostic CSV scale.
   - The cluster had elevated failed-breakdown/reclaim-style evidence:
     - high `st_failed_breakdown_structure_risk`,
     - higher support-bounce evidence,
     - bearish pressure that did not cleanly resolve lower.
4. Variant tested:
   - `TraderRuleBlockBtcSieveCurrentBestNoTinyNegativeDecShortFailureExitStrategy`
   - Inherits from `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy`.
   - Targets only four short families and only after the trade is already losing with failed-breakdown/reclaim evidence.
5. Freqtrade result:
   - Candidate:
     - `332` trades, `4533.53%` return, `6.06%` drawdown, `2.73` profit factor, `55.7%` win rate.
   - Baseline:
     - `332` trades, `4764.00%` return, `5.30%` drawdown, `2.75` profit factor, `56.6%` win rate.
6. Plain-English interpretation:
   - The idea correctly identified a bad-looking cluster, but the exit was still too blunt for the whole strategy.
   - It reduced the worst December month but cut enough good behaviour elsewhere to lower return and increase drawdown.
7. Decision:
   - Reject this exit variant.
   - Keep the diagnostic evidence, but rework as exact per-family entry filtering or loser replay rather than another broad post-entry short exit.
8. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec2025_short_failure_cluster_20260607_failure_cluster.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec2025_short_failure_cluster_20260607_cluster_trades.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec2025_short_failure_cluster_20260607_feature_comparison.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec_short_failure_exit_20260607_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_current_strongest_dec_short_failure_exit_20260607_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_01-02-37.zip`

# 2026-06-07 - Signal stacking as position-management evidence

1. Trader question:
   - When a new signal appears while a BTC trade is already open, should it be ignored as a duplicate entry, or treated as add/hold/reduce evidence?
2. Instruction/guidance update:
   - `AGENTS.md`, `current_objectives.md`, `objective_examples.md`, and `concept_lifecycle_research_process.md` now say overlapping signals are position-management evidence:
     - same direction: possible add/hold/confidence,
     - opposite direction: possible reduce/tighten/exit,
     - not a hard rule; must be tested.
3. Direct audit tool:
   - Added `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\production_alpha_signal_stacking_audit.py`.
   - It joins the current strongest backtest, current-priority signal parquet, and confluence feature cache.
   - It reports overlap events, overlap trade groups, and lightweight management simulations.
4. Direct audit source:
   - Backtest: `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_00-46-44.zip`
   - Signal file: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_current_priority.parquet`
   - Feature cache: `C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet`
5. Direct audit result:
   - `332` baseline trades.
   - `515` signal rows reviewed.
   - `125` signal-overlap events occurred inside open trades.
   - Same-direction overlap:
     - `123` events,
     - parent-trade win rate `88.62%`,
     - average remaining trade-direction return `1.93%`.
   - Opposite-direction overlap:
     - only `2` events,
     - too sparse for a production exit rule.
6. Trade-level overlap result:
   - No overlap:
     - `238` trades,
     - `45.38%` win rate,
     - `0.5678%` average profit,
     - `1.67` profit factor.
   - Same-only overlap:
     - `92` trades,
     - `85.87%` win rate,
     - `2.9759%` average profit,
     - `14.47` profit factor.
7. Lightweight management simulation:
   - `add_0p25_after_first_same` and `add_0p50_after_first_same` looked better than baseline in the direct simulation.
   - These simulations are not a replacement for Freqtrade because Freqtrade wallet/stake accounting differs.
8. Freqtrade subclasses added:
   - `TraderRuleBlockBtcSieveCurrentBestSignalStackAddOnStrategy`
   - `TraderRuleBlockBtcSieveCurrentBestSignalStackReserveAddOnStrategy`
9. Freqtrade stake-accounting findings:
   - Invalid run: without `--pairs BTC/USDT:USDT`, the strategy traded extra configured pairs and cannot be compared to the BTC-only baseline.
   - Invalid run: without `--max-open-trades 1`, `unlimited` stake was divided across ten possible trade lanes, reducing stake size and making the result incomparable.
   - Full-stake add-on run with `--pairs BTC/USDT:USDT --max-open-trades 1` matched the baseline but placed `0` add orders because the original trade used almost all wallet capital.
10. Reserve/add Freqtrade result:
   - Strategy: `TraderRuleBlockBtcSieveCurrentBestSignalStackReserveAddOnStrategy`
   - Command shape:
     - `freqtrade backtesting --config user_data\configs\config_new2026.example.json --strategy TraderRuleBlockBtcSieveCurrentBestSignalStackReserveAddOnStrategy --timerange 20200101-20260115 --pairs BTC/USDT:USDT --max-open-trades 1 --export trades --breakdown year month --cache none`
   - Result:
     - `329` trades,
     - `2052.49%` return,
     - `4.24%` drawdown,
     - `2.60` profit factor,
     - `53.2%` win rate,
     - `1.65x` same-window BTC market-change ratio.
   - Baseline:
     - `332` trades,
     - `4764.00%` return,
     - `5.30%` drawdown,
     - `2.75` profit factor,
     - `56.6%` win rate.
11. Selective reserve/add follow-up:
   - Added `TraderRuleBlockBtcSieveCurrentBestSignalStackSelectiveReserveAddOnStrategy`.
   - It reserves stake only for families where same-direction overlap had enough examples and strong audit results:
     - `sieve1_bos_bull_continuation_long_1h`
     - `sieve1_continuation_flag_present_long_4h`
     - `sieve1_geometry_triangle_squeeze_breakout_long_1h`
     - `sieve1_multi2_tlv2_vp_sup_break_vp_node_short_8h`
     - `sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h`
     - `sieve2_multi2_tlv2_vp_res_reject_vp_vah_short_4h`
     - `sieve2_multi2_vp_prior_month_high_break_vp_node_long`
     - `sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard`
     - `sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard`
   - Result:
     - `330` trades,
     - `3279.01%` return,
     - `4.32%` drawdown,
     - `2.81` profit factor,
     - `54.5%` win rate,
     - `2.63x` same-window BTC market-change ratio.
   - Baseline still has higher return:
     - `4764.00%` return,
     - `5.30%` drawdown,
     - `2.75` profit factor,
     - `56.6%` win rate.
12. Stacked-subset result inside reserve/add run:
   - `93` trades received same-direction add orders.
   - Stacked subset:
     - `77.4%` win rate,
     - `2.41%` average profit ratio,
     - `8.86` profit factor,
     - `16520.81` profit_abs.
   - Non-stacked subset:
     - `236` trades,
     - `1.37` profit factor,
     - `4004.10` profit_abs.
13. Stacked-subset result inside selective reserve/add run:
   - `54` trades received same-direction add orders.
   - Stacked subset:
     - `79.6%` win rate,
     - `2.72%` average profit ratio,
     - `14.29` profit factor,
     - `13811.13` profit_abs.
   - Non-stacked subset:
     - `276` trades,
     - `2.11` profit factor,
     - `18979.00` profit_abs.
14. Sharp selective reserve/add run:
   - Added `TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy`.
   - It removes `sieve2_multi2_vp_prior_month_high_break_vp_node_long` from reserve/add eligibility after the first selective run showed that family was weaker as a reserve/add candidate.
   - Whole-run result:
     - `330` trades,
     - `3748.10%` return,
     - `4.32%` drawdown,
     - `2.87` profit factor,
     - `55.2%` win rate,
     - `3.01x` same-window BTC market-change ratio.
   - Stacked subset:
     - `47` add trades,
     - `83.0%` win rate,
     - `3.12%` average profit ratio,
     - `22.98` profit factor,
     - `15411.25` profit_abs.
   - Non-stacked subset:
     - `283` trades,
     - `50.5%` win rate,
     - `0.85%` average profit ratio,
     - `2.14` profit factor,
     - `22069.77` profit_abs.
   - Comparison to current strongest return baseline:
     - baseline return `4764.00%`,
     - baseline drawdown `5.30%`,
     - baseline profit factor `2.75`,
     - sharp selective return delta `-1015.89` percentage points,
     - drawdown improved by `0.98` percentage points,
     - profit factor improved by `0.11`.
15. Audit tooling correction:
   - Fixed `production_alpha_backtest_audit.py` verdict logic.
   - A candidate with lower return than an explicit baseline now says "Do not replace the current return baseline" even when drawdown, profit factor, and year stability are good.
   - Regenerated reserve, selective-reserve, and sharp-selective promotion reports with corrected wording.
16. Plain-English interpretation:
   - The later same-direction signal is genuinely useful evidence that an open trade is high quality.
   - Blanket reserve/add loses too much return because every non-stacked trade starts smaller.
   - Selective reserve/add is much better and improves drawdown/profit factor, but still gives up too much return to replace the max-return baseline.
   - Sharp selective reserve/add is better again and is the best position-management branch so far, but it still does not beat the max-return baseline.
   - This means the clue is useful for family-specific position sizing, not yet as a blanket "reserve stake on every trade" rule.
17. Decision:
   - Do not promote any add-on subclass as the current max-return baseline.
   - Keep `TraderRuleBlockBtcSieveCurrentBestNoPocNoTinyNegativeLongsStrategy` as the current strongest baseline.
   - Keep `TraderRuleBlockBtcSieveCurrentBestSignalStackSharpSelectiveReserveAddOnStrategy` as the best current risk-adjusted/position-management branch.
   - Promote same-direction signal overlap as a validated position-management lead.
18. Next work:
   - Refine family-specific reserve/add thresholds and test smaller rule sets, because one selected family still became the worst family in the selective run.
   - Preserve multiple same-hour signals in future signal parquet outputs, because current priority files keep only one signal per hour and hide true same-hour confluence.
   - Test opposite-direction overlap only after more examples exist or with broader close-conflict definitions.
19. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_current_strongest_signal_stacking.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_current_strongest_signal_stacking_signal_audit.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_current_strongest_signal_stacking_trade_audit.csv`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_reserve_addon_btc_freqtrade_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_09-52-42.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_selective_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_selective_reserve_addon_btc_freqtrade_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_09-56-54.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharp_selective_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharp_selective_reserve_addon_btc_freqtrade_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_10-01-03.zip`

# 2026-06-07 - Signal stacking branch refinement and multiple-path clarification

1. Trader question:
   - Should strategy development optimize only for highest return, or preserve several useful strategy branches with different risk/return/logical profiles?
2. Guidance update:
   - Added to `current_objectives.md` that agents should not collapse research into one "best" answer.
   - Preserve multiple viable paths, including max-return, balanced risk, lower-drawdown, sparse high-quality, and position-management variants.
3. Tested branch 1:
   - Strategy: `TraderRuleBlockBtcSieveCurrentBestSignalStackSharperReserveAddOnStrategy`
   - Trader logic:
     - Keep the weak overtrade resistance-break family as a normal entry.
     - Do not reserve stake or add to that family when another same-direction signal appears.
   - Result:
     - `330` trades,
     - `3867.94%` return,
     - `4.33%` drawdown,
     - `2.85` profit factor,
     - `55.5%` win rate.
   - Stacked subset:
     - `39` add trades,
     - `82.1%` win rate,
     - `2.93%` average profit ratio,
     - `23.36` profit factor.
4. Tested branch 2:
   - Strategy: `TraderRuleBlockBtcSieveCurrentBestSignalStackSharpNoOvertradeFamilyStrategy`
   - Trader logic:
     - Remove the weak overtrade resistance-break family completely instead of only blocking add behaviour.
   - Result:
     - `304` trades,
     - `3722.38%` return,
     - `4.32%` drawdown,
     - `3.06` profit factor,
     - `58.6%` win rate.
   - Stacked subset:
     - `40` add trades,
     - `82.5%` win rate,
     - `2.99%` average profit ratio,
     - `24.12` profit factor.
5. Plain-English interpretation:
   - The later same-direction add signal remains strong after refinement.
   - The sharper reserve branch is the better total-return stacking branch.
   - The no-overtrade-family branch is the cleaner quality/risk branch because it has higher win rate and profit factor with fewer trades.
   - Neither should be treated as "wrong" just because it does not beat the max-return branch.
6. Decision:
   - Keep both as viable strategy-development paths.
   - Use the sharper reserve branch when prioritizing return with improved drawdown versus the max-return baseline.
   - Use the no-overtrade-family branch when prioritizing cleaner logic, higher win rate, and higher profit factor.
7. Evidence:
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharper_reserve_addon_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharper_reserve_addon_btc_freqtrade_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_16-13-21.zip`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharp_no_overtrade_family_btc_freqtrade_promotion_gate.md`
   - `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\production_alpha_20260607_signal_stack_sharp_no_overtrade_family_btc_freqtrade_promotion_gate.csv`
   - `C:\FreqTradeStuff\user_data\backtest_results\backtest-result-2026-06-07_16-13-53.zip`
