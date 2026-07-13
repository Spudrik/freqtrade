# FreqAI Promotion Readiness Plan

This file defines the gate for promoting direct-test hypotheses into FreqAI queues. It is intentionally stricter than the older broad queue scripts. FreqAI is a validation and ranking step after a named trader-readable hypothesis survives data, coverage, direct-test, control, and ablation checks.

## 1. Promotion Principle

1. A FreqAI candidate must be a named trader hypothesis, not a broad source-family experiment.
2. The candidate must define the trader theory, direction, setup logic, trigger logic, score column, target label, required source-detail groups, and pass/fail threshold before queueing.
3. The candidate must be tested on a frozen 1h snapshot with timestamp-safe source flags and an attached validation report.
4. The candidate must beat price/structure baselines and non-trigger controls before FreqAI. Standalone AUC is not enough.
5. Broad all-feature modelling is blocked until compact named hypotheses repeatedly pass this gate.

## 2. Required Eligibility Checklist

1. Hypothesis registry:
   - `hypothesis_id` exists in `trader_confluence_hypotheses.py`.
   - Theory is trader-readable and states why the source families should matter together.
   - Direction is explicit: bullish, bearish, bidirectional, or reversion.
   - Setup, trigger, score, and target columns all exist in the frozen snapshot.
2. Column explainability:
   - A current column dictionary exists for the snapshot.
   - Every setup, trigger, score, and model feature column maps to `source_family` and `source_detail`.
   - Reports must name source-detail groups such as `structure_volume_profile`, `structure_tlv2_support_resistance`, `structure_bos_choch_market_structure`, `orderbook_spot`, `orderbook_bybit_linear`, `orderbook_bybit_inverse`, `context_gdelt_events`, `context_gkg_documents`, `context_article_source_activity`, `context_google_trends`, `context_btc_etf_flows`, and `context_global_market_macro`.
3. Snapshot validation:
   - Duplicate dates: `0`.
   - Source future violations: `0` for all required sources.
   - Inactive source signal rows: `0` for required context/orderbook confluence signals.
   - Infinite numeric cells: `0`.
   - Base timeline gaps are listed and the direct test/FreqAI scope does not assume candles are continuous through those gaps.
4. Data source eligibility:
   - Required source-detail groups have real nonzero coverage in the chosen window.
   - Missing source data remains missing. It is not forward-filled, backfilled through gaps, or converted into false quiet/balanced states.
   - Orderbook rows require source coverage above the selected minimum and `source_max_ts <= date`.
   - Context rows require `max_source_available_at <= date` and freshness within the selected max-age rule.
5. Direct-test eligibility:
   - Candidate has a direct-test report row for the exact target and selected threshold settings.
   - Candidate passes same-regime, random, shuffled-label, opposite-direction where applicable, and price/structure baseline checks.
   - Candidate has enough setup rows, trigger rows, positive target examples, and monthly windows for its event type.

## 3. Coverage And Clean-Window Checks

1. Full confluence windows:
   - Require price, structure, at least one valid orderbook venue, and all explicitly used context source-detail groups.
   - Use only rows where all required present flags are active after timestamp and coverage checks.
   - Do not describe the window as full confluence if one source-detail group is mostly absent.
2. Structure plus orderbook windows:
   - Require structure-present rows and the exact orderbook venue groups used by the hypothesis.
   - Multi-venue hypotheses require spot, linear, and inverse present flags where those venue features are model inputs.
3. Context windows:
   - Split context by source-detail group. Historical GDELT coverage is not equivalent to live news/web/global/Trends/ETF coverage.
   - A context hypothesis using live news/web/global/Trends/ETF features must prove those detail groups are active in the window.
4. Orderbook windows:
   - Do not carry book state through missing archive gaps.
   - Each archive/day starts from its own snapshot state.
   - Low-coverage venue rows are excluded from present flags and must not be silently treated as zero pressure.
5. Clean-window report requirement:
   - Each queued experiment needs a preflight record with date range, source-detail groups, active rows, nonzero rows, missing rows, low-coverage rows, future-violation rows, and max source age.
   - The report must state whether the experiment is full confluence, structure+orderbook, orderbook-only, GDELT-only, context-source-specific, or a missing-data/control experiment.

## 4. Direct-Test Metrics Required Before FreqAI

1. Binary target metrics:
   - `roc_auc`.
   - `average_precision`.
   - Trigger event rate.
   - Same-regime-without-trigger event rate.
   - Random eligible control event rate.
   - Shuffled-label AUC.
   - Price/structure baseline AUC.
   - Monthly AUC windows and positive monthly windows.
2. Continuous/path target metrics:
   - Oriented top-bucket lift.
   - Spearman correlation and p-value where applicable.
   - Same-regime and random-control target means.
   - Price/structure baseline lift.
   - Monthly or rolling-window stability if enough rows exist.
3. Minimum direct-test pass:
   - Binary AUC `>= 0.55`.
   - AUC beats shuffled-label AUC by at least `0.02`.
   - AUC beats price/structure baseline AUC by at least `0.01`; use `0.02` as the preferred promotion threshold.
   - Trigger event rate beats same-regime and random controls by at least `0.03` absolute for binary labels.
   - At least half of valid monthly windows have AUC `>= 0.55`, with a minimum of two positive windows.
   - For continuous/path labels, oriented lift must be material: at least `0.002` for return/drawdown style targets or at least `1.0` hour for time-to-threshold targets.
4. Threshold-sweep pass:
   - Threshold settings must preserve multi-component confluence. A pass that depends on one component only is not eligible.
   - Sweep-selected thresholds must be recorded in the queue spec and report.
   - A candidate with many passing threshold combinations is stronger than a one-off edge-case setting.

## 5. Minimum Rows And Sample Guidance

These are promotion minimums. A result below them can stay on the watchlist, but should not enter the normal FreqAI queue.

| Event type | Setup rows | Trigger rows | Positive labels | Stability windows | Promotion status |
|---|---:|---:|---:|---:|---|
| Common all-row return/path ranking | `3000+` | not required | target must vary across rows | `6+` monthly/rolling windows | normal if controls pass |
| Common breakout/breakdown success/failure | `1000+` | `150+` | `50+` positives in setup and `25+` positives in trigger | `6+` windows, `>= 50%` positive | normal if controls pass |
| Rare crash/drawdown risk | `2000+` | `75+` | `100+` positives in setup and `8+` positives in trigger | `6+` windows, `>= 50%` positive | pilot only unless trigger positives reach `30+` |
| LVN fast-travel/time-to-threshold | `500+` | `75+` | target must vary; enough reached/not-reached examples | `4+` windows | pilot until wider coverage |
| Multi-venue orderbook agreement | `1000+` all-required-venue rows | `150+` | `50+` positives in setup | `6+` windows | normal only if each venue has clean coverage |
| Source-detail-only probe | `1000+` rows for that detail group | optional | `50+` positives for binary target | `4+` windows | diagnostic, not standalone promotion |

1. Rare-event exceptions:
   - A rare crash-risk candidate may enter a pilot FreqAI queue with fewer trigger positives if setup rows, controls, and monthly stability are strong.
   - Pilot results may validate ranking behaviour but must not be treated as final strategy evidence.
2. Current evidence application:
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h` is a conditional pilot candidate only.
   - It must not be queued until a preflight artifact proves the exact thresholds, clean-window class, source-detail blocks, active/usable rows, positives, controls, baseline deltas, and timestamp-safety counts.
   - It has promising strict direct-test evidence, but trigger positives are still too small for a full promotion claim and source-detail coverage must be locked.
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_24h` is threshold-sweep promising but remains watchlist-only until the exact queued thresholds and source-detail coverage are locked.

## 6. Baseline And Ablation Requirements

1. Every FreqAI candidate queue must include a matched baseline run:
   - Price-only baseline.
   - Price plus structure baseline when the candidate uses structure.
   - Price plus structure plus required setup mask baseline for event-scoped hypotheses.
2. Every promoted feature set must include ablations:
   - Full compact hypothesis feature set.
   - Minus orderbook.
   - Minus context.
   - Minus structure/custom indicators.
   - Minus volume pressure.
   - Minus multi-timeframe confirmation.
   - Minus each orderbook venue where venue features are used.
3. Source-detail ablations are required when a broad family matters:
   - Structure detail: VP, TLV2 support/resistance, BOS/CHoCH market structure, pattern geometry, cached price/volume state.
   - Context detail: article/source activity, topic severity, GDELT events, GKG documents, Google Trends, BTC ETF flows, global/macro.
   - Orderbook detail: spot, Bybit linear, Bybit inverse, and orderbook composites.
4. A candidate passes the FreqAI validation step only if:
   - The full compact feature set beats its matched baseline on the selected target.
   - The edge is not entirely explained by raw price/structure.
   - Ablations identify which source-detail groups contribute.
   - Removing a claimed critical source-detail group causes measurable degradation, or the report downgrades that group to non-critical.

## 7. Queue Design For Source-Detail Feature Sets

1. Queue unit:
   - One queue item equals one hypothesis, one target, one threshold setting, one clean window class, one model family, and one feature-set variant.
2. Required queue metadata:
   - `hypothesis_id`.
   - Trader theory and direction.
   - Target label and horizon.
   - Setup/trigger/score columns.
   - Threshold settings from direct test or sweep.
   - Clean-window class and timerange.
   - Required source-detail groups.
   - Feature columns or feature-selection manifest.
   - Baseline/control profile id.
   - Direct-test artifact paths.
   - Snapshot path and validation artifact path.
3. Queue ordering:
   - Run matched baselines first.
   - Run the full compact candidate second.
   - Run family ablations third.
   - Run source-detail ablations fourth.
   - Run source-detail-only probes last as diagnostics.
4. Feature-set limits:
   - Prefer compact confluence, setup, trigger, source-summary, and lagged state features.
   - Avoid hundreds of raw columns unless an ablation explicitly needs a broad diagnostic pass.
   - Diagnostic timestamp, coverage, missingness, stale/source-age, source future violation, absolute level price, and future-label columns are blocked as model features.
5. Model choice:
   - Use simple baselines and compact tree models first.
   - Do not treat LightGBM improvement in one regime as final unless it is stable across clean windows and ablations.

## 8. Result Report Requirements

Each FreqAI report must explain the result in trader language and in exact column/source-detail language.

1. Required report header:
   - Hypothesis id and trader theory.
   - Target label.
   - Direction.
   - Timerange and clean-window class.
   - Snapshot and validation paths.
   - Queue id and model id.
2. Required feature explanation:
   - Exact setup column.
   - Exact trigger column.
   - Exact score column.
   - Exact setup component columns.
   - Exact trigger component columns.
   - Exact score component columns.
   - Source-detail groups used.
   - Model feature list or feature manifest path.
3. Required metric table:
   - Rows scored.
   - Event/setup rows.
   - Positive/negative labels.
   - AUC/AP/correlation/lift as applicable.
   - Baseline score.
   - Delta versus baseline.
   - Delta versus control profile.
   - Monthly or rolling-window stability.
4. Required ablation table:
   - Full compact candidate.
   - Each family ablation.
   - Each relevant source-detail ablation.
   - Verdict for which source-detail groups appear critical, redundant, or unproven.
5. Required verdict:
   - `promote`, `pilot_only`, `watchlist_refine`, or `blocked`.
   - The verdict must name the exact reason, not just "good AUC" or "bad model".

## 9. Explicit Blocks For Now

1. Block broad all-feature confluence FreqAI queues.
2. Block old loose-gate watchlists as promotion evidence. They can be cited only as exploratory history.
3. Block orderbook resistance-evaporation, support-removal, support-bounce, resistance-rejection, LVN travel, multi-venue orderbook, and context-plus-structure-break hypotheses from FreqAI until they pass the strict direct-test/sweep gate.
4. Block any context result that does not split source-detail coverage. Historical GDELT presence must not be presented as full news/web/global/Trends/ETF coverage.
5. Block source-detail-only feature sets from being treated as standalone promotion candidates. They are diagnostics unless attached to a passing named hypothesis.
6. Block rows with future-known source timestamps, stale context beyond the freshness rule, low orderbook coverage, missing required sources, or carried orderbook state through archive gaps.
7. Block diagnostic/quality columns as model features, including coverage, missing, stale, source age, source timestamp, available-at, future-violation, generated-at, and debug fields.
8. Block raw future labels, target labels, absolute level price columns, and direct OHLC columns as promoted model features unless explicitly used only for baseline construction outside the candidate feature set.
9. Block queue expansion across many models/windows before the baseline, ablation, and reporting format is proven on the pilot candidate.

## 10. Current Promotion Status

1. Conditional pilot candidate, pending preflight:
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_6h`.
   - Reason: strict direct test is promising: it passes trigger lift versus same-regime/random controls, beats shuffled labels, beats price/structure baseline, and has positive monthly stability.
   - Limitation: trigger positives are small and no attached preflight artifact yet proves exact thresholds, clean-window class, required source-detail blocks, active/usable rows, controls, baseline deltas, and timestamp-safety counts.
   - Current action: blocked from queue execution until that preflight exists.
2. Watchlist candidate:
   - `macro_context_liquidity_stress_breakdown -> large_drawdown_next_24h`.
   - Reason: threshold sweep found many passing settings, but the exact queued setting and source-detail coverage need to be locked before queueing.
3. Refine before FreqAI:
   - Orderbook breakout/breakdown, support/resistance bounce/rejection, LVN fast-travel, multi-venue agreement, quiet technical breakout/breakdown, failed breakout/exhaustion, and context-plus-structure-break hypotheses.
   - Required next step: threshold/component calibration that preserves multi-component trader logic while producing enough rows and beating price/structure baselines.
