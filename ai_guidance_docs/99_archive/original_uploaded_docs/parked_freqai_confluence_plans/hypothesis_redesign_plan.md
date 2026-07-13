# Hypothesis Redesign Plan

Status: design plan only. These packs describe how to redesign trader-readable hypotheses from the revised objectives. They do not claim the hypotheses, features, labels, reports, or tests are implemented.

## 1. Design Rules

1. Keep each pack small enough for direct tests before FreqAI.
2. Use source-detail blocks, not broad buckets such as `context` or `structure`.
3. Prefer trader-readable state and transition scores over hundreds of weak raw columns.
4. Require source coverage and timestamp safety before interpreting any result.
5. Treat missing source-detail coverage as a window-selection issue or explicit missing-data experiment, not as zero signal.
6. Report results in trader language: setup, trigger, target, sample size, lift, controls, ablations, stability, verdict.

## 2. Shared Source-Detail Blocks

1. Price and volume:
   - 1h OHLCV returns, range expansion, close location, wick behaviour, realized volatility, relative volume, buy/sell volume pressure where available.
2. VP structure:
   - POC, VAH, VAL, HVN/LVN proximity, value-area acceptance/rejection, next value/liquidity target distance.
3. TLV2 support/resistance:
   - active support/resistance lines, proximity, level age, retest count, break/reclaim status.
4. Market structure:
   - BOS, CHoCH, higher high/lower low transitions, range boundaries, compression/expansion state.
5. Pattern geometry:
   - rectangle/channel boundaries, breakout path, failed break return inside pattern, target distance.
6. Spot orderbook:
   - bid/ask wall persistence, wall evaporation, spread shock, pressure imbalance, post-break support/resistance rebuild.
7. Bybit linear orderbook:
   - same as spot, kept separate for source-detail agreement and venue ablations.
8. Bybit inverse orderbook:
   - same as spot, kept separate for agreement, divergence, and spoof/noise checks.
9. GDELT events:
   - event count/severity, topic persistence, source-country/category grouping, timestamp-safe availability.
10. GKG documents:
   - document volume, theme persistence, source confluence, first mention versus follow-through.
11. Live news/web:
   - timestamp-safe article/event attention, source confluence, topic persistence, known availability time.
12. Google Trends:
   - topic interest change, persistence, anomaly versus baseline, availability constraints.
13. Global/macro/ETF flows:
   - risk-on/risk-off pressure, rates/liquidity shock, ETF flow pressure, cross-market confirmation, release-time safety.

## 3. Shared Controls And Ablations

1. Core controls:
   - Random eligible rows inside the same clean source window.
   - Same regime without trigger.
   - Opposite-direction setup.
   - Shuffled labels.
   - Price/structure-only baseline.
   - Same trigger with future windows shifted outside the expected reaction period.
2. Broad ablations:
   - Full pack minus orderbook.
   - Full pack minus context/news/macro.
   - Full pack minus VP/TLV2/market-structure/pattern detail.
   - Full pack minus volume pressure.
   - Full pack minus multi-timeframe confirmation.
3. Source-detail ablations:
   - Spot orderbook only.
   - Bybit linear only.
   - Bybit inverse only.
   - All orderbook minus one venue at a time.
   - VP only, TLV2 only, market structure only, pattern geometry only.
   - GDELT events only, GKG only, live news/web only, Google Trends only, macro/ETF only.
   - Context excluding source blocks with known timestamp or coverage weakness.

## 4. Hypothesis Packs

### 4.1 Breakout Confirmation

1. Trader theory:
   - Upside continuation improves when price breaks a meaningful resistance level, accepts above it, volume expands, ask-side resistance retreats, and the next target is visible.
2. Required source-detail blocks:
   - Price/volume, VP, TLV2 support/resistance, market structure, pattern geometry, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook.
   - Optional: quiet or supportive GDELT/GKG/live news/macro blocks when timestamp-safe.
3. Candidate setup components:
   - Price within tolerance of multi-source resistance.
   - Resistance confirmed by at least two of VP VAH/HVN, TLV2 resistance, range/pattern top, recent swing high.
   - Compression or reduced realized volatility before the attempt.
   - Upside path has measurable room to next VP/orderbook/structure target.
   - Ask walls above price are weakening or not unusually persistent.
4. Candidate trigger components:
   - 1h close above resistance with range and relative-volume expansion.
   - BOS or higher-high confirmation.
   - Post-break retest holds above the broken level or close remains above level for N candles.
   - Ask-wall evaporation or bid support rebuilds below price on at least one venue, preferably two.
5. Candidate labels/targets:
   - `breakout_success_next_6h`.
   - `future_max_upside_6h` and `future_max_upside_24h`.
   - `time_to_next_upside_target`.
   - `hit_plus_xpct_before_minus_ypct`.
6. Controls:
   - Same resistance setup without close above.
   - Breakout trigger at low volume.
   - Random rows near resistance.
   - Opposite support-break setups.
7. Ablations:
   - Remove each resistance source: VP, TLV2, market structure, pattern geometry.
   - Remove each venue and test spot/linear/inverse agreement separately.
   - Remove acceptance condition.
   - Remove volume expansion.
   - Remove next-target distance.
8. Data-window requirements:
   - Full clean windows with OHLCV, structure, VP/TLV2/pattern, and at least one timestamp-safe orderbook venue.
   - Separate full-confluence window only when all three orderbook venues and context blocks overlap safely.
   - Do not carry orderbook wall state through raw archive gaps.
9. Expected failure modes:
   - Too few clean breakouts after requiring all confirmations.
   - Resistance definitions duplicate the same price action and overstate confluence.
   - Orderbook walls are stale, sparse, or venue-specific noise.
   - Target is too close or too far for the chosen horizon.

### 4.2 Breakdown/Crash Detection

1. Trader theory:
   - Downside continuation and crash risk rise when support breaks during volume expansion, bid support vanishes, sell-side pressure rises, and stress context is active.
2. Required source-detail blocks:
   - Price/volume, VP, TLV2 support/resistance, market structure, pattern geometry, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook, GDELT events, GKG documents, live news/web, macro/ETF/global where available.
3. Candidate setup components:
   - Price near multi-source support or range low.
   - Support confirmed by VP VAL/HVN, TLV2 support, pattern lower boundary, prior lows.
   - Downside path has thin VP/orderbook support or nearby liquidity vacuum.
   - Stress context has elevated severity, topic persistence, or cross-source attention.
   - Bid walls below price are weakening or repeatedly relocating lower.
4. Candidate trigger components:
   - 1h close below support with range expansion and relative volume.
   - Lower-low/BOS confirmation.
   - Bid-wall evaporation or sell pressure shock on at least one venue.
   - Stress context persists into the trigger window rather than appearing only after the move.
5. Candidate labels/targets:
   - `breakdown_success_next_6h`.
   - `large_drawdown_next_6h` and `large_drawdown_next_24h`.
   - `hit_minus_3pct_before_plus_2pct`.
   - `time_to_next_downside_target`.
6. Controls:
   - Same support setup without breakdown trigger.
   - Quiet-context support breaks.
   - Random support-proximity rows.
   - Opposite breakout setups.
7. Ablations:
   - Remove stress context.
   - Remove each context source-detail block: GDELT, GKG, live news/web, Trends, macro/ETF.
   - Remove each orderbook venue.
   - Remove VP/TLV2/pattern support details individually.
   - Remove volume and range expansion.
8. Data-window requirements:
   - Use full-confluence windows only where context sources are timestamp-safe and active before or at candle close.
   - Maintain separate structure plus orderbook windows when historical context coverage is incomplete.
   - Exclude periods where orderbook coverage gaps would create fake support removal.
9. Expected failure modes:
   - Context pressure is coincident but not causally useful.
   - Labels capture normal volatility rather than crash risk.
   - Sparse stress-event samples overfit.
   - Macro/news timestamps are future-known or daily data is assigned too early.

### 4.3 Failed Breakout/Rejection

1. Trader theory:
   - Breakout attempts fail when price probes resistance but cannot gain acceptance, volume fades, ask liquidity remains or rebuilds, and support below fails to form.
2. Required source-detail blocks:
   - Price/volume, VP, TLV2 support/resistance, market structure, pattern geometry, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook.
   - Optional: quiet context or adverse context blocks when available.
3. Candidate setup components:
   - Price touches or slightly exceeds resistance from at least two technical sources.
   - Prior move into the level is extended or volume momentum is fading.
   - Ask walls remain persistent above or reappear after the probe.
   - Bid support below price does not rebuild after the break attempt.
4. Candidate trigger components:
   - Close back below resistance or back inside the range/value area.
   - Upper wick rejection with weak close location.
   - Volume pressure divergence versus price high.
   - CHoCH/lower-high after the failed attempt.
5. Candidate labels/targets:
   - `breakout_failure_next_6h`.
   - `failed_breakout_next_24h`.
   - `fakeout_next_24h`.
   - `future_max_drawdown_6h`.
6. Controls:
   - Successful breakout cases with similar resistance proximity.
   - Resistance touches without break attempt.
   - Random rows near resistance.
   - Opposite failed-breakdown setups.
7. Ablations:
   - Remove acceptance failure.
   - Remove upper-wick/close-location component.
   - Remove volume divergence.
   - Remove each technical resistance source.
   - Remove each orderbook venue and the bid-rebuild component.
8. Data-window requirements:
   - Requires intraperiod high/close data and 1h sequencing sufficient to identify probe then return.
   - Requires orderbook snapshots before, during, and after the attempted break without gap carry.
   - Context is optional and must be tested in separate clean windows.
9. Expected failure modes:
   - Label confuses healthy retests with failures.
   - Resistance tolerance too wide creates noisy samples.
   - Wick-based triggers behave differently across volatile regimes.
   - Orderbook persistence metrics misread spoofing as real supply.

### 4.4 Support Bounce/Reclaim

1. Trader theory:
   - Bounce probability improves when price tests multi-source support, downside pressure fades, bid support rebuilds, and price reclaims support or value.
2. Required source-detail blocks:
   - Price/volume, VP, TLV2 support/resistance, market structure, pattern geometry, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook.
3. Candidate setup components:
   - Price near support from at least two of VP VAL/HVN/POC, TLV2 support, pattern lower boundary, prior swing low.
   - Downside move into support shows deceleration, lower wick, or reduced sell pressure.
   - Bid walls persist or rebuild near/below support.
   - Upside target to POC/VAH/next resistance is clear.
4. Candidate trigger components:
   - Close back above support, POC, or broken range boundary.
   - CHoCH to upside or higher-low after test.
   - Bid pressure shock or ask-side thinning after support test.
   - Volume confirms reclaim without immediate rejection.
5. Candidate labels/targets:
   - `breakdown_failure_next_6h`.
   - `failed_breakdown_next_24h`.
   - `future_return_6h`.
   - `time_to_poc_or_next_resistance`.
6. Controls:
   - Same support setup without reclaim.
   - Support break continuation cases.
   - Random rows near support.
   - Opposite resistance rejection setups.
7. Ablations:
   - Remove VP support.
   - Remove TLV2 support.
   - Remove pattern/range boundary.
   - Remove orderbook bid rebuild.
   - Remove reclaim/acceptance trigger.
   - Test each venue separately.
8. Data-window requirements:
   - Requires clean support-state history and no carried support/wall state through archive gaps.
   - Needs enough post-test candles to distinguish bounce from chop.
   - Context can be added only in windows where quiet/stress classification is available before the bounce.
9. Expected failure modes:
   - Bounce target too small after fees/slippage.
   - Strong downtrends make support tests continuation setups.
   - Bid rebuild may be reactive after the bounce rather than predictive.
   - Multi-source support definitions may be highly collinear.

### 4.5 Liquidity Vacuum/Travel

1. Trader theory:
   - Price travels faster when it breaks into a low-volume, thin-liquidity pocket with little resting book resistance before the next meaningful target.
2. Required source-detail blocks:
   - Price/volume, VP, market structure, pattern geometry, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook.
   - TLV2 support/resistance for target validation where available.
3. Candidate setup components:
   - Current price near edge of value/range and next zone has low VP density or LVN.
   - Orderbook depth is thin in the travel direction across one or more venues.
   - Opposing walls are distant, weak, or evaporating.
   - Recent volatility is compressed enough that expansion is measurable.
4. Candidate trigger components:
   - Break into the LVN/thin-book zone with range expansion.
   - Directional volume pressure agrees with break direction.
   - Spread does not widen so much that execution quality dominates.
   - No immediate counter-wall rebuild.
5. Candidate labels/targets:
   - `future_max_upside_6h` or `future_max_drawdown_6h`.
   - `time_to_plus_2pct` or `time_to_minus_2pct`.
   - `time_to_next_vp_or_wall_target`.
   - `hit_target_before_reversal`.
6. Controls:
   - Same breakout/breakdown trigger without LVN/thin-book path.
   - Random rows inside high-volume nodes.
   - Opposite-direction travel setup.
   - Shuffled target-distance labels.
7. Ablations:
   - Remove VP LVN/path definition.
   - Remove orderbook thin-depth path.
   - Remove target-distance requirement.
   - Remove volume expansion.
   - Test each orderbook venue path separately.
8. Data-window requirements:
   - Requires reliable VP levels and orderbook depth snapshots in the same window.
   - Must separate spot, linear, and inverse depth because each can imply different liquidity paths.
   - Exclude sparse orderbook periods where thin depth is a data outage.
9. Expected failure modes:
   - Thin-book signal is an artifact of missing snapshots.
   - LVN travel direction is symmetric and lacks directional edge without trigger quality.
   - Spread shock makes theoretical travel untradeable.
   - Target labels over-reward rare extreme candles.

### 4.6 Quiet Technical Market

1. Trader theory:
   - When news, macro, and orderbook stress are quiet, technical structure and local volume behaviour should explain more of the move than event context.
2. Required source-detail blocks:
   - Price/volume, VP, TLV2 support/resistance, market structure, pattern geometry, spot/linear/inverse orderbook balance.
   - GDELT events, GKG, live news/web, Trends, macro/ETF blocks only to prove quietness.
3. Candidate setup components:
   - Context attention and severity near baseline across timestamp-safe sources.
   - Orderbook imbalance, spread shock, and wall evaporation are not extreme.
   - Price is inside a defined range/value area or near a clear technical boundary.
   - Volatility is normal or compressed rather than event-driven.
4. Candidate trigger components:
   - Technical trigger selected by sub-mode: range rejection, mean reversion to POC, clean breakout, or clean breakdown.
   - Volume confirms the chosen sub-mode.
   - No context or orderbook stress spike appears before the trigger.
5. Candidate labels/targets:
   - `future_return_6h`.
   - `revert_to_poc_next_6h`.
   - `breakout_success_next_6h` for quiet breakout sub-mode.
   - `fakeout_next_24h` for quiet range sub-mode.
6. Controls:
   - Same technical trigger during stress context.
   - Same technical trigger during orderbook imbalance shock.
   - Random quiet rows without trigger.
   - Shuffled quiet/stress regime labels.
7. Ablations:
   - Remove quiet-context requirement.
   - Remove orderbook-balance requirement.
   - Remove each technical source detail.
   - Test GDELT/GKG/live news/Trends/macro quiet filters individually.
8. Data-window requirements:
   - Requires source coverage sufficient to identify quietness, not merely missing context data.
   - Use separate windows for live-news-era quietness versus GDELT/GKG historical quietness.
   - Must preserve explicit missing flags for context sources.
9. Expected failure modes:
   - Missing news data is mistaken for quiet market.
   - Quiet regime has too little volatility for useful targets.
   - Technical sub-modes conflict if combined into one broad label.
   - Orderbook balance filters remove too many samples.

### 4.7 Macro/News Stress Regime

1. Trader theory:
   - During credible macro/news stress, technical breaks and orderbook thinning should have larger and faster follow-through than the same technical events in quiet regimes.
2. Required source-detail blocks:
   - GDELT events, GKG documents, live news/web, Google Trends, global/macro/ETF flows, price/volume, VP/TLV2/market structure, spot/linear/inverse orderbook.
3. Candidate setup components:
   - Event severity or topic attention above baseline before the move.
   - Multiple source-detail blocks agree on the stress topic or risk channel.
   - Macro/ETF/global pressure aligns with risk-off or liquidity stress where available.
   - Price is near a technical decision level, not already far from structure.
4. Candidate trigger components:
   - Support break, failed support reclaim, or downside BOS during stress.
   - Orderbook support removal, spread shock, or cross-venue sell pressure.
   - Volume expansion confirms risk-off move.
   - Stress signal persists across adjacent windows rather than a one-candle spike.
5. Candidate labels/targets:
   - `large_drawdown_next_6h`.
   - `large_drawdown_next_24h`.
   - `hit_minus_3pct_before_plus_2pct`.
   - `volatility_expansion_next_24h`.
6. Controls:
   - Same stress regime without technical trigger.
   - Same technical trigger during quiet context.
   - Random rows in stress windows.
   - Opposite upside breakout during stress.
7. Ablations:
   - Remove each context block: GDELT, GKG, live news/web, Trends, macro/ETF.
   - Remove context persistence and use only point-in-time spike.
   - Remove orderbook thinning.
   - Remove technical decision-level requirement.
   - Remove volume confirmation.
8. Data-window requirements:
   - Requires exact available-at logic for each context source.
   - Must split historical GDELT/GKG windows from live-news-era windows if coverage differs.
   - Macro releases and ETF flows require release/availability timestamps, not calendar-date backfill.
9. Expected failure modes:
   - Source attention rises after price moves, creating reverse causality.
   - Stress categories are too broad and mix unrelated events.
   - Rare-event sample size is too small for stable lift.
   - Daily or delayed context data is assigned too early.

### 4.8 Orderbook Divergence/Reversal

1. Trader theory:
   - Reversal or fakeout risk rises when price continues in one direction but orderbook pressure, wall relocation, or venue agreement diverges against that move.
2. Required source-detail blocks:
   - Price/volume, market structure, VP/TLV2/pattern decision levels, spot orderbook, Bybit linear orderbook, Bybit inverse orderbook.
3. Candidate setup components:
   - Price trend or breakout/breakdown is extended relative to recent structure.
   - Price approaches a VP/TLV2/pattern level where reversal would be plausible.
   - One or more venues show weakening pressure in trend direction.
   - Opposing liquidity rebuilds or walls relocate against the move.
4. Candidate trigger components:
   - Price makes marginal new high/low while orderbook pressure fails to confirm.
   - Cross-venue divergence: spot confirms but derivative venue does not, or linear/inverse disagree.
   - CHoCH, failed acceptance, or close back inside prior range.
   - Volume pressure fades while price remains extended.
5. Candidate labels/targets:
   - `fakeout_next_24h`.
   - `future_return_6h` signed against prior move.
   - `future_max_drawdown_6h` after upside divergence or `future_max_upside_6h` after downside divergence.
   - `return_inside_range_next_6h`.
6. Controls:
   - Same price extension with orderbook confirmation.
   - Random extended-trend rows.
   - Same divergence away from technical decision levels.
   - Opposite-direction divergence setups.
7. Ablations:
   - Remove cross-venue disagreement.
   - Remove wall relocation.
   - Remove pressure divergence.
   - Remove technical decision-level filter.
   - Test spot, linear, inverse as primary venue one at a time.
8. Data-window requirements:
   - Requires overlapping orderbook venue data with synchronized timestamps.
   - Must detect and exclude archive gaps per venue rather than forward-filling divergence.
   - Needs enough trend history before setup to avoid labeling normal noise as divergence.
9. Expected failure modes:
   - Venue divergence is structural market microstructure, not predictive reversal signal.
   - Divergence appears too often during normal consolidation.
   - Sparse snapshots create false pressure fades.
   - Reversal horizon is too short or too long for observed fakeouts.

## 5. Practical Priority Order

1. Highest priority:
   - Breakout confirmation.
   - Breakdown/crash detection.
   - Failed breakout/rejection.
   - Support bounce/reclaim.
2. Next priority:
   - Liquidity vacuum/travel.
   - Orderbook divergence/reversal.
3. Later priority:
   - Quiet technical market.
   - Macro/news stress regime.

Reason: the highest-priority packs directly map to current implemented/designed hypothesis IDs and can be tested with structure, volume, and orderbook windows before depending on harder context timestamp coverage. Quiet and stress regimes are important, but they should wait until context coverage and available-at rules are audited.
