---
doc_status: active
default_read: routed
owner: agent
purpose: Restart plan for historical news/context overlay research after Sieve strategy work was separated.
do_not_use_for: Sieve Hyperopt instructions, raw report dumps, or final trading-strategy promotion.
last_rebuilt: 2026-06-15
---

# News / Context Overlay Restart Plan - 2026-06-15

## 1. Scope

The goal is to test whether historical news/context can improve Sieve-style trading decisions without replacing Sieve.

Use news/context as overlay evidence for:

1. entry confidence,
2. long/short bias,
3. fakeout risk,
4. crash continuation risk,
5. exit urgency,
6. position reduction or stake reduction,
7. occasional news-only regime warning if evidence is strong.

Do not treat this as a non-Sieve strategy-lane revival. Sieve remains the trading-system discovery path.

## 2. Source Readiness

### 2.1 Usable Now, With Masks

1. `context_gdelt_events`
   - Table: `gdelt_hourly_features`
   - Window: `2020-01-01` to `2026-05-28`
   - Historical coverage is broad enough for first-pass aggregate tests.
   - Example fields: `event_count`, `num_mentions_sum`, `num_sources_sum`, `avg_tone_weighted`, `goldstein_weighted`, `conflict_event_count`, `protest_event_count`, `coercion_event_count`, `macro_url_count`, `crypto_url_count`.
   - Use as market-event pressure, conflict/protest/coercion stress, macro/crypto URL activity, and weighted tone context.

2. `context_gkg_documents`
   - Table: `gdelt_gkg_file_features`
   - Best usable historical windows:
     - `2020-01-01` to `2020-10-31` with October partial.
     - `2021-03-01` to `2022-07-31`, best first research window.
   - Not usable as continuous GKG in `2022-08` onward from current aggregate rows.
   - Example fields: `document_count`, `source_count`, `tone_sum`, `negative_tone_sum`, `positive_tone_sum`, `polarity_sum`, `unique_theme_count`, `crypto_doc_count`, `bitcoin_doc_count`, `ethereum_doc_count`, `stablecoin_liquidity_doc_count`, `regulation_doc_count`, `etf_institutional_doc_count`, `security_exploit_doc_count`, `macro_economic_doc_count`, `central_bank_doc_count`, `rates_doc_count`, `inflation_doc_count`, `jobs_labor_doc_count`, `recession_growth_doc_count`, `banking_credit_doc_count`, `oil_energy_doc_count`, `sanctions_trade_doc_count`, `war_geopolitics_doc_count`.
   - Use as topic-weighted news pressure, not as article-level truth.

3. `context_features_1h`
   - Table: `context_features_1h`
   - Window: `2020-01-01` to `2026-05-24`
   - Rows align hourly and have no detected future-source violations in the audited windows.
   - This is acceptable for quick aggregate tests only. Do not overclaim it as nuanced news analysis.

4. `orderbook_bybit_trader_state`
   - Files:
     - `user_data/orderbook_data/historical_bybit/features/orderbook_trader_state_1h_bybit_linear.parquet`
     - `user_data/orderbook_data/historical_bybit/features/orderbook_trader_state_1h_bybit_inverse.parquet`
   - Window: `2023-01-18 07:00 UTC` to `2026-05-26 01:00 UTC`
   - Historical news/orderbook overlap in the current requested range is mainly `2023-01-18` to `2023-06-30`.
   - Use only where `orderbook_present` is true. Do not carry wall/zone state through missing archive gaps.

### 2.2 Not Ready For Historical 2020-2023 Tests

1. `gkg_documents_silver`
   - Current rows: `0`.
   - Not usable for article/story-level source nuance yet.

2. `gdelt_events_silver`
   - Current rows: `0`.
   - Not usable for normalized event-level source nuance yet.

3. `story_clusters`
   - Current rows: `0`.
   - Not usable for true story lifecycle, first-mention, or source-spread tests yet.

4. `context_live_news_web` and `context_global_market_macro`
   - Current useful data is mostly 2026 live-era data.
   - Do not mix with 2020-2023 historical news tests.

5. Raw GKG rebuild
   - `D:\FreqTradeStuffLargeData\research_news_data\gdelt\raw\gkg` currently has no files on disk.
   - SQLite metadata says many GKG stamps were downloaded, but raw reparse is blocked unless files are recovered or redownloaded.

## 3. Valid Research Windows

1. `W1_gkg_2020_jan_oct`
   - `2020-01-01` to `2020-10-31`.
   - Use GDELT events plus GKG aggregate topics.
   - October is partial, so month-stability checks must identify whether results depend on it.

2. `W2_gkg_2021_mar_2022_jul`
   - `2021-03-01` to `2022-07-31`.
   - Best first window for GDELT events plus GKG aggregate topic testing.
   - Use this as the primary news-overlay discovery window.

3. `W3_gdelt_only_2022_aug_dec`
   - `2022-08-01` to `2022-12-31`.
   - GDELT events only. Do not use GKG topic/document features here except as missing-source flags.

4. `W4_gdelt_orderbook_2023_jan_jun`
   - `2023-01-18 07:00 UTC` to `2023-06-30`.
   - Use GDELT events plus Bybit orderbook features.
   - GKG aggregate is not usable in this window.

5. `W5_live_news_2026`
   - Separate live-era research only.
   - Use later to validate article-level concepts after historical tests define useful shapes.

## 4. Feature Conditioning Rules

Use raw article/event counts only as ingredients. Convert them into trader-readable states.

Required conditioning:

1. `source_present`
   - Whether the source was actually operating for that hour.

2. `source_age_hours`
   - How stale the newest source data is.

3. `topic_intensity_z_24h`, `topic_intensity_z_7d`
   - Whether a topic is unusually active versus its own recent baseline.

4. `topic_escalation_6h`, `topic_escalation_24h`
   - Whether topic pressure is rising fast.

5. `topic_persistence_24h`, `topic_persistence_72h`
   - Whether the story pressure keeps showing up.

6. `topic_relief_6h`, `topic_relief_24h`
   - Whether pressure is fading after a spike.

7. `topic_negative_pressure`
   - Negative tone and polarity after normalizing by document/source activity.

8. `source_confluence`
   - Number of independent source/topic blocks agreeing. For current aggregate data this is approximate, not article-level source truth.

9. `quiet_news_regime`
   - Only valid if the source is present and activity is low versus its own baseline.

10. `news_market_mismatch`
    - Price moves strongly while news is quiet, or news is severe while price is ignoring it.

11. `pre_entry_news_state`
    - News state measured before a Sieve/indicator entry, for entry filtering or stake sizing.

12. `post_break_news_state`
    - News state after a break/drop has started, for crash continuation, stop tightening, or partial exits. This must not be used as an entry predictor for a trade that would already have happened.

## 5. Trader-Readable Concept Packs To Test

### 5.1 News-Only Market State

1. Macro stress escalation predicts downside.
   - If rates, inflation, recession, banking, war, oil, or sanctions pressure rises sharply, does BTC downside risk increase over the next `6h`, `24h`, `3d`, or `7d`?

2. Crypto-specific stress predicts downside.
   - Regulation, security exploit, stablecoin liquidity, and crypto-topic negative pressure may matter more than broad macro news.

3. Relief after panic predicts bounce.
   - If negative pressure was extreme but starts fading while price stops making lower lows, does bounce probability improve?

4. Persistent bad news differs from one-hour shock.
   - A single spike may be noise. Repeated pressure across `24h-72h` may be more tradable.

5. Quiet news regime favours technical continuation.
   - If news is quiet and price breaks a meaningful level with volume, technical levels may dominate.

6. News ignored by price may become delayed risk.
   - Severe news with no immediate price reaction may predict later volatility or failed bullish continuation.

7. News shock after compression predicts expansion.
   - If volatility is compressed and news pressure jumps, does the next range expansion have direction?

8. News-only entries.
   - Test for completeness, but assume this is unlikely to become a final strategy unless it beats controls and survives month splits.

### 5.2 News As Overlay On Sieve / Indicator Entries

1. Bearish entry plus bad news.
   - A Sieve short or support-break short should be stronger when macro/crypto stress is escalating.

2. Bullish entry plus bad news.
   - A Sieve long or breakout long may be lower quality if bad news is escalating.

3. Bullish entry plus relief.
   - A reclaim/bounce long may improve when panic pressure is fading.

4. Breakout with quiet news.
   - If no external shock is active, breakouts may be judged more by volume, VP/TLV2/structure, and orderbook.

5. Breakout into negative news.
   - Test whether bullish breakouts fail more often when severe news is active.

6. Rejection entries with bad news.
   - Resistance rejection shorts should be stronger if negative news pressure is active.

7. Support reclaim against bad news.
   - A bullish reclaim that works despite bad news may indicate strong absorption or institutional support.

8. Multiple-timeframe support/resistance plus news.
   - If daily/3d resistance is active and bad news escalates, short/rejection entries may have better follow-through.

9. Sieve entries during source-unavailable periods.
   - These are not bad trades by default, but they cannot be scored as news-confirmed or news-quiet.

10. News as stake/risk modifier.
    - Do not only ask entry yes/no. Test whether news should reduce stake, tighten stops, or block adds.

### 5.3 Crash / Downside Continuation

1. Drop starts, news escalates, orderbook support retreats.
   - If price has already broken lower and bad news keeps rising, treat news as continuation/risk evidence.

2. Drop starts, news fades.
   - If price breaks down but news pressure fades, the move may be more likely to stall or bounce.

3. Bad news plus liquidity vacuum.
   - In `W4`, combine GDELT stress with Bybit orderbook features showing support removal, thin bid depth, or walls moving lower.

4. Bad news plus sell-side book dominance.
   - If orderbook leans sell-side and price breaks a lower low, test continuation/drawdown risk.

5. Crash exit/reduce signal.
   - Use post-break evidence for reducing longs, tightening stops, or avoiding new longs, not as hindsight entry evidence.

### 5.4 Orderbook / News Confluence

1. News stress plus support wall removed.
   - Bearish continuation should improve if support disappears while stress escalates.

2. News stress plus new walls far below price.
   - Test whether market makers are repricing lower support zones.

3. News stress plus reduced wall persistence.
   - If persistent levels stop holding during a bad-news regime, risk is higher.

4. News quiet plus balanced orderbook.
   - Technical breakout/rejection logic may work better when neither news nor book is making extreme statements.

5. News stress but bid absorption holds.
   - Bad news that fails to break price or book support may indicate a contrarian bounce setup.

6. News relief plus ask wall retreat.
   - Test whether improving context plus disappearing overhead liquidity supports upside continuation.

### 5.5 Market Regime And Level Context

1. Long-running resistance plus bad news.
   - If price has repeatedly failed at a daily/3d resistance and bad news escalates, rejection shorts may improve.

2. Long-running support plus relief.
   - If price is at established support and news pressure fades, bounce/reclaim longs may improve.

3. Range high/low with news quiet.
   - Quiet external context may make VP/TLV2/structure levels more reliable.

4. News shock at range boundary.
   - Bad news at range support may cause breakdown; good/relief news at range resistance may help breakout.

5. Trend state matters.
   - Bad news in a downtrend is not the same as bad news during a strong uptrend. Split tests by trend/range regime.

### 5.6 Failure And Fakeout Concepts

1. Bullish breakout failure after bad news.
   - Price breaks resistance, but negative news escalates and price cannot hold above the level.

2. Bearish breakdown failure after relief.
   - Price breaks support, but news pressure fades and price reclaims the level.

3. Divergence between news and price.
   - Severe news with rising price may be strength; severe news with stalled price may be risk.

4. Divergence between news and orderbook.
   - Bad news but supportive book may be absorption; bad news plus book retreat may be confirmation.

5. Noisy-topic traps.
   - High article/document activity with no topic concentration may be noise, not signal.

## 6. Direct Test Design Before FreqAI

Each concept should be tested in plain evidence first:

1. Define the event condition in trader language.
2. Build an eligible-row mask where required sources are present.
3. Compare future outcomes:
   - `future_return_6h`
   - `future_return_24h`
   - `future_return_3d`
   - `future_return_7d`
   - `future_max_drawdown_24h`
   - `future_max_upside_24h`
   - hit `+2%` before `-2%`
   - hit `-3%` before `+2%`
4. Compare against:
   - same-regime rows,
   - random eligible rows,
   - shuffled labels,
   - month/year splits,
   - Sieve entry outcomes where available.
5. Keep concepts that improve direction, risk, or exit timing without relying on one lucky month.

## 7. FreqAI Use

Use FreqAI only after direct tests show a concept is plausible.

Valid FreqAI tasks:

1. Rank which topic-pressure transforms matter most.
2. Test whether news improves Sieve entry outcome prediction.
3. Test whether news improves exit timing after an entry.
4. Test whether orderbook plus news improves crash continuation detection.
5. Test whether news should change stake/stop/add/reduce decisions.

Invalid FreqAI tasks:

1. Throw all raw article counts at the model and call the result insight.
2. Train on rows where the source was missing and call missing data quiet.
3. Mix 2026 live RSS/web/global with 2020-2023 historical GDELT windows.
4. Use post-entry data to justify a pre-entry decision.

## 8. First Executable Batch

1. Build a source-readiness mask table for `W2_gkg_2021_mar_2022_jul`.
2. Build aggregate news-state features:
   - macro stress escalation,
   - crypto stress escalation,
   - regulation/security/stablecoin stress,
   - negative pressure,
   - relief/fading pressure,
   - quiet-news regime.
3. Join to BTC 1h OHLCV and Sieve entry/outcome exports if available.
4. Direct-test:
   - news-only downside risk,
   - news overlay on bearish entries,
   - news overlay on bullish entries,
   - relief overlay on bounce/reclaim entries,
   - quiet-news technical breakout/rejection rows.
5. Repeat the strongest shapes on `W1`.
6. Use `W4` only for GDELT-events plus Bybit orderbook crash/continuation confluence.

## 9. Audit Artifacts

Quality reports:

1. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_manifest_news_context_overall_2020_2023.json`
2. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_gates_manifest_news_context_overall_2020_2023.json`
3. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_manifest_news_context_gkg_a_2020_jan_oct.json`
4. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_gates_manifest_news_context_gkg_a_2020_jan_oct.json`
5. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_manifest_news_context_gkg_b_2021_mar_2022_jul.json`
6. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_gates_manifest_news_context_gkg_b_2021_mar_2022_jul.json`
7. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_manifest_news_context_event_ob_overlap_2023_jan_jun.json`
8. `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\gdelt_gkg_quality_gates_manifest_news_context_event_ob_overlap_2023_jan_jun.json`

Code fix:

1. `user_data/Custom_Launcher/research/context_features/gdelt_gkg_quality_gates.py`
   - Fixed broad-window quality gate failure by batching raw metadata `IN (...)` queries.

## 10. Current Verdict

Accepted for first-pass research:

1. Aggregate GDELT event features across 2020-2023.
2. Aggregate GKG document/topic features in validated windows only.
3. Bybit orderbook confluence only from `2023-01-18` onward.
4. News overlay as entry confidence, risk, stop/exit urgency, and crash-continuation evidence.

Blocked or parked:

1. Article-level source nuance for historical GKG until raw files or silver tables are repaired.
2. Story-cluster lifecycle testing until `story_clusters` and members exist.
3. Live RSS/web/global context for 2020-2023.
4. Any final trading promotion until direct tests and clean source windows show stable value.
