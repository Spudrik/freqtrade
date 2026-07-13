---
doc_status: active
default_read: routed
owner: agent
purpose: Compact summary of the first historical news/context overlay test batches.
do_not_use_for: Final strategy promotion or Sieve Hyperopt result interpretation.
last_rebuilt: 2026-06-15
---

# Historical News Overlay Results - 2026-06-15

## Scope

Tested source-masked historical GDELT/GKG aggregate news features up to `2022-12-31` as potential overlays for Sieve-style entries, exits, risk, and position management.

This is exploratory evidence only. It is not final trading logic and not a Sieve replacement.

## Artifacts

Script:

- `C:\FreqTradeStuff\user_data\Custom_Launcher\research\context_features\historical_news_overlay_research.py`

Latest batch:

- Theory CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\historical_news_overlay\historical_news_overlay_theory_results_batch3_refined_w2_w1_w3.csv`
- Model screen CSV: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\historical_news_overlay\historical_news_overlay_feature_screen_batch3_refined_w2_w1_w3.csv`
- Markdown report: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\historical_news_overlay\historical_news_overlay_report_batch3_refined_w2_w1_w3.md`
- Metadata: `C:\FreqTradeStuff\user_data\research_news_data\context_features\reports\historical_news_overlay\historical_news_overlay_meta_batch3_refined_w2_w1_w3.json`

Code/theory reviews:

- Theory review sub-agent: challenged raw-count ideas and pushed crypto stress, relief, persistence, quiet-news regime, and news-price mismatch.
- Code review sub-agent: found target-tail and source-mask issues.
- Re-review sub-agent: cleared the fixed script for exploratory use, with caveats around mixed-source composites and quiet-news interpretation.

## Windows

1. `w2_2021_mar_2022_jul`
   - Main discovery window.
   - Uses GDELT events plus GKG aggregate topic features.

2. `w1_2020_jan_oct`
   - Transfer/sanity window.
   - Uses GDELT events plus GKG aggregate topic features, with October partial.

3. `w3_2022_aug_dec_gdelt_only`
   - GDELT-only validation window.
   - Do not treat GKG feature rows as available here.

## Strongest Lead Families

### 1. Relief Bounce

Trader idea:

Bad news or macro/crypto stress was high, but pressure starts fading near support or after a selloff. That can mark exhaustion rather than continuation.

Evidence examples:

- `macro_relief_after_selloff`, `w3`, `up_2pct_next_24h`: `48` trigger rows, event rate `0.563` vs baseline `0.320`, lift `+0.243`, positive in `4/4` active months.
- `macro_relief_bounce`, `w1`, `up_2pct_next_24h`: `46` trigger rows, event rate `0.609` vs baseline `0.413`, lift `+0.196`.
- `macro_relief_bounce`, `w2`, `up_2pct_next_24h`: `180` trigger rows, event rate `0.611` vs baseline `0.535`, lift `+0.076`.
- `crypto_relief_bounce`, `w1`, `up_2pct_next_24h`: `29` trigger rows, event rate `0.586` vs baseline `0.411`, lift `+0.176`.

Verdict:

Good lead. This is trader-readable and repeatedly appears as a bounce/reclaim overlay. Next test should join it to Sieve reclaim/bounce/support entries.

### 2. Bad News Absorbed By Support

Trader idea:

If news is bad but price refuses to break support, that can be absorption and may support a bounce.

Evidence examples:

- `bad_news_price_holds_support`, `w3`, `up_2pct_next_24h`: `121` rows, event rate `0.500` vs baseline `0.317`, lift `+0.183`.
- `bad_news_absorbed_bounce`, `w3`, `hit_plus_3_before_minus_2_72h`: `91` rows, event rate `0.483` vs baseline `0.277`, lift `+0.206`.
- `bad_news_absorbed_bounce`, `w1`, `up_2pct_next_24h`: `28` rows, event rate `0.500` vs baseline `0.413`, lift `+0.087`.

Verdict:

Good but needs Sieve/structure overlay. It is plausible as a stake/add/hold signal for support-reclaim or bounce entries.

### 3. Bad News Breakdown Continuation

Trader idea:

Once price has already broken lower, bad news can help decide whether the drop continues.

Evidence examples:

- `bad_news_breakdown_continuation`, `w2`, `down_2pct_next_24h`: `168` rows, event rate `0.667` vs baseline `0.559`, lift `+0.108`, positive in `11/16` active months.
- `crypto_stress_breakdown_continuation`, `w2`, `down_2pct_next_24h`: `107` rows, event rate `0.682` vs baseline `0.563`, lift `+0.119`.
- `policy_stress_breakdown_continuation`, `w2`, `down_2pct_next_24h`: `116` rows, event rate `0.672` vs baseline `0.563`, lift `+0.109`.
- `bad_news_support_cracks`, `w2`, `down_2pct_next_24h`: `57` rows, event rate `0.649` vs baseline `0.560`, lift `+0.089`.

Verdict:

Good lead. Best near-term use is not new entries by itself, but short confirmation, long reduction, stop tightening, and crash-continuation warning.

### 4. Attention / Compression Expansion

Trader idea:

When news attention is unusually high after compression, the market is more likely to expand. Direction depends on whether price is already leaning up or down.

Evidence examples:

- `attention_compression_downside`, `w2`, `large_drawdown_next_24h`: `258` rows, event rate `0.500` vs baseline `0.321`, lift `+0.179`, positive in `11/15` active months.
- `attention_compression_downside`, `w2`, `down_2pct_next_24h`: `258` rows, event rate `0.694` vs baseline `0.557`, lift `+0.137`.
- `news_attention_expansion`, `w2`, `large_drawdown_next_24h`: `452` rows, event rate `0.471` vs baseline `0.319`, lift `+0.152`.
- `attention_compression_upside`, `w1`, `up_2pct_next_24h`: `269` rows, event rate `0.498` vs baseline `0.410`, lift `+0.088`.

Verdict:

Good lead. Treat as volatility/risk regime first, direction second. Useful for stop tightening, avoiding adds, or expecting larger move after entry.

### 5. High-Attention Breakout Follow-Through

Trader idea:

Breakouts with volume and broad attention may have better follow-through, especially when not already overextended.

Evidence examples:

- `attention_breakout_not_overextended`, `w1`, `up_2pct_next_24h`: `56` rows, event rate `0.554` vs baseline `0.413`, lift `+0.141`.
- `high_attention_breakout_followthrough`, `w1`, `up_2pct_next_24h`: `69` rows, event rate `0.551` vs baseline `0.412`, lift `+0.138`.
- `attention_breakout_not_overextended`, `w2`, `up_2pct_next_24h`: `82` rows, event rate `0.634` vs baseline `0.536`, lift `+0.099`.
- `high_attention_breakout_followthrough`, `w2`, `up_2pct_next_24h`: `100` rows, event rate `0.610` vs baseline `0.536`, lift `+0.074`.

Verdict:

Good lead. Best use is as an overlay on Sieve breakout entries, not as news-only entry logic.

### 6. Stress At Range High / Resistance

Trader idea:

If price is stretched near a range high and macro/crypto stress is active, long continuation is weaker and rejection risk is higher.

Evidence examples:

- `crypto_stress_range_high_rejection`, `w1`, `down_2pct_next_24h`: `327` rows, event rate `0.468` vs baseline `0.388`, lift `+0.079`.
- `macro_stress_range_high_rejection`, `w2`, `down_2pct_next_24h`: `344` rows, event rate `0.628` vs baseline `0.558`, lift `+0.070`.
- `bad_news_price_stalls_resistance`, `w3`, `down_2pct_next_24h`: `35` rows, event rate `0.400` vs baseline `0.344`, lift `+0.056`.

Verdict:

Worth investigating. This is probably a risk/reduce/filter overlay for longs at resistance, and a confirmation overlay for rejection shorts.

### 7. Bad News Ignored By Rising Price

Trader idea:

Bad news does not always mean downside. If price rises anyway, that can show strength.

Evidence examples:

- `macro_stress_price_ignores_up`, `w1`, `up_2pct_next_24h`: `346` rows, event rate `0.529` vs baseline `0.408`, lift `+0.121`.
- `crypto_stress_price_ignores_up`, `w1`, `up_2pct_next_24h`: `319` rows, event rate `0.517` vs baseline `0.406`, lift `+0.111`.
- `crypto_breakout_resilience`, `w1`, `up_2pct_next_24h`: `58` rows, event rate `0.517` vs baseline `0.410`, lift `+0.107`.
- `macro_stress_price_ignores_up`, `w3`, `up_2pct_next_24h`: `74` rows, event rate `0.405` vs baseline `0.321`, lift `+0.084`.

Verdict:

Interesting lead. This is the opposite of naive sentiment filtering and should be tested as a strength/hold/add overlay.

## Model Screen Notes

The model screen produced many lead rows, but price/regime features remain dominant. The most useful news hints were:

1. `news_activity__pct_rank_720h`
2. `gdelt__event_count__pct_rank_720h`
3. `gdelt__num_mentions_sum__pct_rank_720h`
4. `gdelt__num_articles_sum__delta_24h`
5. `gdelt__macro_url_count__persistent_24h`
6. `gdelt__crypto_url_count__pct_rank_720h`
7. `gdelt_negative_tone`

Treat model output as feature-ranking only. Direct theory tests are the stronger evidence from this batch.

## Rejected / Weak For Now

1. Raw article/event/document counts by themselves.
2. News-only entries as a strategy.
3. Generic macro stress alone without price/level context.
4. GKG topic conclusions in `w3_2022_aug_dec_gdelt_only`, because GKG is not usable there.
5. Quiet-news conclusions unless the exact available source subset is reported.

## Next Step

Build a Sieve-overlay validation batch:

1. Join these news states to Sieve entry/export rows.
2. Score by entry family:
   - breakout longs,
   - support reclaim/bounce longs,
   - resistance rejection shorts,
   - support-break shorts,
   - overtrade entries.
3. Test whether each lead improves:
   - win rate,
   - drawdown risk,
   - hit `+2%` before `-2%`,
   - hit `-3%` before `+2%`,
   - exit urgency after entry.
4. Only after that, build a dedicated FreqAI validation profile for the strongest overlay states.
