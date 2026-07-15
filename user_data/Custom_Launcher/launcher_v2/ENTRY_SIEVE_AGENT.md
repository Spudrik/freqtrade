# Entry Sieve Agent Notes

Read this first:

- `user_data/Custom_Launcher/docs/EXPLORER_ENTRY_SIEVE_MANUAL.md`
- `user_data/strategies/AGENTS.md`

This file is intentionally short to avoid stale duplicated rules.

Current source of truth:

- Entry Sieve is entry-quality diagnostics only. The active pass may be Sieve1, Sieve2, or a later sieve generation.
- Active sieve generations are entry-testing only until the user explicitly finalises the entry sieves. Do not add, tune, or optimise exits, stoploss logic, take-profit logic beyond configured sieve target tests, leverage, position sizing, DCA/adds, cooldowns, protections, or broader trade-management behavior during Sieve2/Sieve3 entry refinement.
- Entry-quality Sieve runs must keep `Sieve controls entry exits` enabled so the launcher controls and validates the requested fixed TP/SL. Exit-stage Sieve work must turn that control off so strategy exit logic owns TP/SL, trailing, partials, stop movement, and other exit parameters.
- Entry Sieve results are informative diagnostics, not pass/fail acceptance.
- Strategy agents work on strategy files and explicitly requested strategy infrastructure.
- Strategy agents must not tune, reshape, or edit indicator modules unless explicitly asked.
- Indicator contracts and strategy-facing indicator docs take priority.
- Normal sieve files should test one core entry concept with a small coarse hyperopt surface.
- Sieve entry-test strategy files should use `entry_sieve_minimal_roi(...)` and `entry_sieve_stoploss(...)`, or an exact env-driven equivalent, instead of hardcoded class-level ROI/stoploss values.
- Explicit confluence probes belong in `multiN` files/classes for the active sieve pass.
- Sieve progression is generation-based. During Sieve3, refined Sieve2 foundations and brand-new ideas both use `sieve3_` files/classes.
- Sieve3 refinement files must keep source metadata such as `SIEVE_STAGE`, `SOURCE_STRATEGY`, `SOURCE_RESULT_BATCH`, and `RESEARCH_PATH`.
- Brand-new Sieve3 ideas must be marked as novel, normally with `sieve3_novel_*` naming and `NOVEL_IDEA = True` metadata.
- Sieve3 promotion should use broad judgement rather than hard gates: compare result shape, trade count, winrate, profit, drawdown, TP/SL behavior, and whether the entry has a clear market-structure reason before choosing keep, split, reframe, or archive.
- Use the discussed Sieve3 examples as practical guidance, not an exhaustive rulebook. High-winrate low-trade entries can be loosened or given better lower-timeframe execution. Profitable high-trade sub-50% winrate entries can be refined with retests, VP/TLV2 location guards, local BOS/CHOCH, failed-break filters, volume-pressure confirmation, or market-state guards. Strong 4/2 TP/SL behavior around 40%+ winrate should preserve the asymmetric entry idea while improving entry quality. High-profit but high-drawdown ideas should be guard/refinement candidates before any exit work. Low-drawdown stable ideas and structurally sensible near-misses can be kept when there is a clear repair path. Other good options are allowed if the agent can explain the market-structure reason and keep the branch focused.
- One promising source may branch into several Sieve3 files when each branch tests a distinct refinement path, such as VP guard, TLV2 guard, HTF context, retest, local structure, combined VP/TLV2 guard, or retest plus HTF guard.
- Sieve3 candidates are not limited to one result type. Keep high-winrate low-trade ideas, profitable high-trade sub-50% winrate ideas, strong risk/reward ideas, low-drawdown stable ideas, and structurally sensible near-misses with clear repair paths.
- Do not discard useful foundations just because the current batch concept focuses on one subset.
- Weak or structurally unhelpful strategies may be archived out of active batches, but each archived file must first receive a short module-level comment explaining why it was parked.
- Archive comments should state the review reason, such as consistently negative results, excessive noisy trades, generic/non-structural trigger, redundant duplicate, lookahead-risk concern, or no clear repair path.
- Sieve2 MTF files must be higher-timeframe condition plus lower-timeframe structural execution, not broad context plus generic momentum.
- 1d/3d/4h patterns, TLV2 levels, volume-profile levels, BOS/CHOCH, HH/HL/LH/LL, supply/demand, and prior high/low levels are valid higher-timeframe contexts when they provide a clear execution line or invalidation.
- Lower-timeframe execution should test breakout, break-and-retest, rejection/reclaim, failed break, local BOS/CHOCH, or higher-low/lower-high continuation at the relevant level.
- Retest variants are first-class candidates and should be separated from raw breakout variants where practical.
- Single-candle momentum, close-vs-previous-close, EMA/SMA pullback, RSI/MACD/Bollinger/stochastic, or generic TA-stack conditions must not be the primary MTF trigger.
- Do not add broad defensive workaround code without root-cause analysis and approval.
