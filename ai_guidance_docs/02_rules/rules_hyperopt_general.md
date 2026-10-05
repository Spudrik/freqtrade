---
doc_status: active
default_read: routed
owner: user
purpose: General Hyperopt design and reporting rules, routed through Sieve by default.
do_not_use_for: Non-Sieve Hyperopt packages unless the user explicitly approves them.
last_rebuilt: 2026-10-05
---

# Hyperopt General Rules

## Core Rule

Follow repo-root `AGENTS.md` §7 for the Sieve invariant and the boundary for separately approved research.

Use these rules to design Sieve Hyperopt batches for entries, exits, partial exits, stop movement, trailing, position adjustment, staking, and risk management.

## Core Rules

1. Hyperopt should search broad, explicit theory spaces that would be inefficient to check manually.
2. Ordinary backtests are validation/sanity checks after Sieve identifies candidates, not the main discovery method.
3. Use categorical parameters for genuinely unordered trading modes; use ordered integer/decimal parameters when numeric order matters.
4. For coarse regular numeric values, Hyperopt an integer multiplier and apply the step at point of use.
5. Keep each Hyperopt strategy focused and generally below roughly `10,000` combinations; split only along coherent trading questions.
6. Named categorical modes remain useful for explicit rule choices, but do not hide ordered numeric structure inside categories.
7. Avoid gates that make other Hyperopt parameters irrelevant. A disabled gate must not create fake best values for parameters that never affected trades.
8. If an off state is needed, use an explicit sentinel/mode or split the file when the disabled branch would hide too many inactive parameters.
9. Avoid wide high-resolution decimal/integer sweeps. Small contiguous sweeps and coarse multiplier sweeps are preferred over arbitrary categorical numbers.
10. Review the top four Hyperopt objective/loss candidates per run, not only the winner. The loss calculator can lean too heavily into one metric.
11. Do not manually analyze whole result files in chat. Use scripts, CSV summaries, or result parsers to pull candidate rows.
12. Hyperopt loss files are first-pass ranking tools. If the scoring behaviour is wrong, propose adjusting the loss file rather than overriding results by hand.
13. Label exposure assumptions clearly: stake amount, reference wallet, max open trades, pair scope, and leverage.

## Search Design Standard

1. If a proposed Hyperopt run could be replaced by roughly `10-15` ordinary backtests, the search design is probably too small.
2. Before launching Hyperopt, put the main effort into designing the Sieve search space so Hyperopt can test meaningful combinations across the relevant decision layers.
3. A serious search should normally include several decision layers, such as:
   - entry mode,
   - exit target family,
   - partial-exit behaviour,
   - stop-tightening behaviour,
   - trailing behaviour,
   - add/stack behaviour,
   - reduce/de-risk behaviour,
   - stake/risk sizing,
   - entry-family-specific overrides where feasible.
4. Avoid mixing unrelated search questions into one huge noisy run. Split by Sieve batch/search family when needed.
5. Each split should still use Hyperopt's strength: testing combinations and named modes, not tiny one-off toggles.

## Sieve Exit-Control Boundary

1. Entry-quality Sieve runs must let the Sieve environment control fixed TP/SL. Strategy files used for entry testing should use `entry_sieve_minimal_roi(...)` and `entry_sieve_stoploss(...)`, or an exact env-driven equivalent, so the requested TP/SL is the real tested exit.
2. The runner should fail entry tests when the resolved strategy `minimal_roi` or `stoploss` does not match the requested Sieve TP/SL. A result label is not enough; the class values must match.
3. Exit-stage Sieve work must disable Sieve exit control. In exit research, the strategy's own exit parameters, stop movement, trailing, partial exits, or fixed-exit Hyperopt branch should control exits.
4. Do not judge entry edge from a run where Sieve labels say one TP/SL but the strategy class resolves a different ROI or stoploss.

## Parameter Design

1. Use categorical parameters for named trader theories:
   - `exit_target_mode = fixed_tp / vp_level / tlv2_level / prior_high_low / pattern_target / invalidation`
   - `partial_exit_mode = none / first_obstacle / profit_2pct / profit_3pct / opposing_signal / volume_exhaustion`
   - `stop_mode = static / breakeven_after_partial / trail_structure / trail_vp / tighten_on_risk`
   - `stack_mode = none / same_family / independent_family / retest / profit_buffer`
   - `risk_mode = flat / reduce_chop / reduce_crash_risk / boost_confluence / cap_direction_exposure`
2. Use ordered numeric parameters for ordered thresholds. Small integer ranges can remain ordinary `IntParameter` ranges; coarse regular steps should use an integer multiplier.
3. Do not hide active parameters behind disabled gates. If a mode disables a concept, make the inactive state explicit as a category.
4. For entry-family-specific management, use separate mode parameters per family when feasible instead of one global mode.
5. Design search spaces so the result is interpretable in trader terms. A winning mode should explain what happened, not just which number won.
6. Avoid hidden gates where possible. Hyperopt still varies parameters that sit behind Boolean enables or categorical branches, even when the branch is inactive. This can create misleading associations because inactive values may look good or bad despite having no effect.
7. Prefer a single null/sentinel parameter over an enable flag plus value parameter when the logic allows it. For threshold-style guards, use states such as `ignore / weak / normal / strict / impossible` so the parameter remains meaningful across the whole search.
8. Some logic still needs gates. Crossing events, trigger selection, and branch-specific rules may require Boolean enables or categorical selectors. Avoid gates where a clean null/sentinel state expresses the same thing; use gates only where the trading logic genuinely requires them.
9. For early discovery, use broad named modes for unordered theories and coarse ordered numeric sweeps for thresholds. Do not use fine resolution until the broader idea proves useful.
10. For wide numeric spaces, prefer `final_value = base_value * multiplier` over testing every integer or decimal. Example: `base_value = 1000` and `multiplier = 1..10` gives a controlled `1000..10000` sweep without wasting epochs on every intermediate value.
11. For smaller Hyperopt windows, run two or three random states. Keep ideas or parameter regions that repeat across states; treat one-off winners with caution until they repeat or validate cleanly.
12. For multi-branch Sieve searches, prefer explicit null/sentinel states or a separate focused file over Boolean gates plus many hidden value parameters. If a threshold is ordered, keep it numeric rather than encoding every value as a category.
13. Some branch-local parameters are unavoidable when a file intentionally tests several related concepts. Keep those branch sets coherent, and avoid mixing unrelated branches that make most parameters inactive most of the time.
14. When a file contains several meaningful branch modes, increase exploration budget rather than shrinking the search into backtests. Use more epochs and multiple random states/seeds so the optimizer has a real chance to sample each mode. If the runner or sampler exposes an initial random/startup candidate setting, set it above default for these broad categorical files; do not add unsupported command flags or workaround code.
15. Epoch count should scale with branch complexity. A file with several categorical branch modes, target-action modes, stop-action modes, and profit-gate modes needs materially more epochs than a narrow one-mode file. Do not run tiny epoch counts and then judge branch quality from under-sampled modes.

## FreqAI / Orderbook / Context Boundary

1. FreqAI, orderbook, news/context, and confluence features are parked as optional future overlays.
2. Do not create non-Sieve Hyperopt packages around those features without explicit user approval.
3. If Sieve later consumes those features, keep the feature source explicit and treat it as a Sieve input, not a separate competing research lane.

## Result Summary Required Fields

Each Hyperopt summary should include:

1. Sieve batch/search file path.
2. Config path.
3. Timerange or train/holdout windows.
4. Hyperopt result path.
5. Loss/objective score.
6. Top four candidate modes.
7. Return, drawdown, trade count, win rate, and profit factor where available.
8. Long/short split when relevant.
9. Entry tag split when available.
10. Candidate classification: `hyperopt_promising`, `needs_holdout`, `propose_backtest`, `park`, or `reject`.
11. Exposure assumptions.
12. Time/method context: method used, timerange, timeframe, trading mode, and whether the run was training, holdout, safety, or validation.

## Backtest Boundary

Agents should bring the user the strongest Sieve Hyperopt candidates and ask which candidates should be backtested unless the user has already authorized validation backtests for the task.

## Promotion Parameter Preservation

When promoting a Sieve strategy, preserve the exact selected Hyperopt parameters that justified its promotion.
