---
doc_status: active
default_read: routed
owner: user+agent
purpose: Bounded goal-mode iteration rules for overnight/high-output agents.
do_not_use_for: Replacing strategy success rules, source readiness rules, or coding workflow review.
last_rebuilt: 2026-09-03
---

# Rules - Goal-Mode Iteration Control

## Purpose

Use these rules when an agent is asked to run for a long stretch and produce many useful strategy/research results without wasting context, tokens, or compute on open-ended tweaking.

## Core principle

A goal-mode agent should behave like a bounded research controller:

1. define the baseline,
2. test one logical change at a time,
3. compare against the correct baseline,
4. accept, revise, park, reject, or move on,
5. write compact ledgers and reports.

Do not chase every possible tweak.

## Permission-gated continuity

1. A missing permission blocks only the exact edit, launch, scope expansion, or other
   action that requires it.
2. Continue useful read-only investigation, coverage checks, test design, independent
   authorized branches, or other safe work that advances the active objective.
3. In every user update, list each still-required permission as `Permission 1`,
   `Permission 2`, and so on, with the exact affected files or action and intended
   behaviour.
4. Keep the numbering stable until the user approves, rejects, or makes the request
   unnecessary. State when a numbered request is closed.
5. Do not infer permission from silence and do not use parallel work to broaden the
   objective or bypass a required approval.
6. If no meaningful safe path remains, report the numbered permission blocker rather
   than generating low-value work merely to appear active.

## Default caps

Unless the user gives different limits:

1. Maximum `4` iterations per concept.
2. Maximum `2` variants per iteration.
3. Maximum `24` total new strategy variants for the whole run.
4. Maximum `1` repair attempt after a broad exit damages results badly.
5. Maximum `1` threshold relaxation attempt after an entry filter destroys trade count.
6. Do not combine more than `2` accepted changes at once without a specific combination hypothesis.

## Required loop per concept

For each concept/lane:

1. State the trader hypothesis in plain English.
2. Identify the baseline strategy/class/window.
3. Identify the exact failure mode or improvement target.
4. Make the smallest coherent change.
5. Run the test.
6. Record metrics and artifact path.
7. Decide: `accept`, `keep_as_risk_lane`, `revise_once`, `park`, `reject`, or `blocked`.
8. If the concept fails after logical repairs, move to the next concept.

## Evidence-Triggered One-Minute Replay Cap

When Objective 02b routes a repeated reaction pattern to targeted `1m` replay:

1. candidate capture is a cheap queueing action, not permission to launch a run;
2. freeze the direction-neutral selection rule before inspecting signed `1m` paths;
3. normally use one diagnostic batch of `6-12` independent episodes, treating that
   range as a work cap rather than an evidence threshold;
4. run no more than one replay batch and one worker at a time while a main generation
   is unfinished, unless a live capacity check proves the main work and the user's
   processor reserve are unaffected;
5. perform one direct aligned-path pass before any FreqAI run;
6. permit at most one frozen FreqAI ladder for that queued pattern before returning to
   the main plan;
7. do not let the replay spawn another test, change the main batch order, or tune a
   parent indicator before the complete generation review; and
8. park the lane when `1m`, orderbook, trade-flow, or external-context coverage cannot
   support the declared question without guessing.

## Objective 02b Breadth-First Queue And Success Targets

For Objective 02b reaction-and-event-driven direction work:

1. Finish every frozen main batch before a result redirects the generation.
2. Capture ideas from interim or final results in the rolling branch queue instead of
   launching them immediately. Valid ideas include new indicators, mathematics,
   timeframes, single-level or cluster definitions, convergence patterns, coins or
   cohorts, controls, and timestamp-safe contextual sources.
3. Record the source result, plain-language observation, route family, exact
   hypothesis, alternative explanation/control, required data, smallest useful test,
   cost, earliest eligible generation, and status.
4. At the complete generation review, confirm provisional ideas, deduplicate cosmetic
   variants, freeze one balanced next batch, and send its descendants to the following
   review.
5. Assess at least five materially different route families while the objective is
   unresolved and eventually give at least five a fair terminal test. This is not a
   requirement to consume all five authorized branch layers or invent filler tests.
6. For meme-coin work, use only the frozen `10` most traded eligible meme coins under
   the ranking and coverage contract in the active objective.
7. For issued reaction-and-direction calls, score reaction success, conditional
   direction success, and joint success separately. Joint success requires both on the
   same call.
8. Treat `55%` unseen joint success as the minimum acceptable research-lead floor and
   `65%` as the main target, while also requiring meaningful coverage, adequate
   independent support, and improvement over base-rate, majority-direction,
   simple-trend, and no-level controls.
9. Do not treat either percentage as a native FreqAI threshold, profit criterion, or
   automatic trading promotion.
10. Run the event hierarchy in order: honest historical event reconstruction,
    individual-link tests, frozen pairwise tests, limited three-block chains and
    post-event ranges, then untouched whole-event or later live confirmation.
11. Complete every frozen sibling in one layer before promoting leads to the next;
    interim ideas go to the branch queue and do not redirect the active layer.
12. Keep every observation from one event in the same development or holdout
    partition, and count whole events rather than candles as independent confirmation.
13. Permit an explicit abstention when evidence conflicts or is weak. Report issued
    calls and abstentions separately so selectivity cannot hide poor general coverage.
14. Test BTC, ETH, and broad-market leadership instead of assuming BTC always leads,
    and keep scheduled and unexpected events as separate evidence groups.

## Main decision table

| Result shows | Required action |
|---|---|
| Variant improves return with no worse drawdown/trade-count damage | Keep as candidate and log evidence. |
| Variant materially reduces drawdown while preserving most return | Keep as risk lane; do not force it to replace max-return baseline. |
| Variant improves win rate/profit factor but sharply reduces trade count | Classify as sparse/specialist unless user asked for sparse quality. |
| Exit improves drawdown but kills return | Retest once as partial exit or stop tightening instead of full exit. |
| Exit hurts return, drawdown, win rate, and PF | Park the exit idea. |
| Broad exit cuts winners too early | Convert to stop tightening/reduce-exposure only when trade is profitable or near target; retest once. |
| Entry filter raises win rate but destroys trade count | Relax threshold once or classify as sparse specialist. |
| Entry filter lowers win rate or PF | Park it. |
| Same-direction stacked signals are strong but whole strategy worsens | Try selective add/hold only on strong families; do not reserve stake globally. |
| Opposite-direction signal appears during open trade | Test as reduce/tighten/exit evidence, not automatically as reverse entry. |
| Crash/risk signal exits too early | Change to tighten stop/reduce exposure only when profitable or near target; retest once. |
| Variant improves one lane but worsens another | Keep lane-specific; do not merge blindly. |
| Two individually good variants combine poorly | Keep them separate and record conflict. |
| Four iterations fail to produce useful evidence | Park concept and move on. |
| Non-BTC test is poor | Mark not portable or needs coin-specific rework; do not deep optimise overnight. |
| Required baseline cannot reproduce | Mark blocked and move to next viable lane. |

## Metrics to record for strategy variants

Record at minimum:

1. strategy class,
2. coin/pair,
3. timerange,
4. baseline strategy and baseline timerange,
5. trades,
6. return percentage,
7. max drawdown percentage,
8. profit factor,
9. win rate,
10. long/short ratio where available,
11. same-window market change / buy-and-hold comparison where available,
12. backtest ZIP/report path,
13. verdict,
14. next action or park reason.

## Trade-count rule

The active broad research preference is to improve win rate, risk, and return while maintaining or increasing trade count where practical. A lower-trade variant may still be useful, but it must be classified accurately:

1. `replacement_candidate` only if it beats the lane purpose without unacceptable trade-count loss.
2. `risk_lane` if it reduces drawdown materially while preserving enough return.
3. `sparse_specialist` if quality improves but trade count drops sharply.
4. `parked` if the trade-count loss makes the result too narrow for the user-approved research scope.

## Combination rule

Do not stack changes just because each worked alone. A combination needs its own hypothesis:

1. what conflict/overlap is expected,
2. which lane it is meant to improve,
3. what metric tradeoff is acceptable,
4. when to keep the changes separate.

## Multi-coin rule

For overnight runs, non-BTC work is portability classification first, not optimisation:

1. Run current lane/baseline if data exists.
2. Compare basic metrics and market-change context.
3. Classify the lane/coin.
4. Do not start broad coin-specific tuning unless the objective changes.

## Reporting rule

Detailed results belong in files, not chat. The final chat response should list:

1. dry-run startup status,
2. number of baselines reproduced,
3. number of variants tested,
4. accepted candidates,
5. parked/rejected ideas,
6. blocked items,
7. updated ledger/report paths.
