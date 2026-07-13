# Readiness-Gated Feature Research Goal

Use this when an agent is asked to use goal mode to find better feature definitions for FreqAI/direct-test research while some data sources are incomplete.

## Goal

Find useful trader-readable feature definitions using only data that is actually ready now.

The agent must not chase unavailable data, pretend partial data is full confluence, or spend time testing news/context until that separate rework is explicitly marked ready.

## First Principle

The question is not:

"Can we test every source?"

The question is:

"What can we honestly test with the validated data we have right now?"

## Stage 1: Readiness Audit

Before building or testing features, inspect current source coverage and decide what is usable.

1. Price/OHLCV:
   - Usually usable.
   - Confirm the date range and 1h alignment.
2. Custom indicators:
   - Use only where structural cache coverage exists.
   - Split source-detail groups: VP, TLV2, BOS/CHoCH, pattern geometry, cached price/volume states.
3. Orderbook:
   - Use only rows/windows with explicit present/coverage flags.
   - Do not carry orderbook state through archive gaps.
   - Treat spot, Bybit linear, and Bybit inverse as separate source-detail blocks unless testing a combined orderbook feature.
4. News/context:
   - Exclude for this goal unless a user or guidance doc explicitly says the news/context feature rework is ready.
   - Keep placeholder notes for later confluence only.

Output of this stage:

1. Usable source blocks now.
2. Date windows that are clean enough to test.
3. Sources to park because they are not ready.
4. Any timestamp or missing-data risks.

## Stage 2: Allowed Current Work

While news/context is not ready, focus on:

1. Price and volume behaviour.
2. Custom structure indicators.
3. Orderbook as confirmation, risk filter, or acceleration signal.
4. Trader-readable breakout, failure, support-break, downside-momentum, and exhaustion states.

Do not run "full confluence" unless a real clean overlap exists for every source included.

## Stage 3: Feature Definition Rules

Each feature idea must be described in this shape before coding:

1. Trader question:
   - Example: "After support breaks, can we tell whether the drop will keep going?"
2. What a trader sees at the decision hour:
   - Example: support breaks, volume rises, buy walls disappear.
3. Numeric inputs:
   - Exact feature groups or columns that represent those observations.
4. What happened afterwards:
   - Example: price kept dropping over the next 3h or 6h.
5. Controls:
   - Same setup without volume.
   - Same setup without orderbook.
   - Same setup without structure.
   - Similar random rows in the same regime.
   - Opposite-direction rows where relevant.
6. Pass/fail rule:
   - Define before testing.

## Stage 4: Priority Tests With Current Data

Start with these tests, using only usable source blocks:

1. Breakout keeps going:
   - Price breaks resistance.
   - Volume expands.
   - Orderbook resistance is weak, removed, or thin.
   - Ask: did price keep rising over the next 3h or 6h?
2. Breakout fails:
   - Price breaks or pokes above resistance.
   - Volume fades or orderbook resistance remains.
   - Ask: did price fall back under the level?
3. Downside momentum after first break:
   - Price has already broken support or made an initial fast drop.
   - Next 1h to 4h shows selling pressure, wall movement, volume persistence, and failed support reclaim.
   - Ask: did price keep dropping over the next 3h, 6h, or 24h?
4. Downside move slows or fails:
   - Price drops, but volume fades, buy walls rebuild, or price reclaims broken support.
   - Ask: did downside stop, bounce, or chop?
5. Technical breakout with quiet/balanced book:
   - Structure and volume align.
   - Orderbook is not extreme.
   - Ask: did technical levels alone guide price to the next level?
6. Orderbook panic state:
   - Multiple venues shift in the same direction.
   - Liquidity thins and walls move away from price.
   - Ask: did price move faster or further after the book entered this state?

## Stage 5: Promotion Rules

Direct tests come before FreqAI.

Promote a feature family to FreqAI only if:

1. Data coverage is clean for the selected window.
2. The feature is not just restating the answer label.
3. The direct test beats random and same-regime controls.
4. It has enough rows to trust.
5. It works across more than one month/window, or is explicitly marked as a small pilot.
6. The result can be explained in trader language.

Use component-only FreqAI profiles for discovery. Use trigger/flag profiles only for gating or validation.

## Stop Rules

Stop and report instead of wandering if:

1. A source has no clean coverage for the requested window.
2. The feature cannot be explained as something a trader would see.
3. The test requires news/context while news/context is not ready.
4. The result only works in one tiny window.
5. The model appears to win because it was handed the trigger flag.
6. The agent is about to create a broad "everything crossed with everything" feature set.

## Required Reporting Format

Use `result_communication_format.md`.

Every result must start with:

1. Trader question.
2. What we looked at.
3. What we asked afterwards.
4. What happened in plain English.
5. Numbers that support it.
6. Verdict.
7. Next step.

