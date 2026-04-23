# Historical Orderbook Backtest Notes

This note is for future agents returning to the Bybit historical orderbook work.

## Current State

Already implemented:

- Bybit historical raw archive download support via `freqtrade download-data --dl-orderbook`
- daily archive availability scanning/reporting before download
- raw ZIP archive storage under a dedicated orderbook datadir
- `freqtrade orderbook-to-features` conversion from raw Bybit `ob500` archives into aggregated feature files
- launcher UI support in `Order Book Lab -> Bybit History`

This is enough to:

- see what archive dates are available
- download raw Bybit orderbook archives
- generate 1h-oriented feature files such as spread, mid, depth, imbalance, and microprice aggregates

## Actual Objective

Make the generated historical orderbook feature data useful inside normal Freqtrade backtesting and hyperopt workflows.

The intended outcome is not a full event-driven L2 simulator.

The intended outcome is:

- strategies can load precomputed orderbook feature files
- those features can be merged into the main dataframe without lookahead bias
- backtesting and hyperopt can use those feature columns on `1h` workflows the same way they use other informative data

## What Still Needs To Be Done

### 1. Add a proper loader path for stored orderbook feature files

There is already storage/merge support, but no normal strategy-facing data loading path yet.

Need a clean way to load:

- datadir
- pair
- timeframe
- category/depth/format

Likely touch points:

- `freqtrade/data/dataprovider.py`
- possibly `freqtrade/data/history/*`
- possibly a small strategy helper or dp helper so strategies do not need to know file paths

### 2. Define the strategy API surface

Decide how strategies should request orderbook features.

Preferred direction:

- keep it similar to informative data usage
- avoid making strategies manually open files
- allow a strategy to say, in effect, "give me the orderbook features for this pair/timeframe"

The existing helper `merge_orderbook_features(...)` should remain the merge primitive, but strategies still need a clean loader source.

### 3. Preserve time alignment and no-lookahead behavior

This is the most important correctness point.

When feature rows are merged into a candle dataframe:

- timestamps must behave like closed information only
- no future orderbook information should leak into the current candle
- missing feature rows should have deliberate handling, not accidental forward-fill everywhere

The merge behavior should be verified with focused tests.

### 4. Decide missing-data behavior

Need a clear rule for:

- missing feature files
- partial archive coverage
- gaps inside requested timeranges
- whether to forward-fill, leave NaN, or reject runs when coverage is too thin

Backtesting and hyperopt should fail clearly when coverage is unusable, not silently produce misleading results.

### 5. Add tests around end-to-end usage

Needed tests:

- feature file loading for a pair/timeframe
- merge alignment against a candle dataframe
- no-lookahead behavior
- behavior when feature files are absent
- behavior when only some timerange coverage exists

### 6. Optional launcher follow-up

The launcher already exposes archive download and conversion.

If more UI work is done later, it should be limited to:

- showing feature-file presence/coverage more clearly
- maybe previewing feature columns

Do not spend time on more launcher work until the backtest integration path is actually usable.

## Non-Goal

Do not turn this into a full historical orderbook execution simulator unless explicitly requested later.

That would require:

- event-driven execution logic
- queue position modeling
- slippage/partial fill simulation on L2
- a very different backtest engine surface

That is a separate project.

## Recommended Next Step

When returning to this work, start by wiring stored orderbook feature files into a strategy-accessible loader path, then prove the merge works on a `1h` backtest dataframe with one small sample strategy/test.

## Acceptance Criteria

This work is useful when all of the following are true:

- a strategy can access historical orderbook feature data without manual file reads
- a `1h` backtest can use those columns in indicators/entry-exit logic
- hyperopt can optimize against those columns
- merge timing is verified to avoid lookahead
- missing coverage is surfaced clearly
