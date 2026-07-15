# Strategy Taxonomy Inventory

Scope: all `IStrategy` Python modules under `C:\FreqTradeStuff\user_data\strategies`, including the active top level, `sieve1_to_guards`, `sieve2_complete_patterns`, `sieve3_candidates`, and `Archive`. The `sieve1`, `sieve2_originals_failed`, and `failed_strategies` buckets now live under `Archive`.

Counting rule: this is a filename-led taxonomy with a supporting import scan. A single file can belong to multiple groups, so subgroup counts overlap by design.

## 1. Population

Total strategy modules found: `1300`

### By subtree

| Subtree | Count | Notes |
| --- | ---: | --- |
| `<root>` | 220 | Active top-level strategy modules |
| `sieve1_to_guards` | 144 | Relative-strength / guard conversions |
| `sieve2_complete_patterns` | 22 | Parked complete-pattern candidates |
| `sieve3_candidates` | 64 | Accepted / candidate clones |
| `Archive` | 850 | Archived strategy buckets, including `sieve1`, `sieve2_originals_failed`, and `failed_strategies` |

### Research-only wrappers

These are strategy modules, but they are not entry-first pattern files:

| File | Note |
| --- | --- |
| `ContextFreqAIResearchStrategy.py` | Research-only FreqAI wrapper |
| `TraderRuleBlockResearchStrategy.py` | Research-only signal playback wrapper |
| `Archive/pivot_trendline_mtf_20260514/PivotTrendlineMTFResearchStrategy.py` | Archived research strategy |

## 2. Strategy-Facing Indicator Coverage

This is the clearest view of what the entry corpus has already exercised.

| Indicator module | Files using it | What it mainly covers |
| --- | ---: | --- |
| `complex_volume_profile.py` | 728 | POC, VAH/VAL, HVN/LVN, node, traverse, context |
| `complex_trendline_projection_v2.py` | 503 | TLV2 support/resistance lines and distance logic |
| `pattern_bos_choch.py` | 503 | BOS/CHoCH structure and continuation/reversal context |
| `pattern_geometry_v2.py` | 277 | Triangle, wedge, rectangle, flag, pennant, channel, compression |
| `pattern_continuation.py` | 121 | Flag and pennant continuation |
| `pattern_reversal.py` | 138 | Double top/bottom and head-and-shoulders families |
| `pattern_multi_peak.py` | 45 | Triple top/bottom families |
| `pattern_wolfe_waves.py` | 88 | Wolfe wave long/short setups |
| `complex_relative_strength.py` | 40 | Relative-strength context and guard-style usage |

Note: `pivot_foundation.py` is a foundation layer, not a direct entry family.

The remaining family tables below are filename-tag counts unless a row says otherwise.

## 3. Pattern Family Breakdown

These are filename tags, not a partition. A file can appear in more than one subgroup.

| Subgroup | Count | Typical names |
| --- | ---: | --- |
| Triangle | 35 | `...triangle...` |
| Wedge | 31 | `...wedge...` |
| Rectangle | 37 | `...rectangle...` |
| Flag | 48 | `...flag...` |
| Pennant | 30 | `...pennant...` |
| Ascending channel | 26 | `...ascending_channel...` |
| Descending channel | 23 | `...descending_channel...` |
| Compression | 22 | `...compression...` |
| Double bottom | 19 | `...double_bottom...` |
| Double top | 16 | `...double_top...` |
| Triple bottom | 27 | `...triple_bottom...` |
| Triple top | 22 | `...triple_top...` |
| Head and shoulders | 49 | `...head_shoulders...` or `...hs...` |
| Inverse head and shoulders | 18 | `...inverse_head_shoulders...` or `...ihs...` |
| Wolfe | 39 | `...wolfe...` |
| Ladder | 22 | `...ladder...` |
| Pivot | 18 | `...pivot...` |

What this says in plain terms:

- Triangles, wedges, rectangles, and flags are the densest pure shape families.
- Channels and head-and-shoulders variants are also heavily explored.
- Compression exists, but it is smaller than triangle/wedge/rectangle work.

## 4. Volume Profile Family

Volume profile is the largest structural family in the corpus.

| Subgroup | Count | Notes |
| --- | ---: | --- |
| VP generic | 473 | Any file with `vp` in the name |
| `node` | 129 | Node-oriented VP acceptance / hold / entry work |
| `VAL` | 70 | Value-area-low reclaim / reject style logic |
| `VAH` | 48 | Value-area-high reject / reclaim style logic |
| `LVN` | 23 | Low-volume node traversal / acceptance |
| `POC` | 17 | Point-of-control reclaim / reject |
| `HVN` | 14 | High-volume node reclaim / reject |
| `fast_traverse` | 14 | Fast traversal behavior through weak volume areas |
| `market_guard` | 55 | VP used as a guard overlay |
| `capitulation` | 6 | VP plus capitulation-style impulse behavior |

The dominant VP pattern is node/value-area work, not just a single POC line.

## 5. TLV2 Family

TLV2 is also large and tends to be used as a line-based execution framework.

| Subgroup | Count | Notes |
| --- | ---: | --- |
| TLV2 generic | 337 | Any file with `tlv2` in the name |
| `reclaim` | 74 | Reclaim after break / failure |
| `ride` | 62 | Trendline ride / continuation |
| `retest` | 49 | Retest after break |
| `support` | 44 | Support-side line logic |
| `proximity` | 41 | Proximity or near-line logic |
| `resistance` | 30 | Resistance-side line logic |
| `breakout` | 25 | Breakout through line |
| `breakdown` | 22 | Breakdown through line |

In plain English: TLV2 is used more as break/reclaim/retest mechanics than as a pure line-painting experiment.

## 6. Other Structural Families

| Family | Count | What it is doing |
| --- | ---: | --- |
| BOS/CHoCH | 136 | Market-structure break and change-of-character logic |
| Multi-concept `multiN` | 328 | Mixed confluence files with two or more named concepts |
| Relative strength | 40 | Mostly guard/context, not primary entries |
| AVWAP | 24 | Anchored VWAP reclaim/reject families |
| Liquidity | 17 | Prior-high/low sweeps, equal highs/lows, reclaim/reject |
| Prior-level | 105 | Prior day/week/month/range high/low logic |
| Supply zone | 9 | Supply-breakout / supply-reject variants |
| Demand zone | 9 | Demand-reclaim / demand-breakdown variants |
| Ladder | 22 | Ladder-style support / resistance hold work |
| Capitulation | 6 | Sharp move + volume impulse ideas |
| Regime pullback | 2 | Broad market-state pullback context |
| Volatility breakout | 3 | Simple breakout-through-range ideas |

## 7. Timeframe Architecture

Normalized from filename tokens such as `1d`, `d1`, `4h`, `h4`, `8h`, and `1h`.

| Architecture | Count | Notes |
| --- | ---: | --- |
| `1h` | 268 | Single-timeframe 1h files |
| `4h` | 198 | Single-timeframe 4h files |
| `8h` | 213 | Single-timeframe 8h files |
| `1d` | 190 | Single-timeframe 1d files |
| `4h+1h` | 140 | Higher-timeframe context with 1h execution |
| `1d+4h` | 71 | Higher-timeframe context with 4h execution |
| `1d+1h` | 23 | Mixed 1d/1h naming without a 4h leg |
| `1d+4h+1h` | 8 | Full MTF chain |
| `no explicit TF token` | 189 | Generic or wrapper-style names |

## 8. Overlapping Groups That Matter

These are the mixed families that show the strategy corpus is not just single-concept probes.

| Combination | Count | Example shape |
| --- | ---: | --- |
| `multi + tlv2 + vp` | 179 | The dominant mixed family |
| `vp only` | 119 | Pure VP names with no other core tag |
| `mtf only` | 100 | MTF architecture without another strong core tag |
| `multi + prior_level + vp` | 56 | Prior-level plus VP confluence |
| `mtf + vp` | 43 | Timeframe plus VP execution |
| `mtf + pattern` | 41 | Timeframe plus pattern execution |
| `tlv2 + vp` | 22 | Line plus profile confluence |
| `avwap + mtf` | 7 | Anchored VWAP with MTF execution |
| `pattern + vp` | 5 | Pure shape plus profile overlap |
| `liquidity + mtf + prior_level` | 5 | Sweep / reclaim against a higher-timeframe level |

The main concentration is clear: mixed `VP + TLV2 + multi-concept` work dominates the deeper confluence set.

## 9. What Looks Underexplored

Based on the counts above, the biggest gaps are not in the heavy hitters.

1. `complex_relative_strength.py` is present, but it is still mostly guard/context work. There is no big standalone RS entry family yet.
2. `regime_pullback` only shows up in 2 files. That is barely explored as a direct entry source.
3. `volatility_breakout` shows up in 3 files. That is another thin lane.
4. `compression` exists, but it is smaller than triangle/wedge/rectangle/channel work.
5. `liquidity` and `capitulation` are present, but they are thin compared with VP and TLV2.
6. The parked external-context modules under `C:\FreqTradeStuff\user_data\Indicators\work_in_progress\` are intentionally not active entry inputs yet.

## 10. Archive Note

Failed Sieve hyperopts are archived. Successful strategies are archived too, and the winning concept is cloned up into the next refinement level for further work before the archived copy is parked.
