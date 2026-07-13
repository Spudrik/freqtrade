---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for rare pattern-indicator families and their separate research requirements.
do_not_use_for: Generic pattern scoring, routine Sieve result review, or moving active Freqtrade strategies out of their required directory.
last_rebuilt: 2026-07-13
---

# Track - Pattern Indicators

## 1. Boundary

This is the logically separate lane for geometry, reversal, continuation, multi-peak, and Wolfe patterns. The strategies may remain together in Freqtrade's strategy directory for discovery, while filenames, batches, and evidence keep the research families separate.

## 2. Current state

- Pattern modules were split because each family needs different identity, confirmation, scoring, invalidation, and exit geometry.
- Geometry covers triangles, wedges, channels, and rectangles.
- Reversal and multi-peak work covers head-and-shoulders and double/triple tops/bottoms with family-specific identity and thresholds.
- Continuation and Wolfe families have separate confirmation/reaction contracts.
- Some selected pattern baselines show useful win rate, payoff, and drawdown, but many very high win rates came from only `2-11` trades.
- Sparse pattern exit Hyperopt requires wider pairs and longer/mixed windows; a tiny first training sample is a setup problem, not a result.

## 3. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| Late April-early May 2026 | Initial pattern and geometry review | Established that pattern families could not share one loose detection contract. | Git history; indicator reference. |
| 8-13 May | Families split and outputs standardized | Geometry, reversal, continuation, multi-peak, and Wolfe received separate strategy-facing outputs. | Git history. |
| 14 May | Pattern-specific filters refined | Added reversal scaling/H&S filters, dynamic multi-peak identity, and Wolfe reaction filtering. | Git history; score guide. |
| Sieve1/Sieve2 | Rare patterns tested as exact entry concepts | Some credible rows emerged, but sparse samples prevented broad promotion claims. | Sieve2 baseline lookup. |
| Sieve3 | Family-specific exit contracts required | Necklines, boundaries, measured moves, reaction points, and pattern failure must drive exit hypotheses. | Sieve3 exit/regeneration rules. |

## 4. Current conclusions

- **Demonstrated:** pattern families require separate contracts and coverage handling.
- **Promising:** selected complete triangles, double bottoms, wedges, and rectangles with adequate samples.
- **Insufficient evidence:** extreme win-rate rows supported by only a few trades.
- **Avoid:** vague aggregate labels such as one generic rare-pattern or "sunflower" family when exact pattern names exist.

## 5. Outstanding work

- Preserve exact pattern family, side, timeframe, trigger, target, and invalidation identity.
- Validate pair/window coverage before any exit-stage judgment.
- Compare pattern-specific target/failure logic rather than cloning generic exits.
- Keep a pattern candidate reviewable when its logic is sound but sample coverage is insufficient.
- Record a milestone only for a completed family redesign, adequate validation stage, or durable conclusion.

## 6. Canonical evidence

- Pattern contracts: `../../../user_data/Indicators/INDICATOR_STRATEGY_REFERENCE.md`
- Pattern scoring: `../../../user_data/Indicators/INDICATOR_SCORE_GUIDE.md`
- Sieve2 baseline lookup: `../../04_results/sieve3_sieve2_baseline_lookup.md`
- Sieve3 exit rules: `../../02_rules/rules_sieve3_exit_hyperopt.md`
- Regeneration guidance: `../../02_rules/rules_sieve3_exit_regeneration_agent.md`

