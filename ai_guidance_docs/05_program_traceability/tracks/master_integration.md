---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for entry/exit evidence weighting, overlap control, trade state, and final master-strategy integration.
do_not_use_for: Current implementation authority, inventing promotion thresholds, or replacing subsystem evidence and rules.
last_rebuilt: 2026-07-13
---

# Track - Master Strategy Integration

## 1. Boundary

This track owns the cross-system architecture and integration milestones. Sieve owns deterministic candidate evidence; indicator tracks own signal production; data tracks own source availability; FreqAI owns dynamic risk hypotheses.

## 2. Current state

- The end-state architecture is defined conceptually but is not assembled.
- Entry-family evidence, exit-role selection, overlap penalties, and dynamic risk weights are not yet consolidated into a canonical registry.
- The current priority is to settle Sieve3 exit design before assigning final entry/exit weights.
- Same-direction additions have evidence, but structural price-zone staggering needs a controlled test.
- FreqAI is not yet ready to provide a promotion-grade risk multiplier.

## 3. Intended architecture

### Static entry evidence

For each exact entry family and side, retain trigger/story, target/invalidation providers, trade/sample coverage, target-before-invalidation reliability, payoff, drawdown, stability across windows/seeds/pairs, overlap, and data dependencies.

The prior weight should reward adequate evidence, payoff quality, controlled risk, robustness, and unique information. It should penalize small samples, concentration, instability, drawdown, correlated duplicates, and unavailable sources. Raw win rate alone is insufficient.

### Candle-level combination

1. evaluate eligible exact entry families;
2. group or de-duplicate correlated variants;
3. combine side-aware static evidence;
4. retain opposing entry/guard evidence explicitly; and
5. pass the deterministic state to the bounded risk layer.

### Dynamic risk and trade state

FreqAI should provide named, bounded risk/opportunity estimates. The trade-state engine should retain source entry, trigger health, guards, target/invalidation providers, filled tranches, completed partials, current risk multiplier, and opposing evidence so actions are idempotent and family-specific.

Position adjustment should be first-class, not compulsory on every trade. Leverage follows only after entry, add, reduce, stop, and exit behaviour is stable.

## 4. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| June 2026 | Multi-entry/confluence architecture clarified | Same-direction evidence may support add/hold; opposing evidence may support reduce/tighten/exit. | Master objective and entry/exit rules. |
| 5-9 June | Position-management experiments | Strong adjusted subsets did not guarantee better whole-strategy capital use. | Sieve track and historical progress. |
| 11 June | Sieve-first separation adopted | Deterministic discovery, data readiness, FreqAI risk, and final integration became separate lanes. | Current objective. |
| July | Entry/exit traceability and weighting model clarified | Final weights should separate entry edge/confidence, exit utility, later regime suitability, and overlap penalty. | Programme discussion; Sieve track. |

## 5. Outstanding work

### Prerequisites

- Settle the refined Sieve3 exit framework.
- Build the compact retained entry/exit evidence view.
- Establish adequate validation for rare pattern candidates.
- Complete the controlled price-zone staggering comparison.
- Accumulate clean data blocks and revalidate only strong FreqAI risk hypotheses.

### Integration sequence

1. Create the exact entry-family registry from survivor evidence.
2. Define evidence components without inventing a premature opaque score.
3. Measure overlap and marginal contribution.
4. Retain genuinely distinct primary, defensive, and asymmetric exit roles where supported.
5. Add stateful tranche/partial/action handling.
6. Add bounded FreqAI risk modifiers behind exact source eligibility.
7. Run component ablations and full-system validation before any dry-run freeze.

## 6. Canonical evidence

- Master objective: `../../01_objectives/objective_02_master_strategy_architecture.md`
- Entry/exit/position rules: `../../02_rules/rules_entry_exit_position.md`
- Sieve traceability: `sieve_entry_exit.md`
- FreqAI traceability: `freqai_risk_weighting.md`
- Historical/live data traceability: `data_historical.md` and `data_live.md`

