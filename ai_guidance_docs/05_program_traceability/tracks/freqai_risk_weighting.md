---
doc_status: active
default_read: no
owner: user+agent
purpose: Milestone traceability for FreqAI feature research and the future dynamic risk/weighting layer.
do_not_use_for: Active FreqAI authority, raw model reports, data-ingestion status, or deterministic Sieve promotion.
last_rebuilt: 2026-07-13
---

# Track - FreqAI Risk and Weighting

## 1. Boundary

This track owns trader-readable FreqAI hypotheses, controls, model conclusions, and the planned dynamic risk/weighting role. Historical/live source acquisition belongs to the data tracks. Deterministic entry/exit discovery remains in Sieve.

FreqAI is parked by the current objective and must not become active trading logic until the user explicitly reopens it.

## 2. Intended role

FreqAI should estimate named conditional risks or opportunities—such as follow-through, drawdown, breakdown continuation, absorption, volatility expansion, or source-specific invalidation—and apply a bounded modifier to deterministic entry/exit evidence.

It should influence enter/skip, size, later leverage, add permission, tighten/reduce, partial, runner, or exit decisions. It should not replace the trader-readable entry family with an opaque prediction.

## 3. Current state

- Event-scoped price/structure labels have generally been more stable than generic next-return labels.
- Orderbook wall distance, spread fragility, support removal, and failure states look more useful as precise caution/invalidation evidence than broad entry filters.
- Live media/context produced several useful AUC lifts and cross-pair shapes, but the temporal history remains short.
- Strict live media plus orderbook overlap can leave only tens or hundreds of rows.
- Historical aggregate GDELT leads are exploratory and are not equivalent to article/story-aware live media.
- Earlier false-clean coverage, timestamp, taxonomy, eligibility, and ablation issues invalidate unqualified reuse of some old results.

## 4. Milestones

| Date/period | Milestone | Conclusion/change | Evidence |
|---|---|---|---|
| 23-25 May 2026 | Initial orderbook/FreqAI runs | Feature counts were large relative to training rows; stable model lift was not established. | `../../04_results/context_research_ledger.md`. |
| 27-28 May | Event labels and larger queues | Breakout success/failure shapes were stronger than generic returns; context/orderbook lift was window-specific. | Context ledger and detailed reports. |
| 29 May | Data/model audit | Exposed zero-fill, timestamp-mask, taxonomy, eligibility, and walk-forward weaknesses; some reporting/mask issues were repaired. | Review docs linked by the context ledger. |
| 31 May-5 June | Confluence and branch queues | Some breakout/context combinations helped, breakdown/downside branches were unstable, and NaN-heavy source coverage caused real failures. | Context ledger. |
| 15 June | Historical aggregate GDELT leads | Produced relief, breakdown, attention/compression, breakout, and absorption hypotheses; exploratory only. | Recent summary and context ledger. |
| 26-28 June | True live-media and cross-pair tests | Several follow-through, topic-risk, activity, and trader-shape labels improved controls, but only across a short live window. | Recent summary and live reports. |

## 5. Current conclusions

### Promising

- Trader-readable event labels with price-only controls.
- Live topic-risk, media-activity, and source-burst/volume shapes.
- Orderbook as exact invalidation/caution evidence.
- Conditional rather than universal use of VP/context features.

### Weak or unstable

- Generic next-return targets and broad context-only/fakeout labels.
- Broad orderbook filters and generic risk-size reducers.
- Model results with many more features than effective rows.
- Conclusions derived from zero-filled, stale, or incorrectly eligible sources.

### Not yet established

- Promotion-grade temporal stability.
- A reliable combined live-media/orderbook/structure risk model.
- A calibrated dynamic multiplier suitable for the final strategy.

## 6. Outstanding work

### Before reopening

- Name a trader-readable hypothesis, action, baseline, controls, source mask, and pass/fail condition.
- Select a clean historical or accumulated live block with explicit freshness and gaps.
- Keep deterministic Sieve entries/exits frozen while testing incremental risk information.

### Future tests

- Revalidate only the strongest labels across time and pairs.
- Compare price-only, deterministic-context, and FreqAI-enhanced controls.
- Separate static entry confidence from the dynamic FreqAI risk multiplier.
- Calibrate bounded outputs for enter/skip, size, add, tighten, reduce, partial, and exit actions.
- Perform survivor-only regime analysis after entry/exit candidates have passed initial screening.

## 7. Canonical evidence

- Current parked status: `../../00_project_control/objective_current.md`
- Feature-discovery rules: `../../02_rules/rules_freqai_feature_discovery.md`
- Promotion rules: `../../02_rules/rules_freqai_promotion.md`
- Recent summary: `../../04_results/results_recent_summary.md`
- Context/FreqAI ledger: `../../04_results/context_research_ledger.md`
- Source readiness: `../../03_status/status_source_readiness.md`
- Data tracks: `data_historical.md` and `data_live.md`

