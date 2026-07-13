---
doc_status: active
default_read: routed
owner: user+agent
purpose: Strategy/candidate success, acceptance, and promotion rules.
do_not_use_for: Loose feature discovery or dry-run startup smoke checks.
last_rebuilt: 2026-06-11
---

# Rules - Strategy Success

## Core Rule

A profitable backtest is not automatically a production candidate. A strategy candidate must be judged against its own purpose, anchor, risk, robustness, and evidence path.

## Strategy Comparison Rules

1. Compare a variant against the correct Sieve baseline/search family, not a different historical branch.
2. Highest return alone is not enough.
3. Preserve multiple useful roles: max return, balanced risk, cleaner high-PF, sparse high-quality, orderbook-risk overlay, position-management.
4. Do not erase a useful role just because another candidate has higher return.
5. Do not merge candidates until overlap, conflict, and combined drawdown are understood.

## Required Strategy Metrics

Report where available:

1. return,
2. max drawdown,
3. max underwater / intra-trade risk,
4. profit factor,
5. win rate,
6. trade count,
7. trades per year,
8. long/short trade count and ratio,
9. same-window buy-and-hold / market-change comparison,
10. year/month stability,
11. worst month/family/side clusters,
12. evidence path to Sieve result, Hyperopt result, backtest report, ZIP, or CSV.

## Acceptance Guardrails

1. Any `20%+` drawdown result is research-only unless the user explicitly accepts that risk profile.
2. Promoted candidates should beat same-window buy-and-hold/market-change.
3. Avoid strategies more one-sided than `80/20` long-to-short or short-to-long unless explicitly documented as a specialist.
4. Several tens of entries per year are normally expected unless the candidate is explicitly sparse/high-quality.
5. Broad exits should normally be rejected unless they represent a true market-wide crash/risk-off/reversal state.
6. Entry-family-specific exits are the default design target.

## Evidence Path Rule

Every promoted Sieve candidate should have one of:

1. a Sieve result path,
2. a Hyperopt result path,
3. a backtest ZIP path,
4. a promotion report path,
5. a summary CSV/Markdown report path,
6. `missing_evidence_path` with a note explaining what needs to be recovered.

No active candidate should rely only on a copied metric without an evidence path or missing-evidence marker.

## Verdicts

Use one of:

- `promote_for_next_stage`,
- `keep_as_active_candidate`,
- `research_candidate`,
- `park_for_rework`,
- `reject_current_form`,
- `missing_evidence_path`.

## Research Interpretation

Do not require every useful variant to become the new main baseline. Classify results by role:

1. `replacement_candidate`: improves the candidate's purpose without unacceptable drawdown/trade-count damage.
2. `risk_candidate`: materially reduces drawdown or underwater risk while preserving most of the return.
3. `sparse_specialist`: improves quality/win rate/PF but sharply reduces trade count.
4. `rework_candidate`: gives a useful clue but fails one key metric.
5. `parked`: logical idea tested enough for now and not worth more cycles.
6. `rejected_for_now`: harms core metrics or fails controls without a specific repair path.

Prefer improvements that increase or maintain trade count while improving win rate/risk. If trade count falls, the report must explain whether the result is a valid sparse specialist or too narrow for the active objective.
