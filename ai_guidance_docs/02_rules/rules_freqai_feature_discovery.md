---
doc_status: active
default_read: routed
owner: user+agent
purpose: Rules for loose/early FreqAI feature exploration.
do_not_use_for: Promoting strategies or final production decisions.
last_rebuilt: 2026-06-10
---

# Rules - FreqAI Feature Discovery

## Purpose

Use this for early feature discovery. The standard is intentionally lighter than FreqAI promotion or strategy acceptance.

If the user says to use FreqAI, use FreqAI. Do not substitute custom scripts for FreqAI theory testing, candidate filtering, or ML inference. If the current FreqAI setup does not fit the requested data or theory, create or update a dedicated FreqAI environment/profile for that request.


## Runtime/snapshot requirements

1. Use validated snapshots or exported parquet caches for repeatable feature discovery.
2. Do not run feature discovery directly against live SQLite collectors unless the task is explicitly source-readiness validation.
3. If a source requires collector pause/export, follow `rules_runtime_environment.md` before testing.

## Horizon selection

For live media/news/web/global feature discovery, include fast targets by default: `1h`, `2h`, and `4h`, alongside slower `6h` and `24h` targets when data volume allows. Live media can move markets within minutes to a few hours, so do not rely only on `6h`/`24h` labels unless the user explicitly asks for slower path analysis.

## Acceptable early evidence

A feature or feature family may be kept as a research lead if it:

1. maps to a trader-readable state,
2. is timestamp-safe in the tested window,
3. has enough non-null/active rows for exploration,
4. shows coherent direction or ranking in at least one relevant target/window,
5. does not obviously duplicate the answer label,
6. is recorded as exploratory, not promoted.

## Not acceptable

1. Broad “run everything and see what happens” without source groups and targets.
2. Treating raw AUC/correlation as a trading edge.
3. Treating a single good window as stable proof.
4. Treating missing source rows as zero signal.
5. Allowing debug/source-availability columns as predictive features unless explicitly testing missingness.

## Exploratory metric handling

Early feature discovery may use loose metrics, but the interpretation must stay modest:

1. AUC/correlation/AP/top-bucket lift can mark a feature family as interesting.
2. A single metric cannot make a feature strategy-ready.
3. For rare events, prefer AP and top-bucket enrichment over broad accuracy claims.
4. For path targets, prefer oriented top-bucket lift and Spearman direction over raw correlation alone.
5. Any promising exploratory metric must move to direct tests before FreqAI promotion or strategy integration.

## Required output

Record exploratory results in a concise result file or append to `../04_results/hypothesis_ledger.csv` with status such as:

- `idea`,
- `exploratory_signal`,
- `needs_direct_test`,
- `deferred_for_data`,
- `rejected_for_now`.

## Escalation

Before FreqAI promotion, move to:

- `rules_direct_tests.md`, then
- `rules_freqai_promotion.md`.
