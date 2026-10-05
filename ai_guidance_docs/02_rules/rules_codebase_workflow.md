---
doc_status: active
default_read: routed
owner: user
purpose: Codebase-aware development workflow.
do_not_use_for: Replacing task-specific objectives.
last_rebuilt: 2026-10-05
---

# Rules - Codebase Workflow

## Investigate first

Before writing code or implementation instructions:

1. Locate current entry points, helper functions, state files, metadata files, JSON save/load paths, UI fields, command-building logic, and runner integration points related to the requested change.
2. Identify exact existing functions/classes/modules that already perform part of the behaviour.
3. Prefer reading current files over relying on memory or old docs.
4. Treat previous versions and previous instructions as stale unless verified.
5. Trace mapping, configuration, and state inconsistencies to their authoritative/shared source. Avoid caller-local workarounds, rescue logic, and silent fallback behaviour by default.

## Reuse before adding

1. Reuse existing parsing, validation, scoring, state, metadata, command-building, save/load, JSON writing, and reporting helpers where possible.
2. Do not duplicate validation, scoring, backtest parsing, JSON save/load, or report-writing logic.
3. If the feature can be implemented as post-processing on existing summary data, prefer that over a parallel pipeline.

## Upstream core protection details

Follow the protected-file boundary and approval process in the repo-root `AGENTS.md`. Formatting-only edits to protected core files also require explicit approval. Agents may inspect those files and propose an exact patch, but must wait for approval before applying it. If an upstream change conflicts with the historical Bybit integration, prefer proposing extraction into a standalone `user_data/**` tool before modifying upstream code further.


## Runtime/path discipline

1. For commands, launcher work, backtests, dry-run startup, split worker lanes, or raw archive access, read `rules_runtime_environment.md` before running commands.
2. Do not add interpreter guessing or fallback Python behaviour. Missing configured executables must fail early with the missing path.
3. Do not move data, reports, configs, or backtest archives into worker venv folders.
4. Do not infer missing raw data from empty `C:\FreqTradeStuff\user_data\...\raw` folders; check the documented `D:` raw archive locations first.

## Refactor breakpoint rule

Flag a refactor breakpoint when:

1. three or more code paths do nearly the same thing,
2. the same data is parsed/transformed in more than one place,
3. UI preset fields, CLI args, metadata fields, and runner args drift apart,
4. save/load or JSON update logic is duplicated,
5. validation/scoring/reporting code is similar in several branches,
6. a new feature would require copying and slightly modifying an existing block.

Do not automatically refactor the whole codebase. Recommend either a minimal safe implementation now with a parked refactor, or a small focused refactor first if duplication would otherwise increase fragility.

## Preserve behaviour

Follow repo-root `AGENTS.md` §9 for behaviour-preservation requirements.

## Minimise impact

1. Make the smallest coherent change.
2. Add CLI/UI/preset/metadata fields only when needed.
3. Avoid broad rewrites and speculative features.
4. Keep new helpers narrow and placed near related code.
5. Leave clear TODOs rather than half-implementing speculative future features.

## Explorer/launcher/strategy boundaries

1. Explorer acceptance, validation, scoring, keeper/archive logic, and UI command-building must stay separated.
2. Strategy parameter JSON updates should use existing JSON helpers.
3. Validation summaries should reuse existing backtest metric parsing.
4. Launcher preset fields must stay aligned with runner CLI args.
5. Any new UI field must have a preset key and command-building behaviour.
6. Any new runner feature must have dry-run/no-apply behaviour.
7. When launcher, UI, preset, and runner settings drift apart, fix the shared source rather than patching only one caller.

## Indicator performance

Follow repo-root `AGENTS.md` §9 for indicator-vectorization requirements.
