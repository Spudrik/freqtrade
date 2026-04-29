# Strategies Agent Rules

- Strategy files intended for HyperOpt must be standalone strategy modules, including Sieve1 and future sieve files.
- Do not use parent strategies, mixin strategy bases, or external strategy helper files for hyperopted strategy files. Freqtrade multi-process HyperOpt pickling can break when strategy logic depends on helper files or inherited strategy classes.
- Indicator modules under `user_data/Indicators/` are allowed dependencies when the task explicitly uses indicator outputs; do not edit indicators unless the user asks.
- Sieve1 results have no pass/fail or acceptance criteria. Treat results as informative diagnostics for refining useful entry signals.
- Sieve1 is the entry-quality sieve. Future Sieve2/Sieve3/Sieve4 passes may test exits, adds, global guards, or other ideas, but do not implement those without explicit user request.
- Use the `sieve1_` file prefix and `Sieve1` class prefix for current entry-sieve strategy files so future sieve generations can be filtered cleanly.
- Sieve1 strategy names should describe the entry idea directly. Do not use vague generator-origin tokens such as `test` or `codex` in filenames, class names, or visible entry tags.
- Daily TOP10 strategy files are excluded from Sieve1 because they contain broader strategy-management scaffolding, not just entry-condition refinement.
- Before editing sieve or strategy infrastructure, check that local definitions for "sieve", "strategy", "entry", "exit", "guard", "acceptance", and result interpretation are still accurate. Report deviations or stale instructions to the user instead of silently coding against outdated definitions.
- Root-cause first: before adding fallback/workaround code, identify and fix the underlying mapping/config/state issue.
- Do not add defensive or rescue code by default to "make it work".
- Prefer one-time data/state/path migrations over persistent runtime workaround logic.
- If a workaround seems unavoidable, stop and ask for explicit approval before adding it.
