# Entry Sieve Agent Notes

Read this first:

- `user_data/Custom_Launcher/docs/EXPLORER_ENTRY_SIEVE_MANUAL.md`
- `user_data/strategies/AGENTS.md`

This file is intentionally short to avoid stale duplicated rules.

Current source of truth:

- Entry Sieve is Sieve1: entry-quality diagnostics only.
- Entry Sieve results are informative diagnostics, not pass/fail acceptance.
- Strategy agents work on strategy files and explicitly requested strategy infrastructure.
- Strategy agents must not tune, reshape, or edit indicator modules unless explicitly asked.
- Indicator contracts and strategy-facing indicator docs take priority.
- Normal Sieve1 files should test one core entry concept with a small coarse hyperopt surface.
- Explicit confluence probes belong in `sieve1_multiN_*` files/classes.
- Do not add broad defensive workaround code without root-cause analysis and approval.
