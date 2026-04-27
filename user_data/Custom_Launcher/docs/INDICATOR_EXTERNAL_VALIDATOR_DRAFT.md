# Indicator External Validator (Draft)

Status: experimental draft. This feature is explicitly subject to deletion, rewrite, or removal.

## Purpose

Run `indicator_validation_runner.py` from Launcher V2 and evaluate indicator score behaviour against forward outcomes, separate from Hyperopt.

## Current Surface

- Launcher tab: `Indicator External Validator`
- Runner: `Custom_Launcher/indicator_validation_runner.py`
- Default output folder: `user_data/Indicator_External_Validator`

## Important Notes

- This is not a stable production pipeline.
- Output schemas and checks can change without compatibility guarantees.
- Runtime files in `Indicator_External_Validator` are disposable working artifacts.
- Keep only files you actively need.

## Typical Outputs

- `latest_indicator_validation_summary.json`
- `latest_indicator_score_metrics.csv`
- `latest_indicator_deciles.csv`
- `latest_indicator_behavior.csv`
- `latest_indicator_contract.csv`

## Operational Intention

Use this as a quick indicator honesty/sorting check before including indicator bundles in strategy confluence work.
