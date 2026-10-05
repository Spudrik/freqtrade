"""Re-score active event/FreqAI evidence with one vote per whole event episode."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth as breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_interaction_branch_freeze_2026 as branch_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_meme_transmission_2026 as meme,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_fresh_2026 as fresh,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_portfolio as portfolio,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_simple_signal_families as simple,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "event_active_guardrail_audit_20260910a"
OUTPUT_ROOT = branch_freeze.OUTPUT_ROOT
RESULT_PATH = OUTPUT_ROOT / "active_test_guardrail_audit.json"
FRESH_RESULT_PATH = (
    simple.SOURCE_ROOT
    / "freqai"
    / "event_signal_fresh_2026_20260909a"
    / fresh.RESULT_NAME
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_list(value: Any) -> list[str]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got {value!r}")
    return [str(item) for item in parsed]


def _truthy(values: Series) -> Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def collapse_actual_to_one_anchor(
    frame: DataFrame,
    *,
    period_column: str,
    identity_columns: Sequence[str] = ("pair",),
    anchor_selection: str = "earliest",
) -> tuple[DataFrame, dict[str, Any]]:
    """Keep one actual row per episode and identity while retaining every control."""
    if anchor_selection not in {"earliest", "latest"}:
        raise ValueError(f"Unsupported anchor selection: {anchor_selection}")
    actual = frame.loc[frame["sample_kind"].eq("actual_event")].copy()
    controls = frame.loc[~frame["sample_kind"].eq("actual_event")].copy()
    parsed = actual["parent_episode_ids_json"].map(_json_list)
    if parsed.map(len).ne(1).any():
        raise ValueError(
            "Whole-episode correction requires exactly one parent episode per actual row."
        )
    actual["_episode_id"] = parsed.str[0]
    sort_columns = [
        *identity_columns,
        period_column,
        "_episode_id",
        "date",
        "sample_id",
    ]
    keys = [*identity_columns, period_column, "_episode_id"]
    actual = actual.sort_values(sort_columns, kind="stable").drop_duplicates(
        keys,
        keep="last" if anchor_selection == "latest" else "first",
    )
    corrected = pd.concat(
        [actual.drop(columns="_episode_id"), controls], ignore_index=True
    )
    return corrected, {
        "actual_rows_before": int(frame["sample_kind"].eq("actual_event").sum()),
        "actual_rows_after": len(actual),
        "removed_repeated_actual_anchors": int(
            frame["sample_kind"].eq("actual_event").sum() - len(actual)
        ),
        "control_rows_preserved": len(controls),
        "anchor_selection": anchor_selection,
    }


def episode_independence_audit(frame: DataFrame, *, period_column: str) -> dict[str, Any]:
    actual = frame.loc[frame["sample_kind"].eq("actual_event")].copy()
    actual = actual.drop_duplicates(
        [period_column, "sample_id", "parent_episode_ids_json"]
    )
    actual["episode_id"] = actual["parent_episode_ids_json"].map(_json_list).str[0]
    period_rows: list[dict[str, Any]] = []
    for period, cell in actual.groupby(period_column, observed=True, sort=False):
        counts = cell.groupby("episode_id", sort=False).size()
        period_rows.append(
            {
                "period": str(period),
                "actual_anchor_rows": len(cell),
                "independent_episodes": int(counts.size),
                "repeated_anchor_rows": int(len(cell) - counts.size),
                "episodes_with_multiple_anchors": int(counts.gt(1).sum()),
                "maximum_anchors_in_one_episode": int(counts.max()),
            }
        )
    split_count = actual.groupby("episode_id", sort=False)[period_column].nunique()
    return {
        "periods": period_rows,
        "episodes_crossing_period_boundaries": int(split_count.gt(1).sum()),
    }


def control_reuse_audit(frame: DataFrame, *, period_column: str) -> dict[str, Any]:
    controls = frame.loc[frame["sample_kind"].eq("matched_control")].drop_duplicates(
        [period_column, "sample_id"]
    )
    parent_counts = controls["parent_episode_ids_json"].map(_json_list).map(len)
    exploded = controls[[period_column, "sample_id", "parent_episode_ids_json"]].copy()
    exploded["episode_id"] = exploded["parent_episode_ids_json"].map(_json_list)
    exploded = exploded.explode("episode_id")
    per_episode = exploded.groupby([period_column, "episode_id"], sort=False)[
        "sample_id"
    ].nunique()
    return {
        "unique_control_clocks": len(controls),
        "controls_linked_to_multiple_parent_episodes": int(parent_counts.gt(1).sum()),
        "maximum_parent_episodes_for_one_control": int(parent_counts.max()),
        "median_controls_per_parent_episode": float(per_episode.median()),
        "minimum_controls_per_parent_episode": int(per_episode.min()),
        "maximum_controls_per_parent_episode": int(per_episode.max()),
        "interpretation": (
            "A reused control is valid matching information but is not a new independent "
            "market episode each time it is linked to another parent."
        ),
    }


def _decision_change_summary(
    old: DataFrame,
    new: DataFrame,
    *,
    keys: Sequence[str],
    decision_column: str,
) -> dict[str, Any]:
    old_view = old[[*keys, decision_column]].copy()
    new_view = new[[*keys, decision_column]].copy()
    old_view[decision_column] = _truthy(old_view[decision_column])
    new_view[decision_column] = _truthy(new_view[decision_column])
    joined = old_view.merge(
        new_view,
        on=list(keys),
        how="outer",
        suffixes=("_old", "_episode_weighted"),
        validate="one_to_one",
    )
    old_pass = joined[f"{decision_column}_old"].fillna(False).astype(bool)
    new_pass = joined[f"{decision_column}_episode_weighted"].fillna(False).astype(bool)
    changed = joined.loc[old_pass.ne(new_pass)].copy()
    changed["change"] = np.where(
        old_pass.loc[changed.index], "lost_after_correction", "new_after_correction"
    )
    return {
        "old_pass_cells": int(old_pass.sum()),
        "episode_weighted_pass_cells": int(new_pass.sum()),
        "lost_after_correction": int((old_pass & ~new_pass).sum()),
        "new_after_correction": int((~old_pass & new_pass).sum()),
        "changed_cells": changed[[*keys, "change"]].to_dict(orient="records"),
    }


def _retained_records(
    frame: DataFrame,
    *,
    decision_column: str,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    selected = frame.loc[_truthy(frame[decision_column]), list(columns)].copy()
    return selected.to_dict(orient="records")


def rescore_historical_freqai() -> tuple[dict[str, Any], dict[str, DataFrame]]:
    manifest = _load_json(portfolio.SOURCE_MANIFEST_PATH)
    predictions = portfolio.load_profile_predictions(manifest)
    actual = breadth.load_actual(manifest)
    independence = episode_independence_audit(actual, period_column="period")
    controls = control_reuse_audit(actual, period_column="period")
    corrected, correction = collapse_actual_to_one_anchor(
        actual,
        period_column="period",
        anchor_selection="earliest",
    )

    pair_scores, prediction_audit, fair = breadth.pair_scores(
        manifest=manifest,
        predictions=predictions,
        actual=corrected,
    )
    scope_scores = breadth.scope_scores(pair_scores)
    direct = breadth.direct_decisions(
        scope_scores,
        manifest["validation_periods"],
        technical_smoke=False,
    )
    _, combinations = breadth.combination_decisions(
        scope_scores,
        manifest["comparisons"],
        manifest["validation_periods"],
    )
    joint_pairs = breadth.joint_pair_scores(
        manifest=manifest,
        predictions=predictions,
        fair=fair,
    )
    joint_scopes = breadth.joint_scope_scores(joint_pairs)
    joint = breadth.joint_decisions(
        joint_scopes,
        manifest["validation_periods"],
        technical_smoke=False,
    )

    source_result = _load_json(portfolio.SOURCE_RESULT_PATH)
    old_direct = pd.read_csv(source_result["artifacts"]["direct_decisions"]["path"])
    old_combinations = pd.read_csv(
        source_result["artifacts"]["combination_decisions"]["path"]
    )
    old_joint = pd.read_csv(source_result["artifacts"]["joint_decisions"]["path"])
    direct_change = _decision_change_summary(
        old_direct,
        direct,
        keys=("market_scope", "profile_id", "role", "target", "target_kind"),
        decision_column="repeated_55_percent_pass",
    )
    combination_change = _decision_change_summary(
        old_combinations,
        combinations,
        keys=("market_scope", "candidate", "components", "target", "target_kind"),
        decision_column="all_periods_pass",
    )
    joint_change = _decision_change_summary(
        old_joint,
        joint,
        keys=(
            "market_scope",
            "profile_id",
            "role",
            "horizon_hours",
            "direction_metric",
        ),
        decision_column="joint_55_percent_pass",
    )
    corrected_direction = _retained_records(
        direct.loc[direct["target_kind"].eq("direction")],
        decision_column="repeated_55_percent_pass",
        columns=(
            "market_scope",
            "profile_id",
            "target",
            "minimum_correct_side_rate",
            "minimum_margin_over_training_majority",
            "minimum_rank_relationship",
            "minimum_independent_samples",
        ),
    )
    corrected_combinations = _retained_records(
        combinations,
        decision_column="all_periods_pass",
        columns=(
            "market_scope",
            "candidate",
            "components",
            "target",
            "target_kind",
            "minimum_accuracy_uplift",
        ),
    )
    return (
        {
            "independence": independence,
            "control_reuse": controls,
            "correction": correction,
            "prediction_key_audit": prediction_audit.to_dict(orient="records"),
            "direct_decision_changes": direct_change,
            "combination_decision_changes": combination_change,
            "joint_decision_changes": joint_change,
            "corrected_direction_leads": corrected_direction,
            "corrected_combination_leads": corrected_combinations,
        },
        {"predictions": predictions, "corrected_actual": corrected},
    )


def rescore_signal_portfolio(
    predictions: Mapping[str, DataFrame], corrected_actual: DataFrame
) -> dict[str, Any]:
    manifest = _load_json(portfolio.SOURCE_MANIFEST_PATH)
    rows = portfolio.build_signal_rows(
        source_manifest=manifest,
        predictions=predictions,
        actual=corrected_actual,
    )
    pair_scores = portfolio.score_pair_periods(rows)
    scope_scores = portfolio.add_market_scopes(pair_scores)
    decisions = portfolio.route_decisions(scope_scores)
    old = pd.read_csv(portfolio.DECISIONS_PATH)
    keys = (
        "family_id",
        "route_id",
        "route_kind",
        "market_scope",
        "horizon_hours",
        "activity_metric",
        "activity_target",
        "direction_metric",
        "direction_target",
    )
    activity = _decision_change_summary(
        old, decisions, keys=keys, decision_column="activity_pass"
    )
    direction = _decision_change_summary(
        old, decisions, keys=keys, decision_column="direction_pass"
    )
    joint = _decision_change_summary(
        old, decisions, keys=keys, decision_column="joint_pass"
    )
    activity_logical = decisions.loc[_truthy(decisions["activity_pass"])].drop_duplicates(
        ["family_id", "route_id", "market_scope", "horizon_hours", "activity_metric"]
    )
    direction_logical = decisions.loc[
        _truthy(decisions["direction_pass"])
    ].drop_duplicates(
        ["family_id", "route_id", "market_scope", "horizon_hours", "direction_metric"]
    )
    corrected_direction = direction_logical[
        [
            "family_id",
            "route_id",
            "market_scope",
            "horizon_hours",
            "direction_metric",
            "minimum_direction_correct_rate",
            "minimum_direction_margin_over_majority",
            "minimum_direction_margin_over_trend",
        ]
    ].to_dict(orient="records")
    return {
        "rows_scored_after_episode_collapse": len(rows),
        "activity_decision_changes": activity,
        "direction_decision_changes": direction,
        "joint_decision_changes": joint,
        "corrected_unique_logical_activity_leads": len(activity_logical),
        "corrected_unique_logical_direction_leads": len(direction_logical),
        "corrected_direction_leads": corrected_direction,
        "training_surface_limit": (
            "This re-score gives each evaluation episode one vote, but the historical "
            "models were trained on anchor rows. Do not promote or reuse those models; "
            "a future model must collapse or weight the training surface by episode."
        ),
    }


def rescore_fresh_freqai() -> dict[str, Any]:
    source_result = _load_json(FRESH_RESULT_PATH)
    membership = pd.read_parquet(source_result["artifacts"]["membership"]["path"])
    independence = episode_independence_audit(membership, period_column="period")
    corrected, correction = collapse_actual_to_one_anchor(
        membership,
        period_column="period",
        identity_columns=("profile_id", "pair"),
        anchor_selection="earliest",
    )
    pair_scores = fresh.score_pairs(corrected)
    scope_scores = fresh.add_scope_scores(pair_scores)
    periods = (
        "untouched_confirmation_2026_jan_apr",
        "untouched_confirmation_2026_may_aug",
    )
    decisions = fresh.decide(scope_scores, periods)
    old = pd.read_csv(source_result["artifacts"]["decisions"]["path"])
    changes = _decision_change_summary(
        old,
        decisions,
        keys=("family_id", "route_id", "profile_id", "market_scope"),
        decision_column="fresh_activity_pass",
    )
    return {
        "independence": independence,
        "correction": correction,
        "decision_changes": changes,
    }


def active_direct_result_checks() -> dict[str, Any]:
    simple_calls = pd.read_parquet(simple.CALLS_PATH)
    simple_keys = [
        "signal_id",
        "pair",
        "sample_kind",
        "model_period",
        "analysis_unit_id",
    ]
    simple_duplicates = int(simple_calls.duplicated(simple_keys).sum())
    meme_samples = pd.read_csv(meme.FROZEN_SAMPLES_PATH)
    meme_actual = meme_samples.loc[meme_samples["sample_kind"].eq("actual_event")]
    meme_duplicates = int(meme_actual.duplicated("episode_id").sum())
    return {
        "simple_six_family_sweep": {
            "duplicate_analysis_units_within_scored_cell": simple_duplicates,
            "classification": (
                "already_whole_episode_compliant"
                if simple_duplicates == 0
                else "needs_correction"
            ),
        },
        "meme_transmission": {
            "actual_rows": len(meme_actual),
            "independent_episodes": int(meme_actual["episode_id"].nunique()),
            "repeated_episode_rows": meme_duplicates,
            "classification": (
                "already_whole_episode_compliant"
                if meme_duplicates == 0
                else "needs_correction"
            ),
        },
        "conditional_episode_batch": {
            "classification": "already_whole_episode_compliant_after_20260909_repairs",
            "anchor_rule": (
                "earliest actual anchor for background/level questions; latest actual "
                "anchor only for the explicitly later accumulation question"
            ),
            "sign_branch_limit": (
                "The raw signed-accumulation field is a representation failure because "
                "it adds unlike source arithmetic. Its outcome is not market evidence."
            ),
        },
    }


def role_and_time_audit() -> dict[str, Any]:
    events = pd.read_csv(branch_freeze.EVENT_CATALOG_PATH)
    events["model_anchor_utc"] = pd.to_datetime(
        events["model_anchor_utc"], utc=True, errors="raise"
    )
    events["decision_utc"] = pd.to_datetime(events["decision_utc"], utc=True)
    mapping = branch_freeze.FAMILY_ROLE_AND_DOMAIN
    events["role"] = events["event_family"].map(lambda value: mapping[str(value)][0])
    episode_roles = events.groupby("event_episode_id", sort=False)["role"].agg(set)
    role_counts = (
        events.drop_duplicates(["event_episode_id", "role"])
        .groupby("role", sort=False)["event_episode_id"]
        .nunique()
        .to_dict()
    )
    mixed = int(episode_roles.map(len).gt(1).sum())
    role_combinations = episode_roles.map(
        lambda roles: "+".join(sorted(str(role) for role in roles))
    ).value_counts()
    source_sign = pd.to_numeric(events["source_sign_primary"], errors="coerce")
    relation_sign = pd.to_numeric(events["crypto_relation_sign"], errors="coerce")
    both = source_sign.notna() & relation_sign.notna()
    opposite = both & np.sign(source_sign).ne(np.sign(relation_sign))
    family_signs: list[dict[str, Any]] = []
    for family, cell in events.groupby("event_family", observed=True, sort=False):
        source = pd.to_numeric(cell["source_sign_primary"], errors="coerce")
        relation = pd.to_numeric(cell["crypto_relation_sign"], errors="coerce")
        both_cell = source.notna() & relation.notna()
        family_signs.append(
            {
                "event_family": str(family),
                "role": mapping[str(family)][0],
                "domain": mapping[str(family)][1],
                "events": len(cell),
                "source_sign_available": int(source.notna().sum()),
                "crypto_relation_sign_available": int(relation.notna().sum()),
                "opposite_source_and_crypto_sign": int(
                    (both_cell & np.sign(source).ne(np.sign(relation))).sum()
                ),
            }
        )
    delay_minutes = (
        events["model_anchor_utc"] - events["decision_utc"]
    ).dt.total_seconds() / 60.0
    return {
        "catalogue_episode_roles": {
            "total_event_episodes": int(episode_roles.size),
            "episodes_by_role_membership": {
                str(key): int(value) for key, value in role_counts.items()
            },
            "episodes_containing_more_than_one_role": mixed,
            "episode_role_combinations": {
                str(key): int(value) for key, value in role_combinations.items()
            },
            "interpretation_correction": (
                "The generic event-identity model mixes upstream announcements with "
                "already-observed cross-market responses. It is an event-or-state "
                "prototype, not a pure news model."
            ),
        },
        "source_sign_semantics": {
            "rows_with_both_signs": int(both.sum()),
            "rows_where_raw_source_and_crypto_sign_are_opposite": int(opposite.sum()),
            "family_coverage": family_signs,
            "classification": "confirmed_representation_failure_for_cross_family_sum",
        },
        "availability_alignment": {
            "events": len(events),
            "decision_after_model_anchor_violations": int(delay_minutes.lt(0).sum()),
            "maximum_alignment_delay_minutes": float(delay_minutes.max()),
            "market_features_use_completed_prior_candles": True,
            "early_market_response_is_later_decision_confirmation": True,
        },
    }


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        return _load_json(RESULT_PATH)
    frozen = _load_json(branch_freeze.FREEZE_PATH)
    if frozen.get("status") != "frozen_event_interaction_branches_before_branch_outcomes":
        raise ValueError("The parent interaction branch batch is not frozen.")
    historical, reusable = rescore_historical_freqai()
    portfolio_result = rescore_signal_portfolio(
        reusable["predictions"], reusable["corrected_actual"]
    )
    fresh_result = rescore_fresh_freqai()
    direct_checks = active_direct_result_checks()
    role_time = role_and_time_audit()
    historical_material = bool(
        historical["direct_decision_changes"]["lost_after_correction"]
        or historical["combination_decision_changes"]["lost_after_correction"]
        or historical["joint_decision_changes"]["lost_after_correction"]
    )
    portfolio_material = bool(
        portfolio_result["activity_decision_changes"]["lost_after_correction"]
        or portfolio_result["direction_decision_changes"]["lost_after_correction"]
        or portfolio_result["joint_decision_changes"]["lost_after_correction"]
    )
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_at_utc": g0.utc_now(),
        "status": "completed_active_event_guardrail_audit",
        "objective": (
            "Check active event and FreqAI results for repeated-event weighting, time "
            "availability, driver/response role confusion, control dependence, and "
            "unadjusted interpretation breadth."
        ),
        "historical_freqai_episode_rescore": historical,
        "historical_signal_portfolio_episode_rescore": portfolio_result,
        "fresh_2026_freqai_episode_rescore": fresh_result,
        "active_direct_result_checks": direct_checks,
        "role_and_time_audit": role_time,
        "decisions": {
            "historical_freqai_conclusions_changed_by_episode_weighting": historical_material,
            "signal_portfolio_conclusions_changed_by_episode_weighting": portfolio_material,
            "generic_event_identity_wording_must_change": True,
            "raw_cross_family_signed_accumulation_must_be_withdrawn": True,
            "cpi_fomc_short_activity_reopened": False,
            "simple_volume_persistence_reopened": False,
            "meme_whole_episode_result_reopened": False,
            "conditional_branch_batch_may_continue_after_corrections": True,
        },
        "interpretation": (
            "A corrected result may preserve, weaken, or remove a prototype. None of "
            "these retrospective re-scores is fresh confirmation or a trading rule."
        ),
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "branch_freeze": artifact(branch_freeze.FREEZE_PATH),
            "historical_freqai_result": artifact(portfolio.SOURCE_RESULT_PATH),
            "historical_portfolio_result": artifact(portfolio.RESULT_PATH),
            "fresh_freqai_result": artifact(FRESH_RESULT_PATH),
            "simple_signal_result": artifact(simple.RESULT_PATH),
            "meme_result": artifact(meme.RESULT_PATH),
        },
        "parent_freeze_status": frozen["status"],
        "profit_used": False,
        "trading_rule_created": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(output, RESULT_PATH)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(overwrite=args.overwrite), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
