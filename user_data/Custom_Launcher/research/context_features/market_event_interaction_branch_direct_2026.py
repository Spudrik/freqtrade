"""Run the frozen role-aware event interaction branch batch."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_direct_2026 as parent_direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_freeze_2026 as parent_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_interaction_branch_freeze_2026 as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260910a"
CELLS_PATH = OUTPUT_ROOT / "event_interaction_branch_cells.csv"
DECISIONS_PATH = OUTPUT_ROOT / "event_interaction_branch_decisions.csv"
RESULT_PATH = OUTPUT_ROOT / "event_interaction_branch_direct_result.json"

PAIRS = parent_direct.MARKET_PAIRS
HORIZONS = (1, 4, 8)
DECISION_PERIODS = parent_direct.DECISION_PERIODS
MINIMUM_STATE_EPISODES = 10
MINIMUM_CONDITIONAL_EFFECT = 0.10
MINIMUM_SIGNED_SUCCESS = 0.55


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got {value!r}")
    return [str(item) for item in parsed]


def _verify_contract(contract: Mapping[str, Any]) -> None:
    path = Path(str(contract["path"]))
    if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
        raise ValueError(f"Frozen contract changed: {path}")


def load_freeze() -> dict[str, Any]:
    contract = _load_json(frozen.FREEZE_PATH)
    if contract.get("status") != "frozen_event_interaction_branches_before_branch_outcomes":
        raise ValueError("Interaction branch batch is not frozen.")
    for source in contract["source_contracts"].values():
        _verify_contract(source)
    return contract


def _domain_signs(known: DataFrame, response_domains: set[str]) -> list[int]:
    signs: list[int] = []
    for domain in sorted(response_domains):
        values = pd.to_numeric(
            known.loc[known["domain"].eq(domain), "crypto_relation_sign"],
            errors="coerce",
        ).dropna()
        observed = set(np.sign(values.loc[values.ne(0.0)]).astype(int))
        if len(observed) == 1:
            signs.append(observed.pop())
    return signs


def build_role_context(samples: DataFrame, events: DataFrame) -> DataFrame:
    """Describe distinct known information and response domains at each sample time."""
    mapping = frozen.FAMILY_ROLE_AND_DOMAIN
    events = events.copy()
    events["model_anchor_utc"] = pd.to_datetime(
        events["model_anchor_utc"], utc=True, errors="raise"
    )
    events["role"] = events["event_family"].map(lambda value: mapping[str(value)][0])
    events["domain"] = events["event_family"].map(lambda value: mapping[str(value)][1])
    records: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        timestamp = pd.Timestamp(sample.model_anchor_utc)
        known = events.loc[
            events["model_anchor_utc"].le(timestamp)
            & events["model_anchor_utc"].gt(timestamp - pd.Timedelta(hours=24))
        ].drop_duplicates(["event_episode_id", "event_family", "domain"])
        upstream_domains = set(
            known.loc[known["role"].eq("upstream_information"), "domain"].astype(str)
        )
        detailed_response_domains = set(
            known.loc[known["role"].eq("market_response"), "domain"].astype(str)
        )
        composite_domains = set(
            known.loc[
                known["role"].eq("composite_market_response"), "domain"
            ].astype(str)
        )
        response_domains = (
            detailed_response_domains
            if detailed_response_domains
            else composite_domains
        )
        signs = _domain_signs(known, response_domains)
        aligned_sign = (
            signs[0] if len(signs) >= 2 and len(set(signs)) == 1 else 0
        )
        actual = str(sample.sample_kind) == "actual_event"
        current_families = (
            [
                family
                for family in _json_list(sample.event_families_json)
                if family in mapping
            ]
            if actual
            else []
        )
        current_roles = {mapping[family][0] for family in current_families}
        current_upstream_families = sorted(
            family
            for family in current_families
            if mapping[family][0] == "upstream_information"
        )
        current_response_families = sorted(
            family
            for family in current_families
            if mapping[family][0]
            in {"market_response", "composite_market_response"}
        )
        has_upstream = bool(upstream_domains)
        has_response = bool(response_domains)
        if has_upstream and has_response:
            role_mix = "upstream_plus_response"
        elif has_upstream:
            role_mix = "upstream_only"
        elif has_response:
            role_mix = "response_only"
        else:
            role_mix = "neither_upstream_nor_response"
        records.append(
            {
                "date": timestamp,
                "upstream_domain_count": len(upstream_domains),
                "response_domain_count": len(response_domains),
                "media_attention_proxy_count": int(
                    known["role"].eq("media_attention_proxy").sum()
                ),
                "multiple_upstream_domains": len(upstream_domains) >= 2,
                "multiple_response_domains": len(response_domains) >= 2,
                "has_upstream": has_upstream,
                "has_response": has_response,
                "role_mix": role_mix,
                "signed_response_domain_count": len(signs),
                "aligned_response_sign": aligned_sign,
                "current_has_upstream": "upstream_information" in current_roles,
                "current_has_response": bool(
                    current_roles.intersection(
                        {"market_response", "composite_market_response"}
                    )
                ),
                "current_is_media_attention_proxy": (
                    "media_attention_proxy" in current_roles
                ),
                "current_family_key": json.dumps(sorted(current_families)),
                "current_upstream_family_key": json.dumps(current_upstream_families),
                "current_response_family_key": json.dumps(current_response_families),
            }
        )
    output = DataFrame.from_records(records)
    if output["date"].duplicated().any():
        raise ValueError("Role context contains duplicate sample dates.")
    return output


def _cache_inventory() -> list[dict[str, Any]]:
    manifest = _load_json(parent_freeze.CACHE_MANIFEST_PATH)
    inventory = [item for item in manifest["inventory"] if str(item["pair"]) in PAIRS]
    if {str(item["pair"]) for item in inventory} != set(PAIRS):
        raise ValueError("The event cache does not contain the five frozen pairs.")
    for item in inventory:
        _verify_contract({"path": item["feature_path"], "sha256": item["feature_sha256"]})
        _verify_contract(
            {"path": item["evaluation_path"], "sha256": item["evaluation_sha256"]}
        )
    return inventory


def load_market_rows() -> DataFrame:
    samples = pd.read_csv(frozen.SAMPLE_CATALOG_PATH)
    samples["model_anchor_utc"] = pd.to_datetime(
        samples["model_anchor_utc"], utc=True, errors="raise"
    )
    events = pd.read_csv(frozen.EVENT_CATALOG_PATH)
    role_context = build_role_context(samples, events)
    evaluation_columns = [
        "date",
        "period",
        "sample_id",
        "sample_kind",
        "parent_episode_ids_json",
        "event_families_json",
        "event_kinds_json",
        *[
            f"raw_{metric}_h{horizon}"
            for horizon in HORIZONS
            for metric in ("volume_ratio", "close_return_atr")
        ],
    ]
    parts: list[DataFrame] = []
    for item in _cache_inventory():
        pair = str(item["pair"])
        features = pd.read_parquet(
            item["feature_path"],
            columns=["date", "background__range_position_720h"],
        )
        outcomes = pd.read_parquet(item["evaluation_path"], columns=evaluation_columns)
        for frame in (features, outcomes):
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        joined = outcomes.merge(features, on="date", how="left", validate="many_to_one")
        joined = joined.merge(role_context, on="date", how="left", validate="many_to_one")
        joined = joined.rename(columns={"period": "model_period"})
        joined["pair"] = pair
        joined["event_class"] = joined["event_kinds_json"].map(parent_direct.event_class)
        position = pd.to_numeric(
            joined["background__range_position_720h"], errors="coerce"
        )
        joined["range_zone"] = np.select(
            [position.le(0.20), position.ge(0.80)],
            ["lower_fifth", "upper_fifth"],
            default="middle_sixty_percent",
        )
        joined.loc[position.isna(), "range_zone"] = "missing"
        for horizon in HORIZONS:
            part = joined[
                [
                    "date",
                    "sample_id",
                    "sample_kind",
                    "model_period",
                    "pair",
                    "parent_episode_ids_json",
                    "event_families_json",
                    "event_kinds_json",
                    "event_class",
                    "range_zone",
                    "upstream_domain_count",
                    "response_domain_count",
                    "multiple_upstream_domains",
                    "multiple_response_domains",
                    "has_upstream",
                    "has_response",
                    "role_mix",
                    "signed_response_domain_count",
                    "aligned_response_sign",
                    "current_has_upstream",
                    "current_has_response",
                    "current_is_media_attention_proxy",
                    "current_family_key",
                    "current_upstream_family_key",
                    "current_response_family_key",
                ]
            ].copy()
            part["horizon_hours"] = horizon
            volume = pd.to_numeric(
                joined[f"raw_volume_ratio_h{horizon}"], errors="coerce"
            )
            signed_move = pd.to_numeric(
                joined[f"raw_close_return_atr_h{horizon}"], errors="coerce"
            )
            part["volume_reaction"] = volume.ge(1.0).where(volume.notna())
            part["signed_move"] = signed_move
            parts.append(part)
    return parent_direct._with_combined_2026(pd.concat(parts, ignore_index=True))


def _actual_episode_rows(rows: DataFrame, *, keep: str) -> DataFrame:
    actual = rows.loc[rows["sample_kind"].eq("actual_event")].copy()
    episodes = actual["parent_episode_ids_json"].map(_json_list)
    if episodes.map(len).ne(1).any():
        raise ValueError("An actual anchor maps to more than one whole event episode.")
    actual["episode_id"] = episodes.str[0]
    actual = actual.sort_values(
        ["model_period", "episode_id", "pair", "horizon_hours", "date", "sample_id"],
        kind="stable",
    ).drop_duplicates(
        ["model_period", "episode_id", "pair", "horizon_hours"], keep=keep
    )
    return actual


def _control_episode_rows(rows: DataFrame) -> DataFrame:
    controls = rows.loc[rows["sample_kind"].eq("matched_control")].copy()
    controls["episode_id"] = controls["parent_episode_ids_json"].map(_json_list)
    controls = controls.explode("episode_id")
    return controls.drop_duplicates(
        ["model_period", "episode_id", "pair", "horizon_hours", "sample_id"]
    )


def paired_episode_outcomes(
    rows: DataFrame,
    *,
    anchor: str,
    same_range_zone: bool = False,
) -> DataFrame:
    actual = _actual_episode_rows(rows, keep="last" if anchor == "latest" else "first")
    controls = _control_episode_rows(rows).rename(
        columns={
            "sample_id": "control_sample_id",
            "volume_reaction": "control_volume_reaction",
            "range_zone": "control_range_zone",
        }
    )
    merged = actual.merge(
        controls[
            [
                "model_period",
                "episode_id",
                "pair",
                "horizon_hours",
                "control_sample_id",
                "control_volume_reaction",
                "control_range_zone",
            ]
        ],
        on=["model_period", "episode_id", "pair", "horizon_hours"],
        how="left",
        validate="one_to_many",
    )
    merged = merged.loc[
        merged["volume_reaction"].notna() & merged["control_volume_reaction"].notna()
    ]
    if same_range_zone:
        merged = merged.loc[merged["range_zone"].eq(merged["control_range_zone"])]
    keys = ["model_period", "episode_id", "pair", "horizon_hours"]
    passthrough = [
        "event_class",
        "range_zone",
        "upstream_domain_count",
        "response_domain_count",
        "multiple_upstream_domains",
        "multiple_response_domains",
        "has_upstream",
        "has_response",
        "role_mix",
        "signed_response_domain_count",
        "aligned_response_sign",
        "current_has_upstream",
        "current_has_response",
        "current_is_media_attention_proxy",
        "current_family_key",
        "current_upstream_family_key",
        "current_response_family_key",
        "signed_move",
    ]
    aggregations: dict[str, Any] = {
        column: (column, "first") for column in passthrough
    }
    aggregations.update(
        {
            "event_volume_reaction": ("volume_reaction", "first"),
            "control_volume_reaction": ("control_volume_reaction", "mean"),
            "matched_control_count": ("control_sample_id", "nunique"),
        }
    )
    output = merged.groupby(keys, observed=True, sort=False).agg(**aggregations).reset_index()
    output["event_minus_control"] = (
        output["event_volume_reaction"] - output["control_volume_reaction"]
    )
    return output


def _group_summary(
    frame: DataFrame,
    group_column: str,
    *,
    stratum_column: str,
) -> DataFrame:
    keys = [
        "model_period",
        "pair",
        "horizon_hours",
        stratum_column,
        group_column,
    ]
    return (
        frame.groupby(keys, observed=True, sort=False)
        .agg(
            event_count=("episode_id", "nunique"),
            raw_event_rate=("event_volume_reaction", "mean"),
            matched_control_rate=("control_volume_reaction", "mean"),
            mean_event_minus_control=("event_minus_control", "mean"),
            median_controls_per_event=("matched_control_count", "median"),
        )
        .reset_index()
    )


def _comparison_cells(
    summary: DataFrame,
    *,
    branch_id: str,
    group_column: str,
    stratum_column: str,
    test_group: str,
    baseline_group: str,
    comparison: str,
) -> DataFrame:
    keys = ["model_period", "pair", "horizon_hours"]
    matched_keys = [*keys, stratum_column]
    test = summary.loc[summary[group_column].astype(str).eq(test_group)].rename(
        columns={
            "event_count": "test_count",
            "raw_event_rate": "test_rate",
            "matched_control_rate": "test_control_rate",
            "mean_event_minus_control": "test_adjusted_rate",
        }
    )
    baseline = summary.loc[
        summary[group_column].astype(str).eq(baseline_group)
    ].rename(
        columns={
            "event_count": "baseline_count",
            "raw_event_rate": "baseline_rate",
            "matched_control_rate": "baseline_control_rate",
            "mean_event_minus_control": "baseline_adjusted_rate",
        }
    )
    joined = test.merge(
        baseline,
        on=matched_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_test", "_baseline"),
    )
    joined["common_weight"] = (
        joined["test_count"]
        * joined["baseline_count"]
        / (joined["test_count"] + joined["baseline_count"])
    )
    joined["stratum_conditional_effect"] = (
        joined["test_adjusted_rate"] - joined["baseline_adjusted_rate"]
    )
    weighted_columns = [
        "test_rate",
        "baseline_rate",
        "test_control_rate",
        "baseline_control_rate",
        "test_adjusted_rate",
        "baseline_adjusted_rate",
        "stratum_conditional_effect",
    ]
    for column in weighted_columns:
        joined[f"weighted_{column}"] = joined[column] * joined["common_weight"]
    if joined.empty:
        return DataFrame(
            columns=[
                "branch_id",
                "comparison",
                "decision_type",
                *keys,
                "test_group",
                "baseline_group",
                "matched_strata",
                "test_count",
                "baseline_count",
                "test_rate",
                "baseline_rate",
                "test_control_rate",
                "baseline_control_rate",
                "test_adjusted_rate",
                "baseline_adjusted_rate",
                "conditional_effect",
            ]
        )
    aggregations: dict[str, tuple[str, str]] = {
        "matched_strata": (stratum_column, "nunique"),
        "test_count": ("test_count", "sum"),
        "baseline_count": ("baseline_count", "sum"),
        "total_common_weight": ("common_weight", "sum"),
    }
    aggregations.update(
        {
            f"weighted_{column}": (f"weighted_{column}", "sum")
            for column in weighted_columns
        }
    )
    output = (
        joined.groupby(keys, observed=True, sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    for column in weighted_columns:
        result_column = (
            "conditional_effect"
            if column == "stratum_conditional_effect"
            else column
        )
        output[result_column] = (
            output[f"weighted_{column}"] / output["total_common_weight"]
        )
    output["branch_id"] = branch_id
    output["comparison"] = comparison
    output["decision_type"] = "conditional_activity"
    output["test_group"] = test_group
    output["baseline_group"] = baseline_group
    columns = [
        "branch_id",
        "comparison",
        "decision_type",
        *keys,
        "test_group",
        "baseline_group",
        "matched_strata",
        "test_count",
        "baseline_count",
        "test_rate",
        "baseline_rate",
        "test_control_rate",
        "baseline_control_rate",
        "test_adjusted_rate",
        "baseline_adjusted_rate",
        "conditional_effect",
    ]
    return output[columns]


def role_activity_cells(paired: DataFrame) -> DataFrame:
    cells: list[DataFrame] = []
    response = paired.loc[paired["current_has_response"] & paired["has_response"]].copy()
    response["response_group"] = np.where(
        response["multiple_response_domains"], "multiple_response_domains", "one_response_domain"
    )
    cells.append(
        _comparison_cells(
            _group_summary(
                response,
                "response_group",
                stratum_column="current_response_family_key",
            ),
            branch_id="role_aware_cross_market_overlap",
            group_column="response_group",
            stratum_column="current_response_family_key",
            test_group="multiple_response_domains",
            baseline_group="one_response_domain",
            comparison="multiple_response_domains_vs_one",
        )
    )
    upstream = paired.loc[paired["current_has_upstream"] & paired["has_upstream"]].copy()
    upstream["upstream_group"] = np.where(
        upstream["multiple_upstream_domains"], "multiple_upstream_domains", "one_upstream_domain"
    )
    cells.append(
        _comparison_cells(
            _group_summary(
                upstream,
                "upstream_group",
                stratum_column="current_upstream_family_key",
            ),
            branch_id="role_aware_cross_market_overlap",
            group_column="upstream_group",
            stratum_column="current_upstream_family_key",
            test_group="multiple_upstream_domains",
            baseline_group="one_upstream_domain",
            comparison="multiple_upstream_domains_vs_one",
        )
    )
    current_upstream = paired.loc[paired["current_has_upstream"]].copy()
    current_upstream["response_context_group"] = np.where(
        current_upstream["has_response"],
        "with_response_context",
        "without_response_context",
    )
    cells.append(
        _comparison_cells(
            _group_summary(
                current_upstream,
                "response_context_group",
                stratum_column="current_upstream_family_key",
            ),
            branch_id="role_aware_cross_market_overlap",
            group_column="response_context_group",
            stratum_column="current_upstream_family_key",
            test_group="with_response_context",
            baseline_group="without_response_context",
            comparison="upstream_with_response_vs_upstream_without_response",
        )
    )
    current_response = paired.loc[paired["current_has_response"]].copy()
    current_response["upstream_context_group"] = np.where(
        current_response["has_upstream"],
        "with_upstream_context",
        "without_upstream_context",
    )
    cells.append(
        _comparison_cells(
            _group_summary(
                current_response,
                "upstream_context_group",
                stratum_column="current_response_family_key",
            ),
            branch_id="role_aware_cross_market_overlap",
            group_column="upstream_context_group",
            stratum_column="current_response_family_key",
            test_group="with_upstream_context",
            baseline_group="without_upstream_context",
            comparison="response_with_upstream_vs_response_without_upstream",
        )
    )
    return pd.concat(cells, ignore_index=True)


def range_activity_cells(paired: DataFrame) -> DataFrame:
    eligible = paired.loc[
        paired["event_class"].eq("discrete_media_finance_or_corporate")
        & paired["horizon_hours"].eq(1)
        & ~paired["range_zone"].eq("missing")
    ].copy()
    summary = _group_summary(
        eligible,
        "range_zone",
        stratum_column="current_family_key",
    )
    return pd.concat(
        [
            _comparison_cells(
                summary,
                branch_id="upper_and_lower_range_edge_discrete_events",
                group_column="range_zone",
                stratum_column="current_family_key",
                test_group=edge,
                baseline_group="middle_sixty_percent",
                comparison=f"{edge}_vs_middle",
            )
            for edge in ("upper_fifth", "lower_fifth")
        ],
        ignore_index=True,
    )


def signed_continuation_cells(rows: DataFrame) -> DataFrame:
    actual = _actual_episode_rows(rows, keep="last")
    selected = actual.loc[
        actual["pair"].isin({parent_direct.BTC_PAIR, parent_direct.ETH_PAIR})
        & pd.to_numeric(actual["aligned_response_sign"], errors="coerce").ne(0)
        & pd.to_numeric(actual["signed_move"], errors="coerce").ne(0)
    ].copy()
    selected["predicted_sign"] = pd.to_numeric(
        selected["aligned_response_sign"], errors="coerce"
    ).astype(int)
    selected["actual_sign"] = np.sign(
        pd.to_numeric(selected["signed_move"], errors="coerce")
    ).astype(int)
    selected["correct"] = selected["predicted_sign"].eq(selected["actual_sign"])
    selected = selected.sort_values(
        ["model_period", "pair", "horizon_hours", "date"], kind="stable"
    )
    selected["rotated_sign"] = selected.groupby(
        ["model_period", "pair", "horizon_hours"], observed=True, sort=False
    )["predicted_sign"].transform(lambda values: values.shift(1).fillna(values.iloc[-1]))
    selected["rotated_correct"] = selected["rotated_sign"].eq(selected["actual_sign"])
    output = (
        selected.groupby(
            ["model_period", "pair", "horizon_hours"], observed=True, sort=False
        )
        .agg(
            test_count=("episode_id", "nunique"),
            test_rate=("correct", "mean"),
            baseline_rate=("rotated_correct", "mean"),
        )
        .reset_index()
    )
    output["branch_id"] = "role_aware_cross_market_overlap"
    output["comparison"] = "aligned_response_sign_continuation_vs_rotated_sign"
    output["decision_type"] = "signed_continuation"
    output["test_group"] = "two_or_more_agreeing_response_domain_signs"
    output["baseline_group"] = "same_signs_rotated_between_whole_episodes"
    output["baseline_count"] = output["test_count"]
    output["conditional_effect"] = output["test_rate"] - output["baseline_rate"]
    output["matched_strata"] = np.nan
    for column in (
        "test_control_rate",
        "baseline_control_rate",
        "test_adjusted_rate",
        "baseline_adjusted_rate",
    ):
        output[column] = np.nan
    return output[
        [
            "branch_id",
            "comparison",
            "decision_type",
            "model_period",
            "pair",
            "horizon_hours",
            "test_group",
            "baseline_group",
            "matched_strata",
            "test_count",
            "baseline_count",
            "test_rate",
            "baseline_rate",
            "test_control_rate",
            "baseline_control_rate",
            "test_adjusted_rate",
            "baseline_adjusted_rate",
            "conditional_effect",
        ]
    ]


def decide(cells: DataFrame) -> DataFrame:
    keys = ["branch_id", "comparison", "decision_type", "pair", "horizon_hours"]
    records: list[dict[str, Any]] = []
    period_order = {period: index for index, period in enumerate(DECISION_PERIODS)}
    for values, group in cells.groupby(keys, observed=True, sort=False):
        group = group.loc[group["model_period"].isin(DECISION_PERIODS)].copy()
        group["test_count"] = pd.to_numeric(group["test_count"], errors="coerce")
        group["baseline_count"] = pd.to_numeric(group["baseline_count"], errors="coerce")
        supported = group.loc[
            group["test_count"].ge(MINIMUM_STATE_EPISODES)
            & group["baseline_count"].ge(MINIMUM_STATE_EPISODES)
            & group["conditional_effect"].notna()
        ].copy()
        decision_type = str(values[2])
        if decision_type == "signed_continuation":
            passing = supported.loc[
                supported["test_rate"].ge(MINIMUM_SIGNED_SUCCESS)
                & supported["test_rate"].gt(supported["baseline_rate"])
            ]
            opposing = supported.loc[supported["test_rate"].le(0.45)]
            retained = len(passing) >= 2 and opposing.empty
            verdict = (
                "retained_confirmation_only_direction_lead"
                if retained
                else "coverage_parked"
                if len(supported) < 2
                else "unresolved_direction_not_repeated"
            )
        else:
            material = supported.loc[
                supported["conditional_effect"].abs().ge(MINIMUM_CONDITIONAL_EFFECT)
            ]
            positive = material.loc[material["conditional_effect"].gt(0)]
            negative = material.loc[material["conditional_effect"].lt(0)]
            retained = (len(positive) >= 2 and negative.empty) or (
                len(negative) >= 2 and positive.empty
            )
            verdict = (
                "retained_conditional_amplifier"
                if retained and len(positive) >= 2
                else "retained_conditional_suppressor"
                if retained
                else "unresolved_effect_changes_sign_between_periods"
                if not positive.empty and not negative.empty
                else "coverage_parked"
                if len(supported) < 2
                else "not_retained_effect_not_repeated"
            )
        latest = supported.assign(
            _period_order=supported["model_period"].map(period_order)
        ).sort_values("_period_order", ascending=False)
        has_later_period_support = bool(
            supported["model_period"].isin(
                {"walk_forward_validation_2025", "confirmation_2026"}
            ).any()
        )
        matched_strata = (
            pd.to_numeric(supported["matched_strata"], errors="coerce").dropna()
            if "matched_strata" in supported
            else pd.Series(dtype=float)
        )
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "verdict": verdict,
                "retained": retained,
                "has_later_period_support": has_later_period_support,
                "eligible_for_freqai": retained and has_later_period_support,
                "supported_periods": len(supported),
                "minimum_test_count": int(supported["test_count"].min())
                if len(supported)
                else 0,
                "minimum_baseline_count": int(supported["baseline_count"].min())
                if len(supported)
                else 0,
                "minimum_matched_event_families": int(matched_strata.min())
                if len(matched_strata)
                else 0,
                "median_conditional_effect": float(supported["conditional_effect"].median())
                if len(supported)
                else np.nan,
                "latest_supported_period": str(latest.iloc[0]["model_period"])
                if len(latest)
                else "",
                "latest_effect": float(latest.iloc[0]["conditional_effect"])
                if len(latest)
                else np.nan,
            }
        )
    return DataFrame.from_records(records)


def semantics_audit(events: DataFrame) -> dict[str, Any]:
    source = pd.to_numeric(events["source_sign_primary"], errors="coerce")
    relation = pd.to_numeric(events["crypto_relation_sign"], errors="coerce")
    both = source.notna() & relation.notna()
    opposite = both & np.sign(source).ne(np.sign(relation))
    upstream = events["event_family"].map(
        lambda family: frozen.FAMILY_ROLE_AND_DOMAIN[str(family)][0]
    ).eq("upstream_information")
    upstream_relation = int((upstream & relation.notna()).sum())
    return {
        "status": "representation_failed_closed_without_market_outcome_test",
        "event_rows": len(events),
        "raw_source_sign_rows": int(source.notna().sum()),
        "crypto_relation_sign_rows": int(relation.notna().sum()),
        "opposite_raw_source_and_crypto_sign_rows": int(opposite.sum()),
        "upstream_rows_with_crypto_relation_sign": upstream_relation,
        "reason": (
            "The old count sums positive and negative arithmetic from unlike source "
            "families. Some raw signs are the opposite of their available crypto "
            "relation, many upstream events have no causal expectation surprise, and "
            "market-response measurements are not independent news stories."
        ),
    }


def coherent_patterns(decisions: DataFrame) -> list[dict[str, Any]]:
    retained = decisions.loc[decisions["retained"].astype(bool)].copy()
    if retained.empty:
        return []
    retained["effect_sign"] = np.sign(retained["median_conditional_effect"])
    keys = ["branch_id", "comparison", "decision_type", "horizon_hours", "effect_sign"]
    records: list[dict[str, Any]] = []
    for values, group in retained.groupby(keys, observed=True, sort=False):
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "retained_pairs": sorted(group["pair"].astype(str).unique()),
                "retained_pair_count": int(group["pair"].nunique()),
                "cross_pair_pattern": bool(group["pair"].nunique() >= 3),
                "minimum_supported_periods": int(group["supported_periods"].min()),
            }
        )
    return records


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        return _load_json(RESULT_PATH)
    contract = load_freeze()
    rows = load_market_rows()
    latest = paired_episode_outcomes(rows, anchor="latest")
    earliest_range = paired_episode_outcomes(
        rows, anchor="earliest", same_range_zone=True
    )
    cells = pd.concat(
        [
            role_activity_cells(latest),
            signed_continuation_cells(rows),
            range_activity_cells(earliest_range),
        ],
        ignore_index=True,
    )
    decisions = decide(cells)
    events = pd.read_csv(frozen.EVENT_CATALOG_PATH)
    semantics = semantics_audit(events)
    patterns = coherent_patterns(decisions)
    freqai_eligible = decisions.loc[
        decisions["eligible_for_freqai"].astype(bool)
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(cells, CELLS_PATH)
    g0.atomic_write_csv(decisions, DECISIONS_PATH)
    result = {
        "schema_version": 2,
        "status": "completed_event_interaction_branch_direct_batch",
        "created_at_utc": g0.utc_now(),
        "run_id": frozen.RUN_ID,
        "branch_status": {
            "role_aware_cross_market_overlap": {
                "status": "completed_direct_review",
                "decision_cells": int(
                    decisions["branch_id"].eq("role_aware_cross_market_overlap").sum()
                ),
                "retained_cells": int(
                    (
                        decisions["branch_id"].eq("role_aware_cross_market_overlap")
                        & decisions["retained"].astype(bool)
                    ).sum()
                ),
                "freqai_eligible_cells": int(
                    (
                        decisions["branch_id"].eq(
                            "role_aware_cross_market_overlap"
                        )
                        & decisions["eligible_for_freqai"].astype(bool)
                    ).sum()
                ),
            },
            "signed_accumulation_semantics_audit": semantics,
            "upper_and_lower_range_edge_discrete_events": {
                "status": "completed_direct_review",
                "decision_cells": int(
                    decisions["branch_id"].eq(
                        "upper_and_lower_range_edge_discrete_events"
                    ).sum()
                ),
                "retained_cells": int(
                    (
                        decisions["branch_id"].eq(
                            "upper_and_lower_range_edge_discrete_events"
                        )
                        & decisions["retained"].astype(bool)
                    ).sum()
                ),
            },
        },
        "coherent_patterns": patterns,
        "freqai_decision": (
            "No branch advances to FreqAI because no directly retained relationship "
            "has adequate support in either 2025 or combined 2026."
            if freqai_eligible.empty
            else "Only the explicitly eligible direct relationships may proceed to a "
            "bounded nonlinear ranking or abstention test."
        ),
        "interpretation_boundaries": {
            "cross_market_measurements_are_confirmation_not_news": True,
            "gdelt_aggregate_is_attention_proxy_not_story_driver": True,
            "whole_episode_is_independent_unit": True,
            "conditional_comparisons_are_matched_within_current_event_family": True,
            "family_strata_use_common_harmonic_weights": True,
            "signed_continuation_starts_at_later_confirmation_time": True,
            "profit_used": False,
            "trading_rule_created": False,
        },
        "artifacts": {
            "cells": artifact(CELLS_PATH),
            "decisions": artifact(DECISIONS_PATH),
        },
        "source_contracts": {
            "branch_freeze": artifact(frozen.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "parent_freeze_status": contract["status"],
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(overwrite=args.overwrite), indent=2, default=g0.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
