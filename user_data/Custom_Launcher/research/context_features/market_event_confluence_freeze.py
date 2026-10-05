"""Freeze causal event-family overlap routes before reading their crypto outcomes."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
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
    market_cross_asset_relevance_freeze as cross_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_freeze as event_frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_direct as layer2_direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "event_confluence_20260904a"
)
BROAD_RESULT_PATH = event_frozen.OUTPUT_ROOT / "broad_relevance_freeze_result.json"
BROAD_CATALOG_PATH = event_frozen.OUTPUT_ROOT / "broad_relevance_event_catalog.csv"
CROSS_RESULT_PATH = cross_frozen.OUTPUT_ROOT / "cross_asset_freeze_result.json"
CROSS_CATALOG_PATH = cross_frozen.OUTPUT_ROOT / "cross_asset_event_catalog.csv"

LOOKBACK_HOURS = 24
EPISODE_COOLDOWN_HOURS = 72
EXTREME_SOURCE_MULTIPLE = 2.0
HORIZONS_HOURS = (1, 4, 8, 24, 72)
MINIMUM_PARTITION_EVENTS = 10
MINIMUM_ABOVE_CONTROL_RATE = 0.55
MINIMUM_MEDIAN_ACTIVITY_SCORE = 1.20
MINIMUM_DIRECTION_ACCURACY = 0.55
MINIMUM_DIRECTION_LIFT_VS_CONTROLS = 0.02
JOINT_LEAD_FLOOR = 0.55
JOINT_MAIN_TARGET = 0.65

ROUTES: dict[str, dict[str, Any]] = {
    "two_plus_families": {
        "label": "At least two recent event families",
        "prediction": 0,
    },
    "three_plus_families": {
        "label": "At least three recent event families",
        "prediction": 0,
    },
    "two_plus_source_groups": {
        "label": "At least two different source types",
        "prediction": 0,
    },
    "aligned_positive_market_signs": {
        "label": "At least two positive wider-market signs agree",
        "prediction": 1,
    },
    "aligned_negative_market_signs": {
        "label": "At least two negative wider-market signs agree",
        "prediction": -1,
    },
    "isolated_extreme_measurable_shock": {
        "label": "One isolated measurable shock at least twice its threshold",
        "prediction": 0,
    },
    "isolated_family_reference": {
        "label": "One isolated event family reference",
        "prediction": 0,
    },
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def verify_frozen_catalog(result_path: Path, catalog_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("outcomes_read"):
        raise ValueError(f"Upstream freeze unexpectedly read outcomes: {result_path}")
    if result["artifacts"]["catalog"]["sha256"] != g0.sha256_file(catalog_path):
        raise ValueError(f"Upstream frozen catalog changed: {catalog_path}")


def decision_hour(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.ceil("h")


def signed_contribution(change: float, relation: str) -> int:
    if not np.isfinite(change) or change == 0:
        return 0
    sign = 1 if change > 0 else -1
    if relation.startswith("same_direction"):
        return sign
    if relation.startswith("opposite_direction"):
        return -sign
    return 0


def load_source_events() -> DataFrame:
    verify_frozen_catalog(BROAD_RESULT_PATH, BROAD_CATALOG_PATH)
    verify_frozen_catalog(CROSS_RESULT_PATH, CROSS_CATALOG_PATH)
    broad = pd.read_csv(BROAD_CATALOG_PATH)
    broad["anchor_utc"] = pd.to_datetime(broad["anchor_utc"], utc=True)
    broad["source_group"] = broad["event_group"].replace(
        {
            "scheduled_release": "scheduled_release",
            "gdelt_topic_activity": "archive_news_activity",
        }
    )
    broad["source_change"] = np.nan
    broad["expected_crypto_relation"] = "unknown"
    selection_value = pd.to_numeric(broad["selection_value"], errors="coerce")
    selection_threshold = pd.to_numeric(broad["selection_threshold"], errors="coerce")
    broad["severity_multiple"] = selection_value.div(
        selection_threshold.where(selection_threshold.gt(0))
    )

    cross = pd.read_csv(CROSS_CATALOG_PATH)
    cross["anchor_utc"] = pd.to_datetime(cross["anchor_utc"], utc=True)
    cross["source_group"] = "wider_market"
    cross["severity_multiple"] = pd.to_numeric(
        cross["absolute_source_change"], errors="coerce"
    ).div(pd.to_numeric(cross["shock_threshold"], errors="coerce").where(lambda x: x.gt(0)))

    columns = [
        "event_id",
        "event_family",
        "event_label",
        "anchor_utc",
        "source_group",
        "source_change",
        "expected_crypto_relation",
        "severity_multiple",
    ]
    events = pd.concat([broad[columns], cross[columns]], ignore_index=True)
    events["decision_hour_utc"] = events["anchor_utc"].map(decision_hour)
    events["signed_contribution"] = [
        signed_contribution(float(change), str(relation))
        if pd.notna(change)
        else 0
        for change, relation in zip(
            events["source_change"],
            events["expected_crypto_relation"],
            strict=True,
        )
    ]
    return events.sort_values(["decision_hour_utc", "event_id"], kind="stable")


def causal_anchor_surface(events: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for anchor in events["decision_hour_utc"].drop_duplicates().sort_values():
        window = events.loc[
            events["decision_hour_utc"].gt(anchor - pd.Timedelta(hours=LOOKBACK_HOURS))
            & events["decision_hour_utc"].le(anchor)
        ]
        positive = int(window["signed_contribution"].gt(0).sum())
        negative = int(window["signed_contribution"].lt(0).sum())
        severity = pd.to_numeric(window["severity_multiple"], errors="coerce")
        records.append(
            {
                "anchor_utc": anchor,
                "family_count_24h": int(window["event_family"].nunique()),
                "source_group_count_24h": int(window["source_group"].nunique()),
                "positive_signed_count_24h": positive,
                "negative_signed_count_24h": negative,
                "signed_net_24h": positive - negative,
                "max_measurable_severity_multiple_24h": (
                    float(severity.max()) if severity.notna().any() else np.nan
                ),
                "component_event_ids": ";".join(sorted(window["event_id"].astype(str))),
                "component_families": ";".join(
                    sorted(window["event_family"].astype(str).unique())
                ),
                "component_source_groups": ";".join(
                    sorted(window["source_group"].astype(str).unique())
                ),
            }
        )
    return DataFrame.from_records(records)


def route_mask(surface: DataFrame, route_id: str) -> pd.Series:
    if route_id == "two_plus_families":
        return surface["family_count_24h"].ge(2)
    if route_id == "three_plus_families":
        return surface["family_count_24h"].ge(3)
    if route_id == "two_plus_source_groups":
        return surface["source_group_count_24h"].ge(2)
    if route_id == "aligned_positive_market_signs":
        return surface["positive_signed_count_24h"].ge(2) & surface[
            "positive_signed_count_24h"
        ].gt(surface["negative_signed_count_24h"])
    if route_id == "aligned_negative_market_signs":
        return surface["negative_signed_count_24h"].ge(2) & surface[
            "negative_signed_count_24h"
        ].gt(surface["positive_signed_count_24h"])
    if route_id == "isolated_extreme_measurable_shock":
        return surface["family_count_24h"].eq(1) & surface[
            "max_measurable_severity_multiple_24h"
        ].ge(EXTREME_SOURCE_MULTIPLE)
    if route_id == "isolated_family_reference":
        return surface["family_count_24h"].eq(1)
    raise ValueError(f"Unknown route: {route_id}")


def select_with_cooldown(candidates: DataFrame) -> DataFrame:
    selected: list[int] = []
    last_anchor: pd.Timestamp | None = None
    for row in candidates.sort_values("anchor_utc", kind="stable").itertuples():
        anchor = pd.Timestamp(row.anchor_utc)
        if last_anchor is None or anchor - last_anchor >= pd.Timedelta(
            hours=EPISODE_COOLDOWN_HOURS
        ):
            selected.append(int(row.Index))
            last_anchor = anchor
    return candidates.loc[selected].copy()


def build_route_catalog(surface: DataFrame) -> DataFrame:
    frames: list[DataFrame] = []
    for route_id, definition in ROUTES.items():
        selected = select_with_cooldown(surface.loc[route_mask(surface, route_id)])
        selected["route_id"] = route_id
        selected["route_label"] = definition["label"]
        selected["signed_prediction"] = int(definition["prediction"])
        selected["route_event_id"] = route_id + "_" + selected["anchor_utc"].map(
            lambda value: pd.Timestamp(value).strftime("%Y_%m_%d_%H")
        )
        selected["whole_event_partition"] = selected["anchor_utc"].map(
            layer1.whole_event_partition
        )
        frames.append(selected)
    catalog = pd.concat(frames, ignore_index=True)
    if catalog["route_event_id"].duplicated().any():
        raise ValueError("Confluence route event identifiers are not unique.")
    return catalog.sort_values(["route_id", "anchor_utc"], kind="stable")


def causal_btc_prestate() -> DataFrame:
    frame = g0.load_ohlcv(g0.ohlcv_path("BTC/USDT:USDT", "1h"))
    frame = frame.sort_values("date", kind="stable").drop_duplicates("date")
    returns = frame["close"].pct_change()
    true_range = pd.concat(
        [
            frame["high"].sub(frame["low"]),
            frame["high"].sub(frame["close"].shift(1)).abs(),
            frame["low"].sub(frame["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pre = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)
    local = layer2_direct.local_context(frame, atr_pre)
    pre_return = frame["close"].shift(1).div(frame["close"].shift(721)).sub(1)
    pre_volatility = returns.shift(1).rolling(720, min_periods=360).std()
    scaled = pre_return.div(pre_volatility.mul(np.sqrt(720)).replace(0, np.nan))
    background = np.select(
        [scaled.le(-1), scaled.lt(0), scaled.lt(1)],
        ["strongly_negative", "mildly_negative", "mildly_positive"],
        default="strongly_positive",
    )
    prior_high = frame["high"].shift(1).rolling(720, min_periods=360).max()
    prior_low = frame["low"].shift(1).rolling(720, min_periods=360).min()
    near_high = frame["open"].sub(prior_high).abs().div(atr_pre).le(0.5)
    near_low = frame["open"].sub(prior_low).abs().div(atr_pre).le(0.5)
    range_location = np.select(
        [near_low, near_high], ["near_prior_30d_low", "near_prior_30d_high"], default="middle"
    )
    return DataFrame(
        {
            "anchor_utc": frame["date"],
            "pre_btc_return_30d": pre_return,
            "pre_btc_scaled_background": scaled,
            "background_state": background,
            "local_level_state": local["local_level_state"],
            "nearest_level_family": local["nearest_level_family"],
            "nearest_level_distance_atr": local["nearest_level_distance_atr"],
            "prior_range_location": range_location,
        }
    )


def attach_prestate(catalog: DataFrame) -> DataFrame:
    output = catalog.merge(causal_btc_prestate(), on="anchor_utc", how="left")
    if output["pre_btc_return_30d"].isna().any():
        missing = int(output["pre_btc_return_30d"].isna().sum())
        raise ValueError(f"Missing causal BTC prestate for {missing} route rows.")
    return output


def count_table(catalog: DataFrame) -> DataFrame:
    return (
        catalog.groupby(
            ["route_id", "route_label", "whole_event_partition"], dropna=False
        )
        .size()
        .rename("episodes")
        .reset_index()
        .sort_values(["route_id", "whole_event_partition"])
    )


def build_freeze(catalog: DataFrame, counts: DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_event_confluence_before_crypto_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "plain_theory": (
            "Several individually modest influences may matter when they overlap, and "
            "their effect may depend on the market background and technical location. "
            "One sufficiently large shock may also matter alone."
        ),
        "routes": ROUTES,
        "lookback_hours": LOOKBACK_HOURS,
        "episode_cooldown_hours": EPISODE_COOLDOWN_HOURS,
        "horizons_hours": list(HORIZONS_HOURS),
        "outcomes": [
            "absolute price movement",
            "price range / volatility",
            "traded volume",
            "signed volume-weighted candle pressure",
            "direction where a signed wider-market consensus exists",
        ],
        "descriptive_modifiers_not_separate_branches": [
            "strong or mild positive/negative trailing BTC background",
            "single calculated level, calculated level cluster, or far from levels",
            "near prior 30-day high, near prior 30-day low, or middle",
        ],
        "screening_rules": {
            "minimum_events_in_each_development_and_validation_partition": (
                MINIMUM_PARTITION_EVENTS
            ),
            "activity_fraction_above_control": MINIMUM_ABOVE_CONTROL_RATE,
            "median_activity_multiple": MINIMUM_MEDIAN_ACTIVITY_SCORE,
            "overlap_must_exceed_isolated_reference": True,
            "direction_accuracy": MINIMUM_DIRECTION_ACCURACY,
            "direction_lift_over_same_clock_controls": (
                MINIMUM_DIRECTION_LIFT_VS_CONTROLS
            ),
            "joint_reaction_and_direction_floor": JOINT_LEAD_FLOOR,
            "joint_main_target": JOINT_MAIN_TARGET,
        },
        "limits": [
            "Archive-news families have no reliable positive/negative story sign.",
            "Scheduled occurrence has no direction without historical expectations.",
            "Signed calls use only wider-market changes whose direction was known.",
            "Orderbook/liquidity is coverage-parked rather than filled or guessed.",
            "All routes complete before any descendant test is designed.",
        ],
        "episode_counts": counts.to_dict(orient="records"),
        "artifacts": {
            "broad_source_catalog": artifact(BROAD_CATALOG_PATH),
            "cross_asset_source_catalog": artifact(CROSS_CATALOG_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }


def render_report(freeze: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Event Confluence Freeze",
            "",
            f"- Status: `{freeze['status']}`",
            "- Crypto outcomes read: **No**",
            "- Overlap routes plus controls: `7`",
            "",
            str(freeze["plain_theory"]),
            "",
            "Positive/negative archive-story labels are not available and will not be guessed.",
            "All sibling routes must finish before a follow-up is selected.",
            "",
        ]
    )


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "event_confluence_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_confluence_freeze":
            raise ValueError("Existing confluence freeze is not terminal.")
        return result
    events = load_source_events()
    surface = causal_anchor_surface(events)
    catalog = attach_prestate(build_route_catalog(surface))
    counts = count_table(catalog)
    freeze = build_freeze(catalog, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_path = OUTPUT_ROOT / "combined_source_event_catalog.csv"
    surface_path = OUTPUT_ROOT / "causal_confluence_anchor_surface.csv"
    catalog_path = OUTPUT_ROOT / "event_confluence_route_catalog.csv"
    counts_path = OUTPUT_ROOT / "event_confluence_counts.csv"
    freeze_path = OUTPUT_ROOT / "event_confluence_freeze.json"
    report_path = OUTPUT_ROOT / "event_confluence_freeze_report.md"
    g0.atomic_write_csv(events, source_path)
    g0.atomic_write_csv(surface, surface_path)
    g0.atomic_write_csv(catalog, catalog_path)
    g0.atomic_write_csv(counts, counts_path)
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_event_confluence_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "route_count": int(catalog["route_id"].nunique()),
        "route_rows": len(catalog),
        "artifacts": {
            "freeze": artifact(freeze_path),
            "sources": artifact(source_path),
            "surface": artifact(surface_path),
            "catalog": artifact(catalog_path),
            "counts": artifact(counts_path),
            "report": artifact(report_path),
        },
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "routes": list(ROUTES),
                    "outcomes_will_be_read": False,
                    "output_root": str(OUTPUT_ROOT),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
