"""Jointly review all six frozen Generation 17 sibling branches."""

from __future__ import annotations

# Repository-local imports follow the root-path bootstrap.
# ruff: noqa: E402
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation17 as g17f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_crossing_semantics as g17s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freqai_cache as g17c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_one_minute_analysis as g17a,
)


DEFAULT_REVIEW_ID = "g17_joint_review_20260823a"
REVIEW_ROOT = g17z.OUTPUT_ROOT / "joint_review"


def artifact(path: Path) -> dict[str, Any]:
    return g17z.artifact(path)


def load_json(path: Path, status: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != status:
        raise ValueError(f"Expected {status!r} in {path}, got {value.get('status')!r}.")
    return value


def load_inputs() -> dict[str, Any]:
    freeze = load_json(g17z.FREEZE_PATH, "frozen_before_generation17_outcomes")
    freqai_path = (
        g17f.RECORD_ROOT
        / g17f.DEFAULT_JOINT_REVIEW_ID
        / "g17_freqai_joint_review.json"
    )
    semantics_path = (
        g17s.RECORD_ROOT
        / g17s.DEFAULT_RUN_ID
        / "g17_crossing_semantics_result.json"
    )
    atlas_path = (
        g17l.RECORD_ROOT
        / g17l.DEFAULT_RUN_ID
        / "g17_level_source_atlas_result.json"
    )
    minute_path = (
        g17a.REPORT_ROOT
        / g17a.DEFAULT_RUN_ID
        / "g17_one_minute_analysis_record.json"
    )
    return {
        "freeze": freeze,
        "freqai": load_json(freqai_path, "completed_generation17_freqai_joint_review"),
        "semantics": load_json(
            semantics_path, "completed_generation17_crossing_semantics"
        ),
        "atlas": load_json(atlas_path, "completed_generation17_level_source_atlas"),
        "minute": load_json(
            minute_path, "completed_generation17_one_minute_direction_diagnostic"
        ),
        "paths": {
            "freeze": g17z.FREEZE_PATH,
            "freqai": freqai_path,
            "semantics": semantics_path,
            "atlas": atlas_path,
            "minute": minute_path,
        },
    }


def external_coverage() -> DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        manifest = json.loads(
            (g13c.RECORD_ROOT / f"{cohort}_manifest.json").read_text(encoding="utf-8")
        )
        for item in manifest["inventory"]:
            support = pd.read_parquet(
                item["mtf_support_path"],
                columns=[
                    "date",
                    "period",
                    "ready__g11_news_context",
                    "ready__g11_orderbook_context",
                ],
            )
            for period, frame in support.groupby("period", observed=True, sort=False):
                for source, column in (
                    ("historical_news", "ready__g11_news_context"),
                    ("btc_orderbook", "ready__g11_orderbook_context"),
                ):
                    ready = frame[column].fillna(False).astype(bool)
                    rows.append(
                        {
                            "cohort": cohort,
                            "pair": item["pair"],
                            "period": period,
                            "source": source,
                            "rows": len(frame),
                            "ready_rows": int(ready.sum()),
                            "ready_fraction": float(ready.mean()) if len(frame) else 0.0,
                        }
                    )
    coverage = pd.DataFrame.from_records(rows)
    global_rows = (
        coverage.loc[coverage["source"].eq("historical_news")]
        .assign(
            source="non_crypto_global_markets",
            ready_rows=0,
            ready_fraction=0.0,
        )
        .copy()
    )
    return pd.concat([coverage, global_rows], ignore_index=True)


def freqai_decisions(result: dict[str, Any]) -> DataFrame:
    frame = pd.read_csv(result["artifacts"]["joint_decisions"]["path"])
    registry = json.loads(g17c.REGISTRY_PATH.read_text(encoding="utf-8"))
    branch_by_route = {
        item["route_id"]: item["branch_id"] for item in registry["comparisons"]
    }
    frame["branch_id"] = frame["route_id"].map(branch_by_route)
    if frame["branch_id"].isna().any():
        raise ValueError("A Generation 17 FreqAI route lacks a frozen branch mapping.")
    return frame


def branch_statuses(inputs: dict[str, Any]) -> DataFrame:
    statuses = [
        {
            "branch_id": "g17a_density_geometry_decomposition",
            "status": "completed",
            "evidence": inputs["paths"]["freqai"],
        },
        {
            "branch_id": "g17b_crossing_semantics",
            "status": "completed",
            "evidence": inputs["paths"]["semantics"],
        },
        {
            "branch_id": "g17c_reaction_probability_calibration",
            "status": "completed",
            "evidence": inputs["paths"]["freqai"],
        },
        {
            "branch_id": "g17d_broad_level_sources",
            "status": "completed",
            "evidence": inputs["paths"]["atlas"],
        },
        {
            "branch_id": "g17e_external_context_regimes",
            "status": "completed_orderbook_news_and_global_parked_for_coverage",
            "evidence": inputs["paths"]["freqai"],
        },
        {
            "branch_id": "g17f_bounded_one_minute_expansion",
            "status": "completed",
            "evidence": inputs["paths"]["minute"],
        },
    ]
    return DataFrame.from_records(statuses)


def next_branch_queue(
    freqai: DataFrame,
    semantics: DataFrame,
    atlas: DataFrame,
    minute: DataFrame,
) -> list[dict[str, Any]]:
    strict_freqai = int(freqai["strict_all_three_cells"].astype(bool).sum())
    strict_semantics = int(semantics["strict_repeated"].astype(bool).sum())
    strict_atlas = int(atlas["strict_repeated"].astype(bool).sum())
    minute_leads = int(
        minute["broad_point_lead"].astype(bool).sum()
        + minute["cohort_specific_point_lead"].astype(bool).sum()
    )
    return [
        {
            "branch_id": "g18a_density_geometry_confirmation",
            "status": "queued" if strict_freqai else "parked_no_strict_parent",
            "question": (
                "Do the surviving density/geometry components retain their gain under "
                "new time blocks, feature ablation, and parameter-free monotonic checks?"
            ),
            "parent_evidence_count": strict_freqai,
        },
        {
            "branch_id": "g18b_reaction_form_confirmation",
            "status": "queued" if strict_semantics else "parked_no_strict_parent",
            "question": (
                "Which repeated semantics—recross, traversal, rejection, breakthrough, "
                "or dwell—remain above both ordinary-time and near-miss controls?"
            ),
            "parent_evidence_count": strict_semantics,
        },
        {
            "branch_id": "g18c_level_source_confirmation",
            "status": "queued" if strict_atlas else "parked_no_strict_parent",
            "question": (
                "Do atlas finalists survive later chronology and nearby rational parameter "
                "settings, without choosing a parameter from a known event?"
            ),
            "parent_evidence_count": strict_atlas,
        },
        {
            "branch_id": "g18d_context_and_market_regimes",
            "status": "queued",
            "question": (
                "Do quiet/trending/choppy wider-market regimes or source-ready order-book "
                "state change reaction likelihood beyond the retained level information?"
            ),
            "parent_evidence_count": int(
                freqai["route_id"].eq("orderbook_current_increment").sum()
            ),
        },
        {
            "branch_id": "g18e_coin_group_and_timeframe_portability",
            "status": "queued",
            "question": (
                "Which leads are shared by similar coin groups, memes, BTC separately, and "
                "1h/4h/8h calculations rather than being one-coin coincidences?"
            ),
            "parent_evidence_count": strict_freqai + strict_semantics + strict_atlas,
        },
        {
            "branch_id": "g18f_direction_after_reaction_gate",
            "status": "queued_for_new_sample" if minute_leads else "parked_current_rules_failed",
            "question": (
                "Only after a reaction gate is available, can a newly frozen one-minute "
                "sample predict both reaction and direction at 55% or more?"
            ),
            "parent_evidence_count": minute_leads,
        },
    ]


def friendly_target(value: str) -> str:
    return (
        value.replace("&-g17_", "")
        .replace("_h", " over ")
        .replace("_", " ")
        + " hours"
    )


def markdown_table(frame: DataFrame, columns: Sequence[str]) -> list[str]:
    if frame.empty:
        return ["None."]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.loc[:, list(columns)].iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return rows


def build_report(
    *,
    freqai: DataFrame,
    semantics: DataFrame,
    atlas: DataFrame,
    minute_summary: dict[str, Any],
    queue: Sequence[dict[str, Any]],
) -> str:
    strict_freqai = freqai.loc[freqai["strict_all_three_cells"].astype(bool)].copy()
    strict_freqai["target_plain"] = strict_freqai["target"].map(friendly_target)
    strict_semantics = semantics.loc[semantics["strict_repeated"].astype(bool)].copy()
    strict_atlas = atlas.loc[atlas["strict_repeated"].astype(bool)].copy()
    best = minute_summary["best_observed_causal_method"]
    lines = [
        "# Generation 17 joint review",
        "",
        "## What was tested",
        "",
        (
            "All six sibling batches were frozen before their outcomes were reviewed. "
            "The work covered normal coins, the top-ten meme cohort, older and recent "
            "normal chronology, 1h/2h/4h reaction outcomes, and 15m/60m/240m/720m "
            "one-minute direction outcomes. Profit was never used."
        ),
        "",
        "## What strict means here",
        "",
        (
            "A strict result is not a promise about the next trade. It means the same "
            "question beat every frozen simpler or artificial control in every required "
            "unseen period and coin cohort, with uncertainty and one-coin dominance checks."
        ),
        "",
        "## FreqAI component results",
        "",
    ]
    lines.extend(
        markdown_table(
            strict_freqai,
            [
                "branch_id",
                "route_id",
                "target_plain",
                "minimum_equal_coin_paired_mae_gain",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Direct crossing-semantics results",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            strict_semantics,
            [
                "metric",
                "horizon_hours",
                "market_scope",
                "effect_sign",
                "minimum_effect",
                "maximum_effect",
            ],
        )
    )
    lines.extend(["", "## Rational level-source atlas", ""])
    lines.extend(
        markdown_table(
            strict_atlas.head(40),
            [
                "scope_kind",
                "scope_value",
                "metric",
                "horizon_hours",
                "market_scope",
                "minimum_equal_coin_difference",
            ],
        )
    )
    if len(strict_atlas) > 40:
        lines.extend(["", f"The table shows 40 of {len(strict_atlas)} strict atlas rows."])
    lines.extend(
        [
            "",
            "## One-minute reaction plus direction",
            "",
            (
                f"No method reached 55%. The best causal method was "
                f"{best.get('method')} at {best.get('horizon_minutes')} minutes: "
                f"{best.get('joint_successes')}/{best.get('independent_episodes')} "
                "joint successes "
                f"({100.0 * float(best.get('joint_success_rate_all_episodes', 0.0)):.1f}%)."
            ),
            (
                "Its conditional direction percentage is reported separately because it "
                "removes non-reacting or non-callable cases. It is not the 55% target."
            ),
            "",
            "## Overall conclusion",
            "",
            (
                "Calculated areas contain repeatable information about where activity "
                "and traffic occur. That is useful location evidence. This generation "
                "still found no method that predicts both whether a reaction occurs and "
                "its direction at 55% or more. Direction remains unproven."
            ),
            "",
            "## Next sibling batch",
            "",
        ]
    )
    for item in queue:
        lines.append(
            f"- {item['branch_id']} — {item['status']}: {item['question']}"
        )
    lines.extend(
        [
            "",
            (
                "These are one Generation 18 sibling layer. No result from an early "
                "Generation 18 test may create a Generation 19 descendant until every "
                "Generation 18 sibling is completed or honestly parked and jointly reviewed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_review(review_id: str) -> dict[str, Any]:
    inputs = load_inputs()
    freqai = freqai_decisions(inputs["freqai"])
    semantics = pd.read_csv(inputs["semantics"]["artifacts"]["joint_decisions"]["path"])
    atlas = pd.read_csv(inputs["atlas"]["artifacts"]["whole_decisions"]["path"])
    minute = pd.read_csv(inputs["minute"]["artifacts"]["method_decisions"]["path"])
    coverage = external_coverage()
    statuses = branch_statuses(inputs)
    if len(statuses) != 6 or not statuses["status"].str.startswith(
        ("completed", "parked")
    ).all():
        raise ValueError("All six Generation 17 siblings must be terminal before review.")
    queue = next_branch_queue(freqai, semantics, atlas, minute)
    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "branch_statuses": review_dir / "g17_branch_statuses.csv",
        "freqai_decisions": review_dir / "g17_freqai_decisions.csv",
        "semantic_decisions": review_dir / "g17_semantic_decisions.csv",
        "atlas_decisions": review_dir / "g17_atlas_decisions.csv",
        "minute_decisions": review_dir / "g17_minute_decisions.csv",
        "external_coverage": review_dir / "g17_external_coverage.csv",
        "branch_queue": review_dir / "g18_sibling_branch_queue.json",
        "report": review_dir / "g17_joint_review.md",
    }
    g0.atomic_write_csv(statuses, paths["branch_statuses"])
    g0.atomic_write_csv(freqai, paths["freqai_decisions"])
    g0.atomic_write_csv(semantics, paths["semantic_decisions"])
    g0.atomic_write_csv(atlas, paths["atlas_decisions"])
    g0.atomic_write_csv(minute, paths["minute_decisions"])
    g0.atomic_write_csv(coverage, paths["external_coverage"])
    g0.atomic_write_json(
        {
            "schema_version": 1,
            "generation": 18,
            "created_at_utc": g0.utc_now(),
            "status": "queued_after_complete_generation17_joint_review",
            "branch_layer": 2,
            "siblings": queue,
            "no_descendant_before_all_siblings_terminal_and_jointly_reviewed": True,
            "automatic_launch": False,
        },
        paths["branch_queue"],
    )
    report = build_report(
        freqai=freqai,
        semantics=semantics,
        atlas=atlas,
        minute_summary=inputs["minute"]["summary"],
        queue=queue,
    )
    paths["report"].write_text(report, encoding="utf-8")
    strict_freqai = int(freqai["strict_all_three_cells"].astype(bool).sum())
    strict_semantics = int(semantics["strict_repeated"].astype(bool).sum())
    strict_atlas = int(atlas["strict_repeated"].astype(bool).sum())
    result = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation17_joint_review",
        "all_six_siblings_terminal_before_review": True,
        "summary": {
            "freqai_strict_questions_targets": strict_freqai,
            "direct_semantic_strict_questions": strict_semantics,
            "atlas_strict_questions": strict_atlas,
            "one_minute_methods_at_or_above_55pct": int(
                minute["broad_point_lead"].astype(bool).sum()
                + minute["cohort_specific_point_lead"].astype(bool).sum()
            ),
            "joint_reaction_and_direction_at_or_above_55pct": False,
        },
        "main_conclusion": (
            "Calculated areas carry repeatable direction-neutral location and traffic "
            "information. No Generation 17 method reached the 55% joint reaction-plus-"
            "direction floor."
        ),
        "external_source_status": {
            "orderbook": "tested_on_source-ready rows",
            "historical_news": "parked_zero_ready_rows",
            "non_crypto_global_markets": "parked_no_historical_feature_block",
        },
        "next_branch_layer": 2,
        "next_siblings": len(queue),
        "source_contracts": {
            key: artifact(path) for key, path in inputs["paths"].items()
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = review_dir / "g17_joint_review.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-id", default=DEFAULT_REVIEW_ID)
    args = parser.parse_args(argv)
    result = run_review(str(args.review_id))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
