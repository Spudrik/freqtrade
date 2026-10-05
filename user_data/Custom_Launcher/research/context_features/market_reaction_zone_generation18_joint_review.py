"""Jointly review every frozen Generation 18 sibling after all are terminal."""

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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_timeframe_portability as g18t,
)


DEFAULT_REVIEW_ID = "g18_joint_review_20260823a"
REVIEW_ROOT = g18z.OUTPUT_ROOT / "joint_review"
DIRECT_RESULT_PATH = (
    g18d.RECORD_ROOT / g18d.DEFAULT_RUN_ID / "g18_direct_confirmation_result.json"
)
TIMEFRAME_RESULT_PATH = (
    g18t.RECORD_ROOT / g18t.DEFAULT_RUN_ID / "g18_timeframe_portability_result.json"
)


def artifact(path: Path) -> dict[str, Any]:
    return g18z.artifact(path)


def load_json(path: Path, expected_status: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != expected_status:
        raise ValueError(
            f"Expected {expected_status!r} in {path}, got {value.get('status')!r}."
        )
    return value


def load_inputs() -> dict[str, Any]:
    freeze = load_json(g18z.FREEZE_PATH, "frozen_before_generation18_outcomes")
    direct = load_json(
        DIRECT_RESULT_PATH, "completed_generation18_direct_confirmation"
    )
    timeframe = load_json(
        TIMEFRAME_RESULT_PATH, "completed_generation18_timeframe_portability"
    )
    frozen_statuses = {item["branch_id"]: item["status_at_freeze"] for item in freeze["branches"]}
    completed = set(direct["branches_completed"]) | {timeframe["branch_id"]}
    expected_active = {
        branch_id
        for branch_id, status in frozen_statuses.items()
        if status == "frozen_pending"
    }
    parked = {
        branch_id
        for branch_id, status in frozen_statuses.items()
        if status.startswith("parked")
    }
    if completed != expected_active:
        raise ValueError(
            "Generation 18 active siblings are incomplete: "
            f"expected {expected_active}, got {completed}."
        )
    if parked != {"g18f_direction_after_reaction_gate"}:
        raise ValueError(f"Unexpected Generation 18 parked siblings: {parked}")
    return {"freeze": freeze, "direct": direct, "timeframe": timeframe}


def read_decisions() -> dict[str, DataFrame]:
    direct_dir = g18d.RECORD_ROOT / g18d.DEFAULT_RUN_ID
    timeframe_dir = g18t.RECORD_ROOT / g18t.DEFAULT_RUN_ID
    return {
        "density": pd.read_csv(direct_dir / "g18_density_decisions.csv"),
        "semantics": pd.read_csv(direct_dir / "g18_semantic_decisions.csv"),
        "levels": pd.read_csv(direct_dir / "g18_level_decisions.csv"),
        "context": pd.read_csv(direct_dir / "g18_context_decisions.csv"),
        "timeframe_control": pd.read_csv(
            timeframe_dir / "g18_timeframe_control_decisions.csv"
        ),
        "timeframe_head": pd.read_csv(
            timeframe_dir / "g18_timeframe_head_decisions.csv"
        ),
        "timeframe_cluster": pd.read_csv(
            timeframe_dir / "g18_timeframe_cluster_decisions.csv"
        ),
    }


def strict(frame: DataFrame) -> DataFrame:
    return frame.loc[frame["status"].eq("strict_holdout_confirmation")].copy()


def portable_questions(frame: DataFrame, keys: Sequence[str]) -> DataFrame:
    retained = strict(frame)
    rows: list[dict[str, Any]] = []
    for key, cell in retained.groupby(list(keys), observed=True, sort=False):
        scopes = set(cell["market_scope"].astype(str))
        if {"all_normal", "top_ten_memes"}.issubset(scopes):
            values = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "normal_and_meme_strict": True,
                }
            )
    return DataFrame.from_records(rows)


def valid_timeframe_precedence(decisions: dict[str, DataFrame]) -> DataFrame:
    head = strict(decisions["timeframe_head"])
    controls = strict(decisions["timeframe_control"])
    controls = controls.loc[controls["scope_kind"].eq("family_timeframe")].copy()
    keys_left = [
        "scope_value",
        "candidate_timeframe",
        "metric",
        "horizon_hours",
        "market_scope",
    ]
    keys_right = [
        "scope_value",
        "source_timeframe",
        "metric",
        "horizon_hours",
        "market_scope",
    ]
    return head.merge(
        controls,
        left_on=keys_left,
        right_on=keys_right,
        how="inner",
        suffixes=("_vs_1h", "_vs_controls"),
        validate="one_to_one",
    )


def valid_cluster_superiority(decisions: dict[str, DataFrame]) -> DataFrame:
    cluster = strict(decisions["timeframe_cluster"])
    controls = strict(decisions["timeframe_control"])
    controls = controls.loc[
        controls["scope_kind"].eq("same_family_cross_timeframe_cluster")
    ].copy()
    keys = ["scope_value", "metric", "horizon_hours", "market_scope"]
    return cluster.merge(
        controls,
        on=keys,
        how="inner",
        suffixes=("_vs_isolated", "_vs_controls"),
        validate="one_to_one",
    )


def branch_statuses(inputs: dict[str, Any]) -> DataFrame:
    direct_counts = inputs["direct"]["strict_holdout_question_rows"]
    timeframe = inputs["timeframe"]
    return DataFrame.from_records(
        [
            {
                "branch_id": "g18a_density_geometry_confirmation",
                "status": "completed_narrow_btc_only",
                "strict_rows": direct_counts["density"],
                "plain_result": (
                    "Only BTC proximity retained a density-style relationship; gradual "
                    "cross-market density did not confirm on the later blocks."
                ),
            },
            {
                "branch_id": "g18b_reaction_form_confirmation",
                "status": "completed_retained",
                "strict_rows": direct_counts["semantics"],
                "plain_result": (
                    "Any recross and repeated recross repeated broadly for adaptive VP "
                    "and the combined VP/rolling-boundary surface."
                ),
            },
            {
                "branch_id": "g18c_level_source_confirmation",
                "status": "completed_conditional",
                "strict_rows": direct_counts["levels"],
                "plain_result": (
                    "Exact-location confirmation narrowed to BTC VP crossings and meme "
                    "rolling-boundary/LVN reaction cells; no universal parameter plateau."
                ),
            },
            {
                "branch_id": "g18d_context_and_market_regimes",
                "status": "completed_retained_activity_context",
                "strict_rows": direct_counts["context"],
                "plain_result": (
                    "Crypto dispersion and local activity changed extra volume around "
                    "levels in broad or coherent groups; most other rows were BTC-only."
                ),
            },
            {
                "branch_id": "g18e_coin_group_and_timeframe_portability",
                "status": "completed_no_precedence_or_cluster_claim",
                "strict_rows": timeframe["strict_control_rows"],
                "plain_result": (
                    "Several 1h/group-specific sources beat controls. No higher-timeframe "
                    "or cross-timeframe cluster row beat every required baseline."
                ),
            },
            {
                "branch_id": "g18f_direction_after_reaction_gate",
                "status": "parked_current_rules_failed",
                "strict_rows": 0,
                "plain_result": (
                    "No signed run launched because Generation 17 had no method at the "
                    "55% joint reaction-plus-direction floor."
                ),
            },
        ]
    )


def next_branch_queue(
    *,
    portable_recross_count: int,
    broad_context_count: int,
) -> dict[str, Any]:
    siblings = [
        {
            "branch_id": "g19a_exact_coordinate_recross_specificity",
            "status": "queued",
            "parent_evidence_count": portable_recross_count,
            "question": (
                "Does the exact current coordinate explain broad recrossing after "
                "current, stale, shifted, ordinary-time, and near-miss areas are compared "
                "using the same recross definitions?"
            ),
        },
        {
            "branch_id": "g19b_regime_level_incremental_combinations",
            "status": "queued",
            "parent_evidence_count": broad_context_count,
            "question": (
                "Do crypto dispersion or local completed activity add unseen volume-"
                "reaction information beyond level-only and context-only descriptions?"
            ),
        },
        {
            "branch_id": "g19c_market_specific_level_mechanisms",
            "status": "queued",
            "parent_evidence_count": 3,
            "question": (
                "Do meme rolling boundaries/LVNs, established-coin rolling VWAP, and BTC "
                "VP behaviour repeat under their separate predeclared market labels?"
            ),
        },
        {
            "branch_id": "g19d_recross_timing_and_activity_onset",
            "status": "queued",
            "parent_evidence_count": portable_recross_count,
            "question": (
                "Is repeated traffic newly triggered after contact, already active before "
                "contact, or a persistent acceptance process with measurable timing?"
            ),
        },
        {
            "branch_id": "g19e_rational_acceptance_zone_extension",
            "status": "queued",
            "parent_evidence_count": 1,
            "question": (
                "Can a compact causal repeated-acceptance or event-anchored VWAP area "
                "improve location evidence beyond rolling VWAP and ordinary controls?"
            ),
        },
        {
            "branch_id": "g19f_external_context_accumulation",
            "status": "parked_coverage",
            "parent_evidence_count": 0,
            "question": (
                "Do orderbook, historical news, or non-crypto global sources add value "
                "once two complete timestamp-ready confirmation blocks exist?"
            ),
        },
        {
            "branch_id": "g19g_direction_after_reaction_gate",
            "status": "parked_current_rules_failed",
            "parent_evidence_count": 0,
            "question": (
                "Can a newly frozen one-minute sample reach 55% joint reaction and "
                "direction only after an exact reaction gate is confirmed?"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "generation": 19,
        "branch_layer": 3,
        "created_at_utc": g0.utc_now(),
        "status": "queued_after_complete_generation18_joint_review",
        "automatic_launch": False,
        "no_descendant_before_all_siblings_terminal_and_jointly_reviewed": True,
        "siblings": siblings,
    }


def markdown_table(frame: DataFrame, columns: Sequence[str], limit: int = 30) -> str:
    if frame.empty:
        return "No rows met this joint condition."
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].head(
        limit
    )
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    *,
    branches: DataFrame,
    portable_semantics: DataFrame,
    levels: DataFrame,
    context: DataFrame,
    valid_precedence: DataFrame,
    valid_clusters: DataFrame,
    queue: dict[str, Any],
) -> str:
    broad_context = context.loc[
        context["market_scope"].isin(
            ["all_normal", "smart_contract_platforms", "other_established_alts"]
        )
    ]
    level_table = markdown_table(
        levels,
        [
            "scope_kind",
            "scope_value",
            "metric",
            "horizon_hours",
            "market_scope",
            "minimum_equal_coin_difference",
        ],
        limit=20,
    )
    context_table = markdown_table(
        broad_context,
        [
            "context_id",
            "metric",
            "horizon_hours",
            "market_scope",
            "minimum_equal_coin_difference",
        ],
        limit=20,
    )
    return f"""# Generation 18 joint review

## Batch integrity

All five active siblings completed and the direction sibling remained honestly parked
before this review. The confirmation data consisted of two later blocks for normal coins
and two later blocks for the frozen top-ten meme cohort. Profit and signed future
direction were not used.

## What repeated most broadly

The clearest repeatable behaviour is **recrossing**. After price reached adaptive Volume
Profile nodes or the combined Volume Profile/rolling-high-low surface, it crossed the
area again more often than after matched ordinary times and genuine near misses. The
same questions repeated in normal coins and memes over several horizons.

{markdown_table(portable_semantics, ["scope_kind", "scope_value", "metric", "horizon_hours"])}

This is a busy-location result. It does not say bounce, breakout, up, down, or profit.

## Exact level location

When stale and price-shifted versions were also required, the broad result narrowed.
BTC retained adaptive Volume Profile crossing cells. Meme coins retained a rolling
high/low unsigned-reaction cell and one exact 72-hour/96-bin low-volume-node crossing
cell. No neighbouring Volume Profile parameter plateau confirmed across both normal and
meme cohorts, so one exact setting is not promoted.

{level_table}

## Market and indicator context

High cross-coin dispersion produced a larger extra volume response around levels than
low dispersion in the all-normal group over two hours. High completed local volume/range
activity did the same over one hour in the predeclared smart-contract-platform group.
Most other strict context rows were BTC-only, including Bollinger-width state, RSI
extremes, and wider-crypto activity. These are activity modifiers, not direction calls.

{context_table}

## Higher timeframes and clusters

Some higher-timeframe contacts beat 1h contacts, and some cross-timeframe clusters beat
isolated contacts, when viewed only head to head. None of those same rows also completed
the required artificial-location control ladder. Therefore this generation establishes
neither higher-timeframe precedence nor cluster superiority.

- Higher-timeframe rows that beat both 1h and all location controls: {len(valid_precedence)}
- Cluster rows that beat both isolated contacts and all location controls: {len(valid_clusters)}

Single levels remain as valid as clusters. The result simply says the tested clusters
did not add proven information beyond their components in this batch.

## Direction boundary

No directional model ran. Generation 17's best result was 35% joint reaction plus
direction, below the 55% floor. Reaction-only and volume results do not change that.

## Branch decisions

{markdown_table(branches, ["branch_id", "status", "plain_result"], limit=10)}

## Next complete sibling layer

""" + "\n".join(
        f"- {item['branch_id']} — {item['status']}: {item['question']}"
        for item in queue["siblings"]
    ) + """

No early Generation 19 result may launch a descendant before every sibling is terminal
or honestly parked and the whole layer is jointly reviewed.
"""


def main() -> int:
    inputs = load_inputs()
    decisions = read_decisions()
    branches = branch_statuses(inputs)
    portable_semantics = portable_questions(
        decisions["semantics"],
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
    )
    level_strict = strict(decisions["levels"])
    context_strict = strict(decisions["context"])
    valid_precedence = valid_timeframe_precedence(decisions)
    valid_clusters = valid_cluster_superiority(decisions)
    broad_context_count = int(
        context_strict["market_scope"]
        .isin(["all_normal", "smart_contract_platforms", "other_established_alts"])
        .sum()
    )
    queue = next_branch_queue(
        portable_recross_count=len(portable_semantics),
        broad_context_count=broad_context_count,
    )
    run_dir = REVIEW_ROOT / DEFAULT_REVIEW_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "branch_statuses": run_dir / "g18_branch_statuses.csv",
        "portable_semantics": run_dir / "g18_portable_semantics.csv",
        "strict_levels": run_dir / "g18_strict_level_rows.csv",
        "strict_context": run_dir / "g18_strict_context_rows.csv",
        "valid_precedence": run_dir / "g18_valid_timeframe_precedence.csv",
        "valid_clusters": run_dir / "g18_valid_cluster_superiority.csv",
        "branch_queue": run_dir / "g19_sibling_branch_queue.json",
        "report": run_dir / "g18_joint_review.md",
        "result": run_dir / "g18_joint_review.json",
    }
    for name, frame in (
        ("branch_statuses", branches),
        ("portable_semantics", portable_semantics),
        ("strict_levels", level_strict),
        ("strict_context", context_strict),
        ("valid_precedence", valid_precedence),
        ("valid_clusters", valid_clusters),
    ):
        g0.atomic_write_csv(frame, paths[name])
    g0.atomic_write_json(queue, paths["branch_queue"])
    report = build_report(
        branches=branches,
        portable_semantics=portable_semantics,
        levels=level_strict,
        context=context_strict,
        valid_precedence=valid_precedence,
        valid_clusters=valid_clusters,
        queue=queue,
    )
    paths["report"].write_text(report, encoding="utf-8")
    result = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation18_joint_review",
        "all_siblings_terminal_or_parked_before_review": True,
        "summary": {
            "portable_normal_and_meme_recross_questions": len(portable_semantics),
            "strict_exact_level_rows": len(level_strict),
            "strict_broad_or_group_context_rows": broad_context_count,
            "valid_higher_timeframe_precedence_rows": len(valid_precedence),
            "valid_cluster_superiority_rows": len(valid_clusters),
            "joint_reaction_and_direction_at_or_above_55pct": False,
        },
        "main_conclusion": (
            "Calculated areas robustly locate repeated traffic. Exact coordinate, "
            "market scope, and context effects are conditional; direction remains unproven."
        ),
        "source_contracts": {
            "generation18_freeze": artifact(g18z.FREEZE_PATH),
            "direct_confirmation": artifact(DIRECT_RESULT_PATH),
            "timeframe_portability": artifact(TIMEFRAME_RESULT_PATH),
        },
        "next_branch_layer": 3,
        "next_siblings": len(queue["siblings"]),
        "artifacts": {
            name: artifact(path)
            for name, path in paths.items()
            if name != "result"
        },
        "result_path": str(paths["result"].resolve()),
    }
    g0.atomic_write_json(result, paths["result"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
