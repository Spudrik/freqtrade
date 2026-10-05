"""Small shared helpers for the frozen Generation 23 sibling batch."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import json
import sys
from collections.abc import Callable, Sequence
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
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)


ANALYSIS_PATH = Path(__file__).resolve()


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def load_branch(branch_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(g23z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation23_outcomes":
        raise ValueError("Generation 23 batch is not frozen.")
    branch = next(
        (item for item in frozen["branches"] if item["branch_id"] == branch_id), None
    )
    if branch is None:
        raise ValueError(f"Generation 23 branch is missing: {branch_id}")
    if not str(branch["status_at_freeze"]).startswith("frozen_"):
        raise ValueError(f"Generation 23 branch is not active: {branch_id}")
    return frozen, branch


def all_support_contracts() -> tuple[tuple[Path, str], ...]:
    """Resolve every active sibling lazily so route imports cannot form a cycle."""
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_continuous_context as context,
    )
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_freqai_cache as freqai,
    )
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_multitimeframe_attribution as mtf,
    )
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_price_distribution as distribution,
    )
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_round_numbers as round_numbers,
    )

    return (
        (
            mtf.SUPPORT_MANIFEST,
            "frozen_before_generation23_multitimeframe_attribution_outcomes",
        ),
        (
            context.SUPPORT_MANIFEST,
            "frozen_before_generation23_continuous_context_outcomes",
        ),
        (
            freqai.support_manifest_path("normal"),
            "frozen_outcome_blind_generation23_freqai_support",
        ),
        (
            freqai.support_manifest_path("meme"),
            "frozen_outcome_blind_generation23_freqai_support",
        ),
        (
            round_numbers.SUPPORT_MANIFEST,
            "frozen_before_generation23_round_number_outcomes",
        ),
        (
            distribution.SUPPORT_MANIFEST,
            "frozen_before_generation23_distribution_outcomes",
        ),
    )


def require_all_frozen_supports() -> list[dict[str, Any]]:
    """Block every outcome path until the whole five-route batch is frozen."""
    verified: list[dict[str, Any]] = []
    for path, expected_status in all_support_contracts():
        if not path.is_file():
            raise ValueError(f"Generation 23 sibling support is not frozen: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != expected_status:
            raise ValueError(f"Generation 23 sibling support has wrong status: {path}")
        if manifest.get("future_outcome_values_read") is True:
            raise ValueError(f"Generation 23 sibling support opened outcomes: {path}")
        verified.append(artifact(path))
    return verified


def open_frozen_support_outcomes(
    manifest: dict[str, Any],
    *,
    phase: str,
    scope_events: Callable[[DataFrame], DataFrame],
) -> DataFrame:
    """Open future OHLCV only after a route's outcome-blind support is frozen."""
    parts: list[DataFrame] = []
    inventory = manifest["inventory"]
    for number, item in enumerate(inventory, start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen Generation 23 support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        scoped = scope_events(events)
        scoped["g18_period"] = scoped["period"].astype(str)
        parts.append(scoped)
        print(
            json.dumps(
                {"phase": phase, "processed": number, "total": len(inventory)}
            ),
            flush=True,
        )
    if not parts:
        raise ValueError("Generation 23 support inventory produced no outcomes.")
    return pd.concat(parts, ignore_index=True, sort=False)


def score_control_ladders(
    events: DataFrame,
    *,
    controls: Sequence[str],
    question_keys: Sequence[str] = (
        "scope_kind",
        "scope_value",
        "metric",
        "horizon_hours",
    ),
) -> tuple[DataFrame, DataFrame, DataFrame]:
    summary = g22a.metric_summary(events)
    contrasts = g18d.paired_contrasts(summary, controls)
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, controls)
    return contrasts, scores, decisions


def write_scored_tables(
    run_dir: Path,
    *,
    prefix: str,
    contrasts: DataFrame,
    scores: DataFrame,
    decisions: DataFrame,
) -> dict[str, Path]:
    paths = {
        "contrasts": run_dir / f"{prefix}_pair_contrasts.csv",
        "scores": run_dir / f"{prefix}_period_scores.csv",
        "decisions": run_dir / f"{prefix}_decisions.csv",
    }
    for name, frame in (
        ("contrasts", contrasts),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    return paths
