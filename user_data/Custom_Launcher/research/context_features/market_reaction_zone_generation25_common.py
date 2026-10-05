"""Shared freeze and all-sibling gates for Generation 25 research."""

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
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)


ANALYSIS_PATH = Path(__file__).resolve()


def artifact(path: Path) -> dict[str, Any]:
    return g25z.artifact(path)


def load_branch(branch_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(g25z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation25_outcomes":
        raise ValueError("Generation 25 batch is not frozen.")
    branch = next((item for item in frozen["branches"] if item["branch_id"] == branch_id), None)
    if branch is None or branch.get("status_at_freeze") != "frozen_pending_support":
        raise ValueError(f"Generation 25 branch is not active: {branch_id}")
    return frozen, branch


def all_support_contracts() -> tuple[tuple[Path, str], ...]:
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation25_connected_volume_profile as volume_profile,
    )
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation25_convergence_representation as convergence,
    )
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation25_market_state_activity as activity,
    )

    return (
        (
            convergence.SUPPORT_MANIFEST,
            "frozen_before_generation25_convergence_outcomes",
        ),
        (
            volume_profile.SUPPORT_MANIFEST,
            "frozen_before_generation25_connected_vp_outcomes",
        ),
        (
            activity.SUPPORT_MANIFEST,
            "frozen_before_generation25_market_state_outcomes",
        ),
    )


def require_all_frozen_supports() -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for path, expected_status in all_support_contracts():
        if not path.is_file():
            raise ValueError(f"Generation 25 sibling support is not frozen: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != expected_status:
            raise ValueError(f"Generation 25 sibling support has wrong status: {path}")
        if manifest.get("future_outcome_values_read") is True:
            raise ValueError(f"Generation 25 sibling support opened outcomes: {path}")
        verified.append(artifact(path))
    return verified


def open_direct_support_outcomes(
    manifest: dict[str, Any],
    *,
    phase: str,
    scope_events: Callable[[DataFrame], DataFrame] | None = None,
) -> DataFrame:
    """Open a frozen direct-test surface only after every sibling is frozen."""
    require_all_frozen_supports()
    parts: list[DataFrame] = []
    for number, item in enumerate(manifest["inventory"], start=1):
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen Generation 25 support changed: {path}")
        events = pd.read_parquet(path)
        events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), str(item["cohort"]))
        g20s.add_future_metrics(events, base)
        if scope_events is not None:
            events = scope_events(events)
        events["g18_period"] = events["period"].astype(str)
        parts.append(events)
        print(
            json.dumps({"phase": phase, "processed": number, "total": len(manifest["inventory"])}),
            flush=True,
        )
    if not parts:
        raise ValueError("Generation 25 support inventory produced no outcomes.")
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
    return g24c.score_control_ladders(
        events,
        controls=controls,
        question_keys=question_keys,
    )


def write_scored_tables(
    run_dir: Path,
    *,
    prefix: str,
    contrasts: DataFrame,
    scores: DataFrame,
    decisions: DataFrame,
) -> dict[str, Path]:
    return g24c.write_scored_tables(
        run_dir,
        prefix=prefix,
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
