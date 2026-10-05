# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation5_meme_lvn_replication_freeze as g5a,
)


def selection_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {column: 1.0 for column in g5a.SELECTION_COLUMNS}
    row.update(
        {
            "pair": "DOGE/USDT:USDT",
            "source_timeframe": "1h",
            "level_family": "volume_profile",
            "level_name": "lvn_below",
            "level_column": "vp_lvn_below",
            "representation": "zone",
            "control": "actual",
            "zone_method": "atr",
            "base_index": 100,
            "event_time": "2026-02-01T00:00:00Z",
            "period": "meme_validation_early",
            "source_available_at": "2026-01-31T23:00:00Z",
            "source_open": "2026-01-31T23:00:00Z",
            "approach_state": "from_above",
            "cohort": "meme",
            "parent_run_id": "parent",
            "selection_key": "source-key",
            "selection_hash": "source-hash",
            "episode_id": "episode-1",
            "g4a_selection_hash": "g4-hash",
            "qualification_reason": "causal",
            "direction_used_for_selection": False,
            "retrospective_price_reference": False,
            "price_reference_known_at_event": True,
            "level_role": "prospective_reaction_location",
        }
    )
    row.update(overrides)
    return row


def test_selection_projection_filters_scope_and_drops_parent_outcomes() -> None:
    rows = [
        selection_row(),
        selection_row(episode_id="wrong-side", level_name="lvn_above"),
        selection_row(episode_id="wrong-period", period="meme_development"),
    ]
    frame = pd.DataFrame(rows)
    frame["contact_volume_ratio"] = [99.0, 98.0, 97.0]

    selected = g5a.selection_projection(frame)

    assert selected["episode_id"].tolist() == ["episode-1"]
    assert "contact_volume_ratio" not in selected.columns
    assert selected["g5a_selection_hash"].str.len().eq(64).all()


def test_prior_path_exclusion_retains_exact_48_hour_boundary() -> None:
    candidates = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-01-02T23:59:00Z", "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z"],
                utc=True,
            ),
            "episode_id": ["inside", "boundary", "outside"],
        }
    )
    exposed = pd.Series(pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True))

    result = g5a.exclude_prior_outcome_paths(candidates, exposed_times=exposed, hours=48)

    assert result["episode_id"].tolist() == ["boundary", "outside"]
    assert result["minutes_to_nearest_g4a_episode"].min() == 48 * 60


def test_balanced_confirmation_sample_round_robins_pairs() -> None:
    rows: list[dict[str, object]] = []
    event = pd.Timestamp("2026-01-01T00:00:00Z")
    for period_index, period in enumerate(g5a.CONFIRMATION_PERIODS):
        for index in range(40):
            rows.append(
                {
                    "period": period,
                    "pair": f"COIN{index % 10}/USDT:USDT",
                    "event_time": event + pd.Timedelta(hours=49 * (period_index * 40 + index)),
                    "episode_id": f"{period}-{index}",
                    "g5a_selection_hash": f"{period_index}{index:063d}",
                }
            )
    frame = pd.DataFrame(rows)

    sample = g5a.select_balanced_confirmation_sample(frame)
    decision = g5a.support_decision(sample)

    assert len(sample) == g5a.TARGET_EPISODES
    assert decision["period_counts"] == {
        "meme_validation_early": 30,
        "meme_validation_late": 30,
    }
    assert decision["pairs"] == 10
    assert decision["passed"] is True


def test_support_gate_parks_balanced_sample_below_minimum() -> None:
    frame = pd.DataFrame(
        {
            "period": ["meme_validation_early"] * 14 + ["meme_validation_late"] * 14,
            "pair": [f"COIN{i % 5}" for i in range(28)],
        }
    )

    decision = g5a.support_decision(frame)

    assert decision["episodes"] == 28
    assert decision["balanced_periods"] is True
    assert decision["passed"] is False


def test_frozen_sample_rejects_outcome_column() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "period": ["meme_validation_early"] * 15 + ["meme_validation_late"] * 15,
            "pair": [f"COIN{i % 5}" for i in range(30)],
            "event_time": [start + pd.Timedelta(hours=49 * i) for i in range(30)],
            "minutes_to_nearest_g4a_episode": [4000.0] * 30,
            "contact_volume_ratio": [2.0] * 30,
        }
    )

    try:
        g5a.validate_frozen_sample(frame)
    except ValueError as exc:
        assert "Outcome columns leaked" in str(exc)
    else:
        raise AssertionError("Expected frozen-sample outcome-column rejection")
