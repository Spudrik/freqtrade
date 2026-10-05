from __future__ import annotations

# ruff: noqa: S101
import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_context_independent_freeze as freeze,
)


def _global_source(values: list[float]) -> DataFrame:
    return DataFrame(
        {
            "source_id": ["synthetic_source"] * len(values),
            "metric_key": ["synthetic_metric"] * len(values),
            "configured_enabled": [True] * len(values),
            "available_at": pd.date_range(
                "2026-06-01T00:00:00Z", periods=len(values), freq="1D"
            ),
            "extract_sequence": list(range(len(values))),
            "source_ts": [""] * len(values),
            "value": values,
            "change_kind": ["live_change"] * len(values),
        }
    )


def _global_spec(
    *, relation: str = "same", transform: str = "difference"
) -> dict[str, object]:
    return {
        "label": "Synthetic source",
        "source_id": "synthetic_source",
        "metric_key": "synthetic_metric",
        "value_column": "value",
        "transform": transform,
        "sample_mode": "native",
        "selection_quantile": 0.50,
        "minimum_history": 2,
        "rolling_history": 2,
        "cooldown_hours": 0,
        "horizons_hours": (1, 4),
        "direction_relation": relation,
        "target_kind": "market",
        "source_role": "synthetic_test_source",
    }


def _observation(
    anchor: pd.Timestamp,
    *,
    family: str = "synthetic_family",
    is_event: bool = False,
    source_direction: int = 1,
    source_available: bool = True,
) -> dict[str, object]:
    return {
        "family": family,
        "anchor_utc": anchor,
        "whole_event_partition": freeze.partition_for(anchor),
        "is_event": is_event,
        "source_available": source_available,
        "source_direction": source_direction,
    }


def test_partition_boundaries_are_start_inclusive_and_end_exclusive() -> None:
    first_name, first_start, first_end = freeze.PARTITIONS[0]
    second_name, second_start, second_end = freeze.PARTITIONS[1]

    assert freeze.partition_for(first_start) == first_name
    assert freeze.partition_for(first_end - pd.Timedelta(nanoseconds=1)) == first_name
    assert freeze.partition_for(second_start) == second_name
    assert freeze.partition_for(second_end - pd.Timedelta(nanoseconds=1)) == second_name
    assert freeze.partition_for(second_end) is None


def test_global_threshold_uses_only_values_before_the_current_observation() -> None:
    spec = _global_spec(transform="level")
    baseline = freeze.sample_global_family(
        _global_source([10.0, 12.0, 100.0, 101.0]), "synthetic", spec
    )
    changed_current = freeze.sample_global_family(
        _global_source([10.0, 12.0, 1000.0, 1001.0]), "synthetic", spec
    )

    assert baseline["event_threshold"].iloc[:2].isna().all()
    assert baseline["event_threshold"].iloc[2] == 11.0
    assert baseline["event_threshold"].iloc[3] == 56.0
    assert changed_current["event_threshold"].iloc[2] == baseline["event_threshold"].iloc[2]


def test_cooldown_keeps_exact_boundary_and_suppresses_inside_boundary() -> None:
    anchors = pd.Series(
        pd.to_datetime(
            [
                "2026-06-01T00:00:00Z",
                "2026-06-01T01:00:00Z",
                "2026-06-01T04:00:00Z",
                "2026-06-01T05:00:00Z",
            ],
            utc=True,
        )
    )

    selected = freeze.apply_cooldown(anchors, pd.Series([True] * 4), hours=4)

    assert selected.tolist() == [True, False, True, False]


def test_signed_source_predicts_sign_but_unsigned_attention_abstains() -> None:
    source = _global_source([10.0, 13.0, 9.0])
    signed = freeze.sample_global_family(source, "signed", _global_spec())
    unsigned = freeze.sample_global_family(
        source,
        "unsigned_attention",
        _global_spec(relation="abstain_unsigned_attention"),
    )

    assert signed["source_direction"].tolist() == [0, 1, -1]
    assert signed["predicted_direction"].tolist() == [0, 1, -1]
    assert unsigned["source_direction"].tolist() == [0, 1, -1]
    assert unsigned["predicted_direction"].tolist() == [0, 0, 0]


def test_controls_are_prior_far_and_sign_matched_when_enough_signs_exist() -> None:
    event_anchor = pd.Timestamp("2026-07-20T12:00:00Z")
    prior_controls = [
        pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(days=offset)
        for offset in range(7)
    ]
    family = "live_news_activity_spike"
    rows = [_observation(anchor, family=family) for anchor in prior_controls]
    rows.extend(
        [
            _observation(
                pd.Timestamp("2026-06-15T00:00:00Z"),
                family=family,
                source_direction=-1,
            ),
            _observation(event_anchor, family=family, is_event=True),
            _observation(event_anchor - pd.Timedelta(days=1), family=family),
            _observation(event_anchor + pd.Timedelta(days=1), family=family),
        ]
    )
    observations = DataFrame(rows)
    events = DataFrame(
        [
            {
                "event_id": "synthetic_event",
                "family": family,
                "anchor_utc": event_anchor,
                "predicted_direction": 1,
                "source_direction": 1,
            }
        ]
    )

    controls = freeze.control_catalog(observations, events)

    assert len(controls) == len(prior_controls)
    assert set(controls["control_anchor_utc"]) == set(prior_controls)
    assert controls["control_anchor_utc"].lt(event_anchor).all()
    assert controls["control_source_direction"].eq(1).all()
    assert controls["control_rank"].tolist() == list(range(1, len(prior_controls) + 1))


def test_no_events_produce_no_controls() -> None:
    observations = DataFrame(
        [
            _observation(pd.Timestamp("2026-06-01T00:00:00Z")),
            _observation(pd.Timestamp("2026-06-02T00:00:00Z")),
        ]
    )
    events = freeze.event_catalog(observations)

    controls = freeze.control_catalog(observations, events)

    assert events.empty
    assert controls.empty
