# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)


def test_resample_uses_candle_open_clock_and_complete_ohlcv() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC"),
            "open": range(8),
            "high": range(1, 9),
            "low": range(8),
            "close": range(1, 9),
            "volume": [1.0] * 8,
        }
    )

    output = g13c.resample_ohlcv(frame, "4h")

    assert output["date"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T04:00:00Z"),
    ]
    assert output["volume"].tolist() == [4.0, 4.0]
    assert output["close"].tolist() == [4, 8]


def test_long_anchor_selection_enforces_48_hour_spacing() -> None:
    frame = pd.DataFrame(
        {
            "control": ["actual"] * 4,
            "cohort": ["normal"] * 4,
            "pair": ["BTC/USDT:USDT"] * 4,
            "period": ["validation_early"] * 4,
            "event_time": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05"], utc=True
            ),
            "source_timeframe": ["4h"] * 4,
            "contact_close_distance_atr": [0.1, 0.2, 0.3, 0.4],
            "level_score": [1.0] * 4,
            "level_name": ["x"] * 4,
            "level_family": ["generic_prior_range"] * 4,
            "representation": ["x"] * 4,
            "zone_method": ["standard_base_atr"] * 4,
        }
    )

    output = g13c.select_long_anchors(frame)

    assert output["event_time"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-03T00:00:00Z"),
        pd.Timestamp("2026-01-05T00:00:00Z"),
    ]


def test_recent_period_relabel_is_half_open() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-06-30T23:00:00Z", "2026-07-01T00:00:00Z", "2026-08-21T00:00:00Z"]
            ),
            "period": ["exposed_recent_diagnostic"] * 3,
        }
    )

    output = g13c.relabel_periods(frame, "normal")

    assert output["period"].tolist() == [
        "recent_confirmation_early",
        "recent_confirmation_late",
        "exposed_recent_diagnostic",
    ]


def test_source_manifest_routes_by_cohort() -> None:
    assert g13c.manifest_for_cohort("normal").name == "g6_normal_source_manifest.json"
    assert g13c.manifest_for_cohort("meme").name == "g6_meme_source_manifest.json"


def test_mtf_materialization_replaces_base_readiness_with_extended_readiness(
    tmp_path,
) -> None:
    support = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "period": ["validation_early"],
            "ready__g11_minimal_contact": [True],
            "ready__g13_mtf_4h_activity_volatility": [True],
        }
    )
    source = pd.DataFrame(
        {
            "date": support["date"],
            "period": support["period"],
            "&-g11_reaction_h1": [1.0],
            "ready__g11_minimal_contact": [False],
        }
    )
    support_path = tmp_path / "support.parquet"
    event_source = tmp_path / "event_source.parquet"
    evaluation_source = tmp_path / "evaluation_source.parquet"
    support.to_parquet(support_path)
    source.to_parquet(event_source)
    source.to_parquet(evaluation_source)

    result = g13c.materialize_mtf_targets(
        {"pair": "BTC/USDT:USDT", "mtf_support_path": str(support_path)},
        g11_item={
            "event_path": str(event_source),
            "evaluation_path": str(evaluation_source),
        },
        event_dir=tmp_path / "event",
        evaluation_dir=tmp_path / "evaluation",
    )
    materialized = pd.read_parquet(result["mtf_event_path"])

    assert bool(materialized.loc[0, "ready__g11_minimal_contact"])
    assert bool(materialized.loc[0, "ready__g13_mtf_4h_activity_volatility"])
