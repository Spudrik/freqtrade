# ruff: noqa: S101

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_timeframe_portability as g18t,
)


def test_source_registry_is_compact_and_multitimeframe() -> None:
    specs = g18t.level_specs()

    assert len(specs) == 8
    assert g18t.SOURCE_TIMEFRAMES == ("1h", "4h", "8h")
    assert {spec.family for spec in specs} == {
        "adaptive_volume_profile_nodes",
        "donchian_boundaries",
        "rolling_vwap_deviation_bands",
    }


def test_profile_columns_are_causal_completed_bar_outputs() -> None:
    rows = 80
    source = pd.DataFrame(
        {
            "high": [float(index + 2) for index in range(rows)],
            "low": [float(index) for index in range(rows)],
            "close": [float(index + 1) for index in range(rows)],
            "volume": [100.0 + index for index in range(rows)],
        }
    )

    result = g18t.profile_columns(source, "1h")

    assert result.iloc[:71].isna().all().all()
    assert result.iloc[71:].notna().all().all()


def test_source_surface_availability_follows_source_close() -> None:
    source = g18t.source_level_surface("BTC/USDT:USDT", "4h")

    assert (source["available_at"] - source["source_open"]).eq(
        pd.Timedelta(hours=4)
    ).all()
