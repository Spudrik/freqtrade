# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_event_cache as g6e,
)


def test_attach_enrichment_uses_event_base_index() -> None:
    events = pd.DataFrame({"base_index": [2, 0], "event_time": ["b", "a"]})
    enrichment = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
            "state__value": [10.0, 20.0, 30.0],
        }
    )

    attached = g6e.attach_enrichment(events, enrichment)

    assert attached["state__value"].tolist() == [30.0, 10.0]


def test_relationship_flags_are_sampled_at_each_event_clock() -> None:
    events = pd.DataFrame({"base_index": [3, 1, 0]})
    masks = {
        ("4h", "same_mechanism_agreement"): np.array([True, False, True, False]),
        ("1d", "opposing_side_overlap"): np.array([False, True, False, True]),
    }

    g6e.attach_relationship_flags(events, masks)

    assert events["rel__4h__same_mechanism_agreement"].tolist() == [False, False, True]
    assert events["rel__1d__opposing_side_overlap"].tolist() == [True, True, False]


def test_market_groups_keep_btc_doge_and_memes_distinct() -> None:
    assert g6e.market_group("normal", "BTC/USDT:USDT") == "btc_separate"
    assert g6e.market_group("normal", "DOGE/USDT:USDT") == "doge_bridge"
    assert g6e.market_group("normal", "SOL/USDT:USDT") == "established_altcoins"
    assert g6e.market_group("meme", "DOGE/USDT:USDT") == "frozen_top_ten_memes"


def test_topic_groups_are_distinct_and_low_dimensional() -> None:
    flattened = [column for columns in g6e.TOPIC_GROUPS.values() for column in columns]

    assert len(g6e.TOPIC_GROUPS) == 4
    assert len(flattened) == len(set(flattened))
    assert all(column.endswith("_count_1h") for column in flattened)
