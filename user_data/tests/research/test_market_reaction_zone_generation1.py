from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as generation0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation2_density_reaction as g2e_reaction,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_density_arrival as g3d_arrival,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_density_arrival_controls as g3d_controls,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_density_arrival_states as g3d_states,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_external_context as g3h_context,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_generic_levels as g3f_generic,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_generic_levels_artificial_controls as g3f_artificial,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_generic_levels_attribution_review as g3f_attribution,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_generic_levels_reaction as g3f_reaction,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_generic_levels_review as g3f_review,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_lvn_attribution as g3a_attribution,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3g_replay,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay_analysis as g3g_analysis,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_participation_origin as g3e_origin,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_participation_origin_artificial_controls as g3e_artificial,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_participation_origin_components as g3e_components,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_participation_origin_reaction as g3e_reaction,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_room_obstacles as g3b_obstacles,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_breadth as g4a_breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_confirmation as g4a_confirmation,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_confirmation_review as g4a_review,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_download as g4a_download,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_expansion_download as g4a_expansion,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_repair as g4a_repair,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_unconditioned_freeze as g4a_unconditioned,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    ClusterComponent,
    cache_family_groups,
    cluster_storage_roots,
    component_dependency_group,
    contacted_component_metadata,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (  # noqa: E501
    PROFILES,
    SCORE_METRICS,
    attach_profile_deltas,
    event_targets,
    profile_config,
    select_non_overlapping_event_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_cluster_increment import (  # noqa: E501
    anchor_support_inventory,
    explode_contacted_anchors,
    recurring_pattern_inventory,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_cluster_outcomes import (  # noqa: E501
    canonical_multi_anchors,
    strict_episode_rows,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    OUTPUT_SCHEMA_VERSION,
    attach_control_geometry,
    attach_overlap,
    balance_diagnostics,
    matched_cell_rows,
    nearest_state_pairs,
    outcome_columns,
    outcome_metadata,
    purge_overlapping_event_pairs,
    purge_overlapping_event_pairs_by_key,
    validate_existing_run_record,
    validate_pair_output_contract,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    assign_attribute_tertiles,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    OUTCOMES,
    OUTCOMES_BY_RESPONSE_WINDOW,
    STATE_FEATURES,
    AnchorLevel,
    causal_anchor_levels,
    declared_matching_balance,
    leave_one_coin_out_results,
    matched_outcome_rows,
    matched_pair_balance,
    reset_anchor_stats,
    swing_anchor_stats,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density import (  # noqa: E501
    causal_density_zones,
    confirmed_swing_points,
    connected_density_candidates,
    coverage_record,
    manifest_analysis_end_exclusive,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    AlignedLevel,
    attach_independent_overlap,
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_regime import (  # noqa: E501
    LENSES,
    attach_pair_regimes,
    causal_tercile_bucket,
    coherent_pair_regime,
    validate_source_integrity_contract,
    vote_regime,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    CORE_STATE_FEATURES,
    FULL_STATE_FEATURES,
    persistence_bars,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration1Strategy import (
    TARGET_COLUMNS,
    MarketReactionZoneG1DLocalStateFreqAIResearchStrategy,
)


STATE_COLUMNS = ("state_0", "state_1", "state_2", "state_3")


def test_cluster_storage_and_cache_surface_follow_isolated_manifest() -> None:
    manifest = {
        "storage": {
            "cache_dir": "C:/tmp/mrz/cache",
            "report_dir": "C:/tmp/mrz/reports",
            "event_dir": "D:/tmp/mrz/events",
            "detailed_summary_dir": "D:/tmp/mrz/summaries",
        },
        "cache_surface": {"families": ["core", "generic"]},
    }

    report_root, artifact_root = cluster_storage_roots(manifest)

    assert report_root == Path("C:/tmp/mrz/cluster_reports")
    assert artifact_root == Path("D:/tmp/mrz/cluster_artifacts")
    assert cache_family_groups(manifest) == (("core", "generic"),)


def _matching_frame(
    values: list[float],
    *,
    distances: list[float],
    times: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame({column: values for column in STATE_COLUMNS})
    frame["pre_distance_atr"] = distances
    frame["event_time"] = pd.to_datetime(times, utc=True)
    return frame


def _cell_frame(
    values: list[float],
    *,
    distances: list[float],
    times: list[str],
    contact_addition: float,
) -> pd.DataFrame:
    frame = _matching_frame(values, distances=distances, times=times)
    frame["period"] = "p1"
    frame["approach_state"] = "from_below"
    frame["pair"] = "BTC/USDT:USDT"
    frame["source_timeframe"] = "1h"
    frame["level_family"] = "generic"
    frame["level_name"] = "rolling_high_24"
    frame["level_column"] = "generic_rolling_high_24"
    frame["zone_method"] = "tight_base_atr"
    frame["overlap_any_real_level"] = False
    frame["overlap_other_real_level_count"] = 0
    frame["overlap_source_current_level"] = False
    frame["contact_close_distance_atr"] = 0.05
    frame["contact_range_ratio"] = np.asarray(values) + contact_addition
    frame["contact_volume_ratio"] = np.asarray(values) + 2.0 + contact_addition
    frame["contact_pressure_change"] = np.asarray(values) / 10.0
    frame["abs_excursion_atr_h1"] = np.asarray(values) + 0.5
    frame["close_abs_displacement_atr_h1"] = np.asarray(values) + 0.25
    frame["range_ratio_h1"] = np.asarray(values) + 1.0
    frame["volume_ratio_h1"] = np.asarray(values) + 1.5
    frame["pressure_change_h1"] = np.asarray(values) / 5.0
    frame["dwell_fraction_h1"] = 0.5
    frame["crossings_h1"] = 1.0
    return frame


def test_matching_enforces_pre_contact_distance_gate() -> None:
    actual = _matching_frame(
        [0.0, 1.0, 2.0, 3.0],
        distances=[1.0, 1.1, 1.2, 1.3],
        times=[f"2024-01-10T0{hour}:00:00Z" for hour in range(4)],
    )
    control = _matching_frame(
        [0.01, 1.01, 2.01, 3.01, 0.0, 1.0, 2.0, 3.0],
        distances=[1.05, 1.15, 1.25, 1.35, 1.5, 1.6, 1.7, 1.8],
        times=[f"2023-01-01T0{hour}:00:00Z" for hour in range(8)],
    )

    pairs, audit = nearest_state_pairs(
        actual,
        control,
        state_columns=STATE_COLUMNS,
        pre_distance_atr_caliper=0.10,
    )

    assert sorted(pair[1] for pair in pairs) == [0, 1, 2, 3]
    assert audit["geometry_eligible_actual"] == 4


def test_matching_rejects_overlapping_future_windows() -> None:
    actual = _matching_frame(
        [0.0, 1.0, 2.0, 3.0],
        distances=[1.0, 1.1, 1.2, 1.3],
        times=[f"2024-01-10T0{hour}:00:00Z" for hour in range(4)],
    )
    control = _matching_frame(
        [0.0, 1.0, 2.0, 3.0, 0.01, 1.01, 2.01, 3.01],
        distances=[1.0, 1.1, 1.2, 1.3, 1.0, 1.1, 1.2, 1.3],
        times=[
            *[f"2024-01-09T0{hour}:00:00Z" for hour in range(4)],
            *[f"2023-01-01T0{hour}:00:00Z" for hour in range(4)],
        ],
    )

    pairs, audit = nearest_state_pairs(
        actual,
        control,
        state_columns=STATE_COLUMNS,
        minimum_event_separation_hours=48,
    )

    assert sorted(pair[1] for pair in pairs) == [4, 5, 6, 7]
    assert audit["geometry_eligible_actual"] == 4


def test_matching_keeps_control_reuse_bounded() -> None:
    actual_values = [0.0, 1.0] * 6
    actual = _matching_frame(
        actual_values,
        distances=[1.0] * 12,
        times=[f"2024-01-{day:02d}T00:00:00Z" for day in range(1, 13)],
    )
    control = _matching_frame(
        [0.0, 1.0],
        distances=[1.0, 1.0],
        times=["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"],
    )

    pairs, _ = nearest_state_pairs(actual, control, state_columns=STATE_COLUMNS)
    reuse = pd.Series([pair[1] for pair in pairs]).value_counts()

    assert len(pairs) == 6
    assert int(reuse.max()) == 3


def test_shifted_controls_are_split_by_price_geometry() -> None:
    events = pd.DataFrame(
        {
            "base_index": [0, 1, 2, 3],
            "level_price": [120.0, 105.0, 95.0, 90.0],
        }
    )
    result = attach_control_geometry(
        events,
        control_name="shift_+1atr",
        source_level=np.array([110.0, 110.0, 110.0, 90.0]),
        pre_close=np.array([100.0, 100.0, 100.0, 100.0]),
        base_atr=np.array([10.0, 10.0, 10.0, 10.0]),
    )

    assert result["control_geometry"].tolist() == [
        "outward",
        "inward",
        "crossed_price",
        "same_distance",
    ]


def test_actual_contacts_separate_isolated_levels_from_cluster_members() -> None:
    events = pd.DataFrame(
        {
            "base_index": [0, 1],
            "level_price": [100.0, 110.0],
            "zone_half_width": [0.1, 0.1],
        }
    )
    result = attach_overlap(
        events,
        real_matrix=np.array([[100.0, 105.0], [110.0, 110.15]]),
        base_atr=np.array([1.0, 1.0]),
        zone="tight_base_atr",
        source_index=0,
    )
    scopes = np.where(
        result["overlap_other_real_level_count"].eq(0),
        "isolated_level",
        "level_cluster_member",
    )

    assert scopes.tolist() == ["isolated_level", "level_cluster_member"]


def test_cluster_contacts_retain_exact_component_pattern() -> None:
    components = [
        ClusterComponent(
            index=0,
            key="1h|generic_prior_range|rolling_high_24|settled",
            source_key="1h|generic_prior_range|generic_rolling_high_24",
            family="generic_prior_range",
            timeframe="1h",
            name="rolling_high_24",
            column="generic_rolling_high_24",
            representation="settled",
            interpretation="activity_transit",
            dynamic_representation=False,
            mechanism="rolling_price_extreme",
            dependency_group="rolling_price_extreme",
        ),
        ClusterComponent(
            index=1,
            key="4h|volume_profile_nodes|lvn_above|settled",
            source_key="4h|volume_profile_nodes|vp_lvn_above",
            family="volume_profile_nodes",
            timeframe="4h",
            name="lvn_above",
            column="vp_lvn_above",
            representation="settled",
            interpretation="activity_transit",
            dynamic_representation=False,
            mechanism="volume_profile",
            dependency_group="volume_profile",
        ),
    ]

    metadata = contacted_component_metadata(
        cluster={"member_indexes_list": (0, 1)},
        row_values=np.array([100.0, 100.1]),
        components=components,
        atr=1.0,
        candle_high=100.2,
        candle_low=99.9,
        scale="tight",
        price_shift=0.0,
    )

    assert metadata["contacted_component_keys"] == (
        "1h|generic_prior_range|rolling_high_24|settled;4h|volume_profile_nodes|lvn_above|settled"
    )
    assert metadata["contacted_component_source_keys"] == (
        "1h|generic_prior_range|generic_rolling_high_24;4h|volume_profile_nodes|vp_lvn_above"
    )
    assert metadata["contacted_component_mechanisms_aligned"] == (
        "rolling_price_extreme;volume_profile"
    )
    assert metadata["contacted_component_dependency_groups_aligned"] == (
        "rolling_price_extreme;volume_profile"
    )
    assert metadata["contacted_component_interpretations_aligned"] == (
        "activity_transit;activity_transit"
    )
    assert metadata["contacted_component_indexes"] == "0;1"
    assert metadata["contacted_component_prices"] == "100;100.09999999999999"
    assert metadata["contacted_component_mechanisms"] == ("rolling_price_extreme;volume_profile")
    assert metadata["contacted_component_timeframes"] == "1h;4h"
    assert metadata["contacted_mechanism_count"] == 2
    assert metadata["contacted_dependency_group_count"] == 2
    assert metadata["contacted_independent_mechanisms"] is True
    assert metadata["contacted_independent_dependency_groups"] is True
    assert metadata["contacted_component_signature"] != "none"


def test_bollinger_and_moving_average_share_a_dependency_group() -> None:
    bollinger = generation0.LevelSpec(
        name="bb20_upper",
        family="generic_bollinger",
        batch="g0b2",
        column="generic_bb20_upper",
    )
    moving_average = generation0.LevelSpec(
        name="sma_20",
        family="generic_moving_average",
        batch="g0b2",
        column="generic_sma_20",
    )

    assert component_dependency_group(bollinger) == "price_average_family"
    assert component_dependency_group(moving_average) == "price_average_family"


def _cluster_contact_row(
    *,
    future_path_key: str,
    episode_key: str,
    keys: str,
    indexes: str,
    prices: str,
    sources: str,
    families: str,
    mechanisms: str,
    dependency_groups: str,
    timeframes: str,
    interpretations: str,
    independent: bool,
) -> dict[str, object]:
    count = len(keys.split(";"))
    return {
        "pair": "BTC/USDT:USDT",
        "representation_mode": "projected",
        "cluster_scale": "tight",
        "base_index": int(future_path_key.rsplit("|", 1)[1]),
        "event_time": pd.Timestamp("2024-01-01", tz="UTC")
        + pd.Timedelta(hours=int(future_path_key.rsplit("|", 1)[1])),
        "period": "development",
        "approach_state": "from_below",
        "future_path_key": future_path_key,
        "cluster_contact_episode_key": episode_key,
        "contacted_component_signature": "pattern-ab" if count == 2 else "pattern-a",
        "contacted_component_count": count,
        "contacted_mechanism_count": len(set(mechanisms.split(";"))),
        "contacted_dependency_group_count": len(set(dependency_groups.split(";"))),
        "contacted_independent_mechanisms": independent,
        "contacted_independent_dependency_groups": independent,
        "cluster_class": "different_family_cross_timeframe",
        "contact_structure_class": (
            "different_family_cross_timeframe" if count == 2 else "single_component_contact"
        ),
        "cluster_width_atr": 0.2,
        "local_level_density_2atr": 4,
        "active_level_count": 12,
        "pre_close_at_event": 99.0,
        "base_atr": 1.0,
        "contacted_component_keys": keys,
        "contacted_component_indexes": indexes,
        "contacted_component_prices": prices,
        "contacted_component_source_keys_aligned": sources,
        "contacted_component_families_aligned": families,
        "contacted_component_mechanisms_aligned": mechanisms,
        "contacted_component_dependency_groups_aligned": dependency_groups,
        "contacted_component_timeframes_aligned": timeframes,
        "contacted_component_interpretations_aligned": interpretations,
        "contacted_component_families": ";".join(sorted(set(families.split(";")))),
        "contacted_component_mechanisms": ";".join(sorted(set(mechanisms.split(";")))),
        "contacted_component_dependency_groups": ";".join(
            sorted(set(dependency_groups.split(";")))
        ),
        "contacted_component_timeframes": ";".join(sorted(set(timeframes.split(";")))),
        "contacted_interpretation_class": "activity_transit",
    }


def test_anchor_support_uses_separate_episodes_and_deduplicates_future_paths() -> None:
    key_a = "1h|generic_prior_range|rolling_high_24|settled"
    key_b = "4h|volume_profile_nodes|lvn_above|settled"
    multi = _cluster_contact_row(
        future_path_key="BTC/USDT:USDT|1",
        episode_key="multi-1",
        keys=f"{key_a};{key_b}",
        indexes="0;1",
        prices="100;100.1",
        sources="source-a;source-b",
        families="generic_prior_range;volume_profile_nodes",
        mechanisms="rolling_price_extreme;volume_profile",
        dependency_groups="rolling_price_extreme;volume_profile",
        timeframes="1h;4h",
        interpretations="activity_transit;activity_transit",
        independent=True,
    )
    duplicate_nested = dict(multi, cluster_contact_episode_key="multi-1-nested")
    single = _cluster_contact_row(
        future_path_key="BTC/USDT:USDT|2",
        episode_key="single-a",
        keys=key_a,
        indexes="0",
        prices="100",
        sources="source-a",
        families="generic_prior_range",
        mechanisms="rolling_price_extreme",
        dependency_groups="rolling_price_extreme",
        timeframes="1h",
        interpretations="activity_transit",
        independent=False,
    )
    anchors = explode_contacted_anchors(pd.DataFrame([multi, duplicate_nested, single]))
    support = anchor_support_inventory(anchors).set_index("anchor_component_key")

    assert len(anchors) == 5
    assert support.loc[key_a, "future_path_episodes__independent_multi_component_contact"] == 1
    assert support.loc[key_a, "future_path_episodes__single_component_contact"] == 1
    assert bool(support.loc[key_a, "has_anchor_only_common_support"])
    assert support.loc[key_b, "future_path_episodes__single_component_contact"] == 0
    assert not bool(support.loc[key_b, "has_anchor_only_common_support"])


def test_repeated_cluster_patterns_count_unique_market_paths() -> None:
    key_a = "1h|generic_prior_range|rolling_high_24|settled"
    key_b = "4h|volume_profile_nodes|lvn_above|settled"
    first = _cluster_contact_row(
        future_path_key="BTC/USDT:USDT|1",
        episode_key="first",
        keys=f"{key_a};{key_b}",
        indexes="0;1",
        prices="100;100.1",
        sources="source-a;source-b",
        families="generic_prior_range;volume_profile_nodes",
        mechanisms="rolling_price_extreme;volume_profile",
        dependency_groups="rolling_price_extreme;volume_profile",
        timeframes="1h;4h",
        interpretations="activity_transit;activity_transit",
        independent=True,
    )
    duplicate = dict(first, cluster_contact_episode_key="nested-copy")
    second = dict(
        first,
        future_path_key="BTC/USDT:USDT|3",
        cluster_contact_episode_key="second",
        base_index=3,
        event_time=pd.Timestamp("2024-01-01T03:00:00Z"),
    )

    patterns = recurring_pattern_inventory(pd.DataFrame([first, duplicate, second]))

    assert len(patterns) == 1
    assert int(patterns.loc[0, "future_path_episodes"]) == 2
    assert bool(patterns.loc[0, "repeated_exact_pattern"])


def test_canonical_cluster_episode_keeps_all_anchors_from_one_nested_choice() -> None:
    frame = pd.DataFrame(
        [
            {
                "representation_mode": "projected",
                "cluster_scale": "tight",
                "future_path_key": "BTC|1",
                "cluster_contact_episode_key": "wide-three",
                "contacted_component_count": 3,
                "cluster_width_atr": 0.4,
                "anchor_component_key": anchor,
            }
            for anchor in ("a", "b", "c")
        ]
        + [
            {
                "representation_mode": "projected",
                "cluster_scale": "tight",
                "future_path_key": "BTC|1",
                "cluster_contact_episode_key": "narrow-two",
                "contacted_component_count": 2,
                "cluster_width_atr": 0.2,
                "anchor_component_key": anchor,
            }
            for anchor in ("a", "b")
        ]
    )

    selected = canonical_multi_anchors(frame)

    assert set(selected["cluster_contact_episode_key"]) == {"narrow-two"}
    assert set(selected["anchor_component_key"]) == {"a", "b"}


def test_strict_cluster_episode_requires_every_anchor_and_uses_worst_delta() -> None:
    matches = pd.DataFrame(
        {
            "pair": ["BTC", "BTC"],
            "representation_mode": ["projected", "projected"],
            "cluster_scale": ["tight", "tight"],
            "period": ["development", "development"],
            "multi_future_path_key": ["BTC|1", "BTC|1"],
            "multi_cluster_episode_key": ["episode", "episode"],
            "multi_event_time": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"]),
            "multi_base_index": [1, 1],
            "contacted_component_signature": ["ab", "ab"],
            "contacted_component_count": [2, 2],
            "contacted_dependency_group_count": [2, 2],
            "contacted_component_keys": ["a;b", "a;b"],
            "contacted_component_dependency_groups": ["x;y", "x;y"],
            "contacted_component_timeframes": ["1h;4h", "1h;4h"],
            "contacted_interpretation_class": [
                "activity_transit",
                "activity_transit",
            ],
            "anchor_component_key": ["a", "b"],
            "match_distance": [0.2, 0.3],
            "pre_distance_atr_abs_difference": [0.02, 0.03],
            "event_separation_hours": [72.0, 96.0],
            "delta__contact_range_ratio": [0.5, -0.1],
            "delta__contact_volume_ratio": [0.4, 0.2],
            "delta__contact_pressure_change_abs": [0.1, 0.05],
            "delta__abs_excursion_atr_h1": [0.4, 0.1],
            "delta__close_abs_displacement_atr_h1": [0.2, 0.1],
            "delta__range_ratio_h1": [0.3, 0.2],
            "delta__volume_ratio_h1": [0.3, 0.2],
            "delta__pressure_change_abs_h1": [0.1, 0.05],
            "delta__dwell_fraction_h1": [-0.1, -0.2],
            "delta__crossings_h1": [1.0, 0.0],
        }
    )

    strict = strict_episode_rows(matches, horizons=(1,))

    assert len(strict) == 1
    assert bool(strict.loc[0, "all_anchors_matched"])
    assert strict.loc[0, "minimum_delta_across_anchors__contact_range_ratio"] == pytest.approx(-0.1)
    assert strict.loc[0, "minimum_delta_across_anchors__contact_volume_ratio"] == pytest.approx(0.2)


def test_volume_profile_attribute_tiers_ignore_outcomes_and_shuffle_reproducibly() -> None:
    frame = pd.DataFrame(
        {
            "source_timeframe": ["4h"] * 15,
            "level_name": ["lvn_above"] * 15,
            "zone_method": ["tight_base_atr"] * 15,
            "period": ["development"] * 15,
            "approach_state": ["from_below"] * 15,
            "level_score": np.arange(15, dtype=float),
            "future_outcome_not_used": np.arange(15, dtype=float)[::-1],
        }
    )

    actual = assign_attribute_tertiles(
        frame,
        attribute_column="level_score",
        assignment="actual",
        seed_key="fixed",
    )
    shuffled_a = assign_attribute_tertiles(
        frame,
        attribute_column="level_score",
        assignment="shuffled_attribute",
        seed_key="fixed",
    )
    shuffled_b = assign_attribute_tertiles(
        frame.assign(future_outcome_not_used=999.0),
        attribute_column="level_score",
        assignment="shuffled_attribute",
        seed_key="fixed",
    )

    assert actual.loc[actual.attribute_tier.eq("high"), "level_score"].min() >= 9
    assert actual.loc[actual.attribute_tier.eq("low"), "level_score"].max() <= 4
    assert shuffled_a["attribute_tier"].tolist() == shuffled_b["attribute_tier"].tolist()
    assert (
        shuffled_a["attribute_value_for_tier"].tolist()
        == shuffled_b["attribute_value_for_tier"].tolist()
    )


def test_matched_cells_report_geometry_separation_and_contact_outcomes() -> None:
    actual = _cell_frame(
        [0.0, 1.0, 2.0, 3.0],
        distances=[1.00, 1.10, 1.20, 1.30],
        times=[f"2024-01-0{day}T00:00:00Z" for day in range(1, 5)],
        contact_addition=1.0,
    )
    control = _cell_frame(
        [0.01, 1.01, 2.01, 3.01],
        distances=[1.02, 1.12, 1.22, 1.32],
        times=[f"2023-01-0{day}T00:00:00Z" for day in range(1, 5)],
        contact_addition=0.0,
    )
    before_overlap = pd.concat(
        [control, control.iloc[[0]].assign(overlap_any_real_level=True)],
        ignore_index=True,
    )

    rows = matched_cell_rows(
        actual=actual,
        control=control,
        actual_scope="isolated_level",
        control_name="shift_+1atr",
        control_scope="no_real_level_overlap",
        control_geometry="outward",
        control_events_before_overlap=before_overlap,
        role="activity_transit",
        state_columns=STATE_COLUMNS,
        horizons=(1,),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["actual_scope"] == "isolated_level"
    assert row["control_geometry"] == "outward"
    assert row["control_events_before_overlap"] == 5
    assert row["pre_distance_atr_abs_difference_max"] <= 0.10
    assert row["event_separation_hours_min"] > 1
    assert row["delta_mean__contact_range_ratio"] == pytest.approx(0.99)


def test_run_record_reuse_requires_identical_request_fingerprint() -> None:
    validate_existing_run_record({"request_sha256": "same"}, request_sha256="same")
    with pytest.raises(ValueError, match="incompatible settings"):
        validate_existing_run_record(
            {"request_sha256": "old"},
            request_sha256="new",
        )


def test_existing_pair_output_requires_repaired_schema_and_request(tmp_path) -> None:
    path = tmp_path / "BTC.parquet"
    pd.DataFrame(
        {
            "pair": ["BTC/USDT:USDT"],
            "output_schema_version": [OUTPUT_SCHEMA_VERSION],
            "run_request_sha256": ["request-a"],
            "actual_scope": ["isolated_level"],
            "control_scope": ["no_real_level_overlap"],
            "control_geometry": ["outward"],
            "pre_distance_atr_abs_difference_median": [0.02],
            "event_separation_hours_min": [72.0],
        }
    ).to_parquet(path, index=False)

    validate_pair_output_contract(
        path,
        pair="BTC/USDT:USDT",
        request_sha256="request-a",
    )
    with pytest.raises(ValueError, match="different run request"):
        validate_pair_output_contract(
            path,
            pair="BTC/USDT:USDT",
            request_sha256="request-b",
        )


def test_cache_validation_rejects_changed_indicator_source(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    source = tmp_path / "source.feather"
    indicator = tmp_path / "indicator.py"
    cache = tmp_path / "levels.parquet"
    manifest.write_text('{"frozen": true}', encoding="utf-8")
    source.write_bytes(b"ohlcv")
    indicator.write_text("VALUE = 1\n", encoding="utf-8")
    cache.write_bytes(b"cache placeholder")
    metadata = {
        "manifest_sha256": generation0.sha256_file(manifest),
        "source": str(source),
        "source_sha256": generation0.sha256_file(source),
        "indicator_sha256": {"indicator.py": generation0.sha256_file(indicator)},
        "direction_prediction": False,
        "profit_optimization": False,
    }
    cache.with_suffix(".meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    monkeypatch.setattr(generation0, "REPO_ROOT", tmp_path)

    generation0.validate_cache_metadata(cache, manifest)
    indicator.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="indicator source changed"):
        generation0.validate_cache_metadata(cache, manifest)


def test_manifest_storage_paths_keep_defaults_and_allow_isolated_roots(tmp_path) -> None:
    defaults = generation0.manifest_storage_paths({})
    assert defaults.cache_dir == generation0.CACHE_DIR
    assert defaults.report_dir == generation0.REPORT_DIR
    assert defaults.event_dir == generation0.EVENT_DIR
    assert defaults.detailed_summary_dir == generation0.DETAILED_SUMMARY_DIR

    configured = generation0.manifest_storage_paths(
        {
            "storage": {
                "cache_dir": str(tmp_path / "cache"),
                "report_dir": str(tmp_path / "reports"),
                "event_dir": str(tmp_path / "events"),
                "detailed_summary_dir": str(tmp_path / "summaries"),
            }
        }
    )
    assert configured.cache_dir == tmp_path / "cache"
    assert configured.report_dir == tmp_path / "reports"
    assert configured.event_dir == tmp_path / "events"
    assert configured.detailed_summary_dir == tmp_path / "summaries"


def test_manifest_storage_paths_reject_relative_roots() -> None:
    with pytest.raises(ValueError, match="must be an absolute path"):
        generation0.manifest_storage_paths({"storage": {"cache_dir": "relative/cache"}})


def test_worker_cap_can_be_frozen_per_manifest_without_changing_default() -> None:
    generation0.validate_worker_count(4)
    with pytest.raises(ValueError, match="at most 4"):
        generation0.validate_worker_count(5)

    manifest = {"compute": {"maximum_threads_after_live_process_check": 14}}
    generation0.validate_worker_count(14, manifest=manifest)
    with pytest.raises(ValueError, match="at most 14"):
        generation0.validate_worker_count(15, manifest=manifest)


def test_g2_period_filter_embargoes_outcomes_at_period_end() -> None:
    events = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-08T23:00:00Z",
                    "2026-01-09T01:00:00Z",
                    "2026-02-01T00:00:00Z",
                ],
                utc=True,
            )
        }
    )
    manifest = {
        "data": {
            "chronological_periods": [
                {
                    "start_utc": "2026-01-01T00:00:00Z",
                    "end_utc_exclusive": "2026-01-11T00:00:00Z",
                }
            ]
        }
    }

    selected = eligible_period_events(events, manifest, embargo_hours=48)

    assert selected.index.tolist() == [0, 1]


def test_g2_independent_overlap_excludes_same_mechanism_levels() -> None:
    timestamps = pd.Series(pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True))
    empty_state = pd.DataFrame(index=range(1))
    levels = [
        AlignedLevel(
            timeframe="8h",
            spec=generation0.LevelSpec(
                name="round_nearest",
                family="generic_round_number",
                batch="g0b2",
                column="generic_round_nearest",
            ),
            level=np.array([100.0]),
            valid=np.array([True]),
            source_available=timestamps,
            source_open=timestamps,
            source_state=empty_state,
        ),
        AlignedLevel(
            timeframe="4h",
            spec=generation0.LevelSpec(
                name="round_nearest",
                family="generic_round_number",
                batch="g0b2",
                column="generic_round_nearest",
            ),
            level=np.array([100.05]),
            valid=np.array([True]),
            source_available=timestamps,
            source_open=timestamps,
            source_state=empty_state,
        ),
        AlignedLevel(
            timeframe="1h",
            spec=generation0.LevelSpec(
                name="rolling_high_24",
                family="generic_prior_range",
                batch="g0b2",
                column="generic_rolling_high_24",
            ),
            level=np.array([100.05]),
            valid=np.array([True]),
            source_available=timestamps,
            source_open=timestamps,
            source_state=empty_state,
        ),
    ]
    events = pd.DataFrame({"base_index": [0], "level_price": [100.0], "zone_half_width": [0.1]})

    result = attach_independent_overlap(
        events,
        aligned_levels=levels,
        real_matrix=np.array([[100.0, 100.05, 100.05]]),
        base_atr=np.array([1.0]),
        zone="tight_base_atr",
        source_index=0,
    )

    assert result.loc[0, "independent_other_level_count"] == 1


def test_g2_vp_level_persistence_resets_when_level_moves_or_is_missing() -> None:
    level = pd.Series([100.0, 100.05, 100.25, np.nan, 100.26])
    width = pd.Series([0.10] * len(level))

    result = persistence_bars(level, width)

    assert result.iloc[:3].tolist() == [1.0, 2.0, 1.0]
    assert np.isnan(result.iloc[3])
    assert result.iloc[4] == 1.0


def test_g2_vp_core_state_keeps_required_controls_without_full_technical_duplication() -> None:
    assert set(CORE_STATE_FEATURES).issubset(FULL_STATE_FEATURES)
    assert {
        "state_local_return_24h",
        "state_local_atr_pct",
        "state_local_volume_ratio",
        "state_local_pressure_6h",
        "state_source_atr_pct",
        "state_source_volume_ratio",
        "state_btc_return_24h",
        "state_top10_breadth",
        "state_selected_level_density_2atr",
        "state_vp_value_area_width_pct",
        "state_vp_level_persistence_bars",
    }.issubset(CORE_STATE_FEATURES)
    assert "state_local_rsi14" not in CORE_STATE_FEATURES
    assert "state_local_macd_hist_atr" not in CORE_STATE_FEATURES


def test_g2_vp_pair_pooling_purges_overlapping_future_paths_greedily() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["PUMP"] * 3,
            "question_id": ["lvn"] * 3,
            "attribute_assignment": ["actual"] * 3,
            "period": ["validation"] * 3,
            "level_name": ["lvn_above"] * 3,
            "approach_state": ["from_below"] * 3,
            "match_distance": [0.1, 0.2, 0.3],
            "pre_distance_atr_abs_difference": [0.01, 0.01, 0.01],
            "high_event_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-10T00:00:00Z",
                ],
                utc=True,
            ),
            "low_event_time": pd.to_datetime(
                [
                    "2026-01-05T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                    "2026-01-15T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    result = purge_overlapping_event_pairs(
        frame,
        separation_hours=48,
        group_columns=("pair", "question_id", "attribute_assignment", "period"),
        left_time_column="high_event_time",
        right_time_column="low_event_time",
    )

    assert result.index.size == 2
    assert result["match_distance"].tolist() == [0.1, 0.3]
    assert result["independent_selection_order"].tolist() == [1, 2]


def test_pair_pooling_uses_each_questions_response_horizon() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["PUMP"] * 4,
            "question_id": ["short", "short", "long", "long"],
            "attribute_assignment": ["actual"] * 4,
            "period": ["validation"] * 4,
            "level_name": ["level"] * 4,
            "approach_state": ["from_below"] * 4,
            "match_distance": [0.1, 0.2, 0.1, 0.2],
            "pre_distance_atr_abs_difference": [0.01] * 4,
            "high_event_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T10:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T10:00:00Z",
                ],
                utc=True,
            ),
            "low_event_time": pd.to_datetime(
                [
                    "2026-01-10T00:00:00Z",
                    "2026-01-10T10:00:00Z",
                    "2026-01-10T00:00:00Z",
                    "2026-01-10T10:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    result = purge_overlapping_event_pairs_by_key(
        frame,
        separation_hours_by_key={"short": 4, "long": 48},
        key_column="question_id",
        group_columns=("pair", "question_id", "attribute_assignment", "period"),
        left_time_column="high_event_time",
        right_time_column="low_event_time",
    )

    assert result.groupby("question_id").size().to_dict() == {"long": 1, "short": 2}
    assert set(result.loc[result["question_id"].eq("short"), "independence_hours"]) == {4}
    assert set(result.loc[result["question_id"].eq("long"), "independence_hours"]) == {48}


def test_contact_outcomes_have_explicit_zero_hour_metadata() -> None:
    columns = outcome_columns((1, 6))

    assert "contact_range_ratio" in columns
    assert "contact_volume_ratio" in columns
    assert "contact_pressure_change_abs" in columns
    assert outcome_metadata("contact_range_ratio") == ("contact_range_activity", 0)
    assert outcome_metadata("contact_volume_ratio") == ("contact_volume_activity", 0)
    assert outcome_metadata("contact_pressure_change_abs") == (
        "contact_pressure_change_magnitude",
        0,
    )


def test_g1d_event_targets_are_direction_neutral_and_bounded() -> None:
    frame = pd.DataFrame(index=range(2))
    for horizon in (1, 2, 4, 8, 12, 24, 48):
        frame[f"abs_excursion_atr_h{horizon}"] = [2.0, 50.0]
        frame[f"range_ratio_h{horizon}"] = [1.5, 40.0]
        frame[f"volume_ratio_h{horizon}"] = [1.2, 35.0]
        frame[f"pressure_change_h{horizon}"] = [-0.5, 3.0]
        frame[f"dwell_fraction_h{horizon}"] = [0.25, 2.0]
        frame[f"crossings_h{horizon}"] = [0.0, horizon * 2.0]

    targets = event_targets(frame)

    assert tuple(targets.columns) == TARGET_COLUMNS
    assert not any("up" in column or "down" in column for column in targets)
    assert targets.filter(like="absolute_excursion").max().max() == 30.0
    assert targets.filter(like="range_ratio").max().max() == 30.0
    assert targets.filter(like="volume_ratio").max().max() == 30.0
    assert targets.filter(like="pressure_change").max().max() == 2.0
    assert targets.filter(like="dwell_fraction").max().max() == 1.0
    assert targets.filter(like="crossing_rate").max().max() == 1.0


def test_g1d_pair_selection_purges_every_retained_future_path() -> None:
    events = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-10T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-20T00:00:00Z",
                    "2024-02-01T00:00:00Z",
                    "2024-02-10T00:00:00Z",
                ],
                utc=True,
            )
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    candidates = [
        (10, 11, 0.1),
        (12, 13, 0.2),
        (14, 15, 0.3),
    ]

    selected = select_non_overlapping_event_pairs(
        events,
        candidates,
        minimum_gap_hours=48,
    )

    assert selected == [(10, 11, 0.1), (14, 15, 0.3)]
    retained = [index for pair in selected for index in pair[:2]]
    times = events.loc[retained, "event_time"].sort_values()
    assert times.diff().dropna().min() > pd.Timedelta(hours=48)


def test_g1d_freqai_config_removes_unpurged_internal_split(tmp_path) -> None:
    base = {
        "exchange": {"pair_whitelist": []},
        "freqai": {
            "feature_parameters": {
                "plot_feature_importances": 5,
                "include_corr_pairlist": ["BTC/USDT:USDT"],
                "include_timeframes": ["1h", "4h"],
                "include_shifted_candles": 2,
            },
            "data_split_parameters": {"test_size": 0.2, "shuffle": False},
            "model_training_parameters": {"n_estimators": 100, "n_jobs": 8},
        },
    }

    config = profile_config(
        base,
        identifier="test",
        pairs=("BTC/USDT:USDT",),
        feature_dir=tmp_path,
        event_dir=tmp_path,
        train_days=365,
        backtest_days=180,
        technical_smoke=True,
    )

    assert config["freqai"]["data_split_parameters"] == {
        "test_size": 0,
        "shuffle": False,
    }
    assert config["freqai"]["feature_parameters"]["include_corr_pairlist"] == []
    assert config["freqai"]["model_training_parameters"]["n_jobs"] == 1
    assert config["freqai"]["model_training_parameters"]["n_estimators"] == 20
    assert config["market_reaction_zone_g1d"]["train_prediction_embargo_hours"] == 48


def test_g1d_profile_controls_are_attached_without_merge_suffix_collisions() -> None:
    rows: list[dict[str, object]] = []
    for position, profile_id in enumerate(PROFILES, start=1):
        row: dict[str, object] = {
            "profile_id": profile_id,
            "pair": "BTC/USDT:USDT",
            "period": "validation_early",
            "scope": "actual_contacts",
            "target": "&-event_absolute_excursion_atr_1h",
        }
        row.update({metric: float(position) for metric in SCORE_METRICS})
        rows.append(row)
    scores = pd.DataFrame(rows)

    result = attach_profile_deltas(scores)
    by_profile = result.set_index("profile_id")

    assert (
        by_profile.loc["native_current", "incremental_baseline_prediction_actual_spearman"]
        == by_profile.loc["crypto_wide_state", "prediction_actual_spearman"]
    )
    assert (
        by_profile.loc["vp_current", "matching_placebo_prediction_actual_spearman"]
        == by_profile.loc["vp_placebo", "prediction_actual_spearman"]
    )
    assert (
        by_profile.loc["cluster_current", "matching_placebo_prediction_actual_spearman"]
        == by_profile.loc["cluster_placebo", "prediction_actual_spearman"]
    )
    assert not any(column.endswith(("_x", "_y")) for column in result.columns)


def test_g1d_training_labels_embargo_last_48_hours(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    events = pd.DataFrame({"date": dates})
    for target in TARGET_COLUMNS:
        events[target] = 1.0
    strategy = object.__new__(MarketReactionZoneG1DLocalStateFreqAIResearchStrategy)
    monkeypatch.setattr(strategy, "_event_cache", lambda pair: events)
    frame = pd.DataFrame({"date": dates.astype("datetime64[ms, UTC]")})

    labelled = strategy.set_freqai_targets(frame, metadata={"pair": "BTC/USDT:USDT"})

    cutoff = dates.max() - pd.Timedelta(hours=48)
    assert labelled.loc[dates <= cutoff, list(TARGET_COLUMNS)].notna().all().all()
    assert labelled.loc[dates > cutoff, list(TARGET_COLUMNS)].isna().all().all()


def _g2d_anchor_base() -> pd.DataFrame:
    close = np.asarray(
        [1.5, 2.5, 3.5, 9.5, 6.5, 5.5, 4.5, 3.5, 2.5, 3.0, 4.0, 5.0],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=len(close), freq="1h", tz="UTC"),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.arange(1.0, len(close) + 1.0),
            "pre_volume_median_24": np.full(len(close), 6.0),
        }
    )


def test_g2d_reset_anchor_excludes_contact_and_future_candles() -> None:
    base = _g2d_anchor_base()
    original = reset_anchor_stats(
        base,
        family="utc_daily_reset",
        frequency="D",
        delay_hours=24,
    )
    event_position = 7
    typical = (base["high"] + base["low"] + base["close"]) / 3.0
    expected = np.average(
        typical.iloc[:event_position],
        weights=base["volume"].iloc[:event_position],
    )

    changed = base.copy()
    changed.loc[event_position:, ["high", "low", "close"]] = 10_000.0
    changed.loc[event_position:, "volume"] = 1_000_000.0
    recalculated = reset_anchor_stats(
        changed,
        family="utc_daily_reset",
        frequency="D",
        delay_hours=24,
    )

    assert original.vwap[event_position] == pytest.approx(expected)
    assert recalculated.vwap[event_position] == pytest.approx(expected)
    assert recalculated.weighted_std[event_position] == pytest.approx(
        original.weighted_std[event_position]
    )


def test_g2d_swing_anchor_starts_only_after_full_right_side_confirmation() -> None:
    base = _g2d_anchor_base()
    stats = swing_anchor_stats(base, side="high")
    pivot_position = 3
    first_available_position = pivot_position + 3 + 1

    assert np.all(stats.start_index[:first_available_position] == -1)
    assert stats.start_index[first_available_position] == pivot_position
    assert stats.source_open.iloc[first_available_position] == base["date"].iloc[pivot_position]
    assert stats.age[first_available_position] == 4.0


def test_g2d_all_anchor_levels_are_unchanged_by_later_data() -> None:
    base = _g2d_anchor_base()
    event_position = 8
    original = causal_anchor_levels(base)
    changed = base.copy()
    changed.loc[event_position + 1 :, ["high", "low", "close"]] *= 1_000.0
    changed.loc[event_position + 1 :, "volume"] *= 1_000_000.0
    recalculated = causal_anchor_levels(changed)

    assert [level.name for level in recalculated] == [level.name for level in original]
    for left, right in zip(original, recalculated, strict=True):
        np.testing.assert_allclose(
            left.level[: event_position + 1],
            right.level[: event_position + 1],
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            left.valid[: event_position + 1],
            right.valid[: event_position + 1],
        )


def test_g2d_stores_one_wide_row_per_matched_response_window() -> None:
    def event_frame(event_time: str, addition: float) -> pd.DataFrame:
        rows = []
        for sample in range(4):
            row: dict[str, object] = {
                "event_time": pd.Timestamp(event_time) + pd.Timedelta(days=sample),
                "period": "validation_early",
                "approach_state": "from_below",
                "pre_distance_atr": 1.0 + sample / 100.0,
                "overlap_anchor_level_count": 1,
                "overlap_other_anchor_family_count": 0,
                "base_index": sample,
            }
            row.update({feature: float(sample) for feature in STATE_FEATURES})
            row.update(
                {outcome: position + sample + addition for position, outcome in enumerate(OUTCOMES)}
            )
            rows.append(row)
        return pd.DataFrame(rows)

    actual = event_frame("2026-01-01T00:00:00Z", 1.0)
    control = event_frame("2026-02-01T00:00:00Z", 2.0)
    anchor = AnchorLevel(
        family="utc_daily_reset",
        name="utc_daily_reset_center",
        multiplier=0,
        level=np.asarray([1.0]),
        sma_control=np.asarray([1.0]),
        ema_control=np.asarray([1.0]),
        valid=np.asarray([True]),
        source_open=pd.Series(pd.to_datetime(["2025-12-31"], utc=True)),
        age=np.asarray([2.0]),
        average_volume_ratio=np.asarray([1.0]),
        delay_hours=24,
    )

    rows = matched_outcome_rows(
        actual=actual,
        control=control,
        pair="PUMP/USDT:USDT",
        anchor=anchor,
        anchor_scope="isolated_anchor_family",
        control_name="ordinary_sma_same_anchor_horizon",
        state_features=STATE_FEATURES,
    )

    assert len(rows) == 4 * len(OUTCOMES_BY_RESPONSE_WINDOW) == 12
    assert {row["response_window"] for row in rows} == set(OUTCOMES_BY_RESPONSE_WINDOW)
    assert all("outcome" not in row for row in rows)
    for row in rows:
        for outcome in OUTCOMES_BY_RESPONSE_WINDOW[row["response_window"]]:
            assert row[f"delta__{outcome}"] == pytest.approx(-1.0)


def test_g2d_vectorized_balance_matches_reference_formula() -> None:
    features = STATE_FEATURES[:4]
    actual = pd.DataFrame(
        {
            feature: np.asarray([1.0, 2.0, 4.0, 7.0]) + position
            for position, feature in enumerate(features)
        }
    )
    control = pd.DataFrame(
        {
            feature: np.asarray([1.5, 2.5, 3.5, 6.5]) + position
            for position, feature in enumerate(features)
        }
    )
    matched = pd.DataFrame(
        {
            **{f"actual_state__{feature}": actual[feature] for feature in features},
            **{f"control_state__{feature}": control[feature] for feature in features},
        }
    )

    expected = balance_diagnostics(actual, control, features)
    observed = matched_pair_balance(matched, features)

    assert observed["features_scored"] == expected["features_scored"]
    assert observed["max_absolute_smd"] == pytest.approx(expected["max_absolute_smd"])
    assert observed["median_absolute_smd"] == pytest.approx(expected["median_absolute_smd"])


def test_g2d_balance_uses_each_controls_declared_matching_features() -> None:
    frame = pd.DataFrame(
        {
            "matching_state_features": ["a;b"] * 4,
            "actual_state__a": [0.0, 1.0, 2.0, 3.0],
            "control_state__a": [0.1, 1.1, 2.1, 3.1],
            "actual_state__b": [2.0, 3.0, 4.0, 5.0],
            "control_state__b": [2.1, 3.1, 4.1, 5.1],
            "actual_state__unmatched": [0.0, 1.0, 2.0, 3.0],
            "control_state__unmatched": [100.0, 101.0, 102.0, 103.0],
        }
    )

    declared = declared_matching_balance(frame)
    all_fields = matched_pair_balance(frame, ("a", "b", "unmatched"))

    assert declared["features_scored"] == 2
    assert declared["max_absolute_smd"] < 0.5
    assert all_fields["max_absolute_smd"] > 50.0


def test_g2d_leave_one_coin_out_uses_equal_coin_values() -> None:
    pair_period = pd.DataFrame(
        {
            "route_id": ["g2d"] * 3,
            "anchor_family": ["daily"] * 3,
            "level_name": ["centre"] * 3,
            "dispersion_multiplier": [0] * 3,
            "anchor_scope": ["isolated"] * 3,
            "control": ["shifted"] * 3,
            "period": ["validation"] * 3,
            "response_window": ["h4"] * 3,
            "outcome": ["volume_ratio_h4"] * 3,
            "pair": ["A", "B", "C"],
            "pair_period_coverage_eligible": [True] * 3,
            "delta_mean": [1.0, -2.0, 3.0],
            "independent_event_pairs": [10, 20, 30],
        }
    )

    result = leave_one_coin_out_results(pair_period).set_index("omitted_pair")

    assert result.loc["A", "remaining_independent_event_pairs"] == 50
    assert result.loc["A", "equal_coin_delta_median"] == pytest.approx(0.5)
    assert result.loc["A", "equal_coin_positive_fraction"] == pytest.approx(0.5)
    assert result.loc["B", "equal_coin_delta_median"] == pytest.approx(2.0)
    assert result.loc["B", "equal_coin_positive_fraction"] == pytest.approx(1.0)


def _g2e_density_frame(rows: int = 220) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    close = 100.0 + np.where((positions.astype(int) % 12) < 8, 0.10, 1.10)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC"),
            "open": close,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": 1.0,
            "base_atr": 1.0,
            "pre_close": pd.Series(close).shift(1),
        }
    )


def test_g2e_repeated_close_density_excludes_current_and_future_prices() -> None:
    frame = _g2e_density_frame()
    event_position = 200
    original = causal_density_zones(
        frame,
        family="repeated_close_density",
        history_hours=168,
        bin_width_atr=0.25,
    )
    changed = frame.copy()
    changed.loc[event_position:, ["open", "high", "low", "close"]] = 10_000.0
    recalculated = causal_density_zones(
        changed,
        family="repeated_close_density",
        history_hours=168,
        bin_width_atr=0.25,
    )

    np.testing.assert_allclose(
        original.levels[: event_position + 1],
        recalculated.levels[: event_position + 1],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        original.half_widths[: event_position + 1],
        recalculated.half_widths[: event_position + 1],
        equal_nan=True,
    )


def test_g2e_swing_prices_wait_for_full_confirmation() -> None:
    frame = _g2e_density_frame(rows=20)
    frame.loc[:, "high"] = 101.0
    frame.loc[:, "low"] = 99.0
    frame.loc[3, "high"] = 110.0
    frame.loc[10, "low"] = 90.0

    swings = confirmed_swing_points(frame)

    high_index = int(np.flatnonzero(swings.pivot_positions == 3)[0])
    low_index = int(np.flatnonzero(swings.pivot_positions == 10)[0])
    assert swings.available_positions[high_index] == 7
    assert swings.available_positions[low_index] == 14
    assert swings.prices[high_index] == 110.0
    assert swings.prices[low_index] == 90.0


def test_g2e_connected_density_merges_only_adjacent_selected_bins() -> None:
    source = np.asarray([10.1] * 4 + [11.1] * 4 + [15.1] * 4, dtype=float)

    candidates = connected_density_candidates(
        source,
        atr=1.0,
        reference=11.0,
        family="repeated_close_density",
        history_hours=168,
        bin_width_atr=1.0,
    )

    assert len(candidates) == 2
    assert candidates[0][0] == pytest.approx(10.6)
    assert candidates[0][1] == pytest.approx(1.4)
    assert candidates[0][2] == 8.0
    assert candidates[1][0] == pytest.approx(15.1)


def test_g2e_coverage_preflight_contains_no_reaction_or_direction_score() -> None:
    frame = _g2e_density_frame()
    zones = causal_density_zones(
        frame,
        family="repeated_close_density",
        history_hours=168,
        bin_width_atr=0.25,
    )
    development = np.ones(len(frame), dtype=bool)

    record = coverage_record(
        pair="PUMP/USDT:USDT",
        frame=frame,
        zones=zones,
        development=development,
    )

    assert record["reaction_outcomes_calculated"] is False
    assert record["direction_prediction"] is False
    assert record["profit_optimization"] is False
    measured_keys = set(record).difference(
        {"reaction_outcomes_calculated", "direction_prediction", "profit_optimization"}
    )
    assert not any(
        token in key
        for key in measured_keys
        for token in ("future", "profit", "return_after", "direction_score")
    )


def test_g2e_manifest_analysis_end_supports_both_frozen_manifest_forms() -> None:
    explicit = {
        "data": {
            "analysis_end_utc_exclusive": "2026-07-14T00:00:00Z",
            "chronological_periods": [],
        }
    }
    period_bounded = {
        "data": {
            "analysis_end": "latest common causally usable candle",
            "chronological_periods": [
                {"end_utc_exclusive": "2024-01-01T00:00:00Z"},
                {"end_utc_exclusive": "2026-07-20T00:00:00Z"},
            ],
        }
    }

    assert manifest_analysis_end_exclusive(explicit) == pd.Timestamp("2026-07-14T00:00:00Z")
    assert manifest_analysis_end_exclusive(period_bounded) == pd.Timestamp("2026-07-20T00:00:00Z")


def test_g2e_shuffled_swing_control_is_causal_and_not_an_identity_copy() -> None:
    frame = _g2e_density_frame(rows=420)
    positions = np.arange(len(frame), dtype=float)
    oscillating = 100.0 + 2.0 * np.sin(positions * 2.0 * np.pi / 12.0)
    frame.loc[:, "open"] = oscillating
    frame.loc[:, "high"] = oscillating + 0.25
    frame.loc[:, "low"] = oscillating - 0.25
    frame.loc[:, "close"] = oscillating
    frame.loc[:, "pre_close"] = pd.Series(oscillating).shift(1)
    original_swings = confirmed_swing_points(frame)
    actual = causal_density_zones(
        frame,
        family="confirmed_swing_price_density",
        history_hours=168,
        bin_width_atr=0.25,
        swings=original_swings,
    )
    shuffled = g2e_reaction.causal_shuffled_swing_zones(
        frame,
        history_hours=168,
        bin_width_atr=0.25,
        swings=original_swings,
    )
    finite = np.isfinite(actual.levels) & np.isfinite(shuffled.levels)
    assert finite.any()
    assert np.any(np.abs(actual.levels[finite] - shuffled.levels[finite]) > 1e-9)

    event_position = 320
    changed = frame.copy()
    changed.loc[event_position:, ["open", "high", "low", "close"]] = 10_000.0
    changed_swings = confirmed_swing_points(changed)
    recalculated = g2e_reaction.causal_shuffled_swing_zones(
        changed,
        history_hours=168,
        bin_width_atr=0.25,
        swings=changed_swings,
    )
    np.testing.assert_allclose(
        shuffled.levels[: event_position + 1],
        recalculated.levels[: event_position + 1],
        equal_nan=True,
    )


def test_g2e_density_scope_keeps_single_and_cluster_contacts_separate() -> None:
    events = pd.DataFrame(
        {
            "overlap_any_reference_level": [False, False, True],
            "overlap_other_density_zone_count": [0, 2, 0],
        }
    )

    scopes = g2e_reaction.classify_actual_scope(events)

    assert scopes.tolist() == [
        "single_density_zone",
        "density_cluster",
        "density_plus_reference_cluster",
    ]


def test_g2e_union_coverage_merges_overlaps_without_double_counting() -> None:
    levels = np.asarray([[99.5, 100.0, 104.0]], dtype=float)
    widths = np.asarray([[1.0, 1.0, 0.5]], dtype=float)

    coverage = g2e_reaction.matrix_union_coverage(
        level_matrix=levels,
        width_matrix=widths,
        base_atr=np.asarray([1.0]),
        pre_close=np.asarray([100.0]),
    )

    assert coverage[0] == pytest.approx(2.5 / 4.0)


def test_g2e_empty_control_overlap_filter_stays_empty() -> None:
    empty = pd.DataFrame()

    filtered = g2e_reaction.without_density_overlap(empty)

    assert filtered.empty


def test_g2e_reaction_guard_matches_the_frozen_branch_identifier() -> None:
    g2e_reaction.validate_frozen_branch()


def test_g2e_reaction_uses_the_shared_response_window_purge_contract() -> None:
    times = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
        utc=True,
    )
    matches = pd.DataFrame(
        {
            "route_id": ["g2e_causal_swing_price_density_zones"] * 2,
            "density_family": ["repeated_close_density"] * 2,
            "history_hours": [168] * 2,
            "level_name": ["repeated_close_density__168h"] * 2,
            "actual_scope": ["single_density_zone"] * 2,
            "control": ["shift_+1atr"] * 2,
            "pair": ["PUMP/USDT:USDT"] * 2,
            "period": ["meme_validation_early"] * 2,
            "approach_state": ["from_below"] * 2,
            "response_window": ["h4"] * 2,
            "actual_event_time": times,
            "control_event_time": times + pd.Timedelta(days=10),
            "match_distance": [0.2, 0.1],
            "pre_distance_atr_abs_difference": [0.02, 0.01],
        }
    )

    retained = g2e_reaction.purge_density_matches(matches)
    group_columns = (
        "route_id",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "pair",
        "period",
        "approach_state",
        "response_window",
    )
    reference = purge_overlapping_event_pairs_by_key(
        matches,
        separation_hours_by_key=g2e_reaction.SEPARATION_HOURS_BY_RESPONSE_WINDOW,
        key_column="response_window",
        group_columns=group_columns,
        left_time_column="actual_event_time",
        right_time_column="control_event_time",
    )

    assert len(retained) == 1
    assert retained.loc[0, "actual_event_time"] == times[1]
    assert retained.loc[0, "independence_hours"] == 4
    compare = [
        "actual_event_time",
        "control_event_time",
        "independent_selection_order",
        "independence_hours",
    ]
    pd.testing.assert_frame_equal(
        retained[compare].reset_index(drop=True),
        reference[compare].reset_index(drop=True),
    )


def test_g2e_balance_uses_each_controls_declared_matching_features() -> None:
    frame = pd.DataFrame(
        {
            "matching_state_features": ["a;b"] * 4,
            "actual_state__a": [0.0, 1.0, 2.0, 3.0],
            "control_state__a": [0.1, 1.1, 2.1, 3.1],
            "actual_state__b": [2.0, 3.0, 4.0, 5.0],
            "control_state__b": [2.1, 3.1, 4.1, 5.1],
            "actual_state__unmatched": [0.0, 1.0, 2.0, 3.0],
            "control_state__unmatched": [100.0, 101.0, 102.0, 103.0],
        }
    )

    declared = g2e_reaction.declared_matching_balance(frame)
    all_fields = matched_pair_balance(frame, ("a", "b", "unmatched"))

    assert declared["features_scored"] == 2
    assert declared["max_absolute_smd"] < 0.5
    assert all_fields["max_absolute_smd"] > 50.0


def test_g2f_causal_terciles_ignore_future_mutation() -> None:
    values = pd.Series(np.linspace(1.0, 5.0, 1_000))
    original = causal_tercile_bucket(values)
    changed = values.copy()
    changed.iloc[800:] = 10_000.0
    recalculated = causal_tercile_bucket(changed)

    pd.testing.assert_series_equal(original.iloc[:800], recalculated.iloc[:800])
    assert set(original.iloc[300:].unique()).issubset({-1, 0, 1})


def test_g2f_coherent_regime_combinations_require_aligned_inputs() -> None:
    left = pd.Series([-1, -1, 1, 1, 0], dtype="int8")
    right = pd.Series([-1, 1, 1, 0, -1], dtype="int8")

    assert coherent_pair_regime(left, right).tolist() == [-1, 0, 1, 0, 0]
    votes = np.asarray(
        [
            [-1, -1, 0],
            [1, 1, -1],
            [1, -1, 0],
        ],
        dtype=np.int8,
    )
    assert vote_regime(votes, required=2).tolist() == [-1, 1, 0]


def test_g2f_event_indexes_attach_causal_regimes_and_pressure() -> None:
    rows = 260
    dates = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    close = pd.Series(100.0 + np.arange(rows, dtype=float) * 0.01)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
            "candle_pressure": np.linspace(-0.5, 0.5, rows),
            "pre_pressure_mean_24": pd.Series(np.linspace(-0.5, 0.5, rows)).shift(1),
        }
    )
    state = pd.DataFrame({"date": dates})
    for lens in LENSES:
        state[f"regime__{lens.name}"] = -1
        for column in lens.raw_state_columns:
            state[column] = np.linspace(0.0, 1.0, rows)
    events = pd.DataFrame(
        {
            "actual_base_index": [200],
            "control_base_index": [220],
            "actual_event_time": [dates[200]],
            "control_event_time": [dates[220]],
        }
    )

    attached, violations = attach_pair_regimes(events, base=base, state=state)

    assert violations == 0
    assert bool(attached.loc[0, "regime_agreement__quiet_vs_expanding"])
    assert attached.loc[0, "pair_regime__quiet_vs_expanding"] == -1
    assert np.isfinite(attached.loc[0, "delta__pressure_change_abs_h24"])


def test_g2f_validates_explicit_g2ab_integrity_without_inventing_pass_flag() -> None:
    integrity = {
        "pair_failures": 0,
        "matched_event_pairs_before_overlap_purge": 100,
        "independent_event_pairs_after_horizon_purge": 60,
        "pre_distance_atr_abs_difference_max": 0.099,
        "event_separation_hours_min": 5.0,
        "period_end_embargo_hours": 48,
        "periods_present": ["development", "validation"],
        "routes_present": ["g2b_lvn_thinness"],
        "direction_prediction": False,
        "profit_optimization": False,
    }

    validate_source_integrity_contract(integrity)
    invalid = dict(integrity, pre_distance_atr_abs_difference_max=0.11)
    with pytest.raises(ValueError, match="0.10 ATR caliper"):
        validate_source_integrity_contract(invalid)


def test_g3a_largest_connected_interval_count_uses_transitive_overlap() -> None:
    assert g3a_attribution.largest_connected_interval_count([]) == 0
    assert (
        g3a_attribution.largest_connected_interval_count(
            [(0.0, 1.0), (0.9, 2.0), (1.9, 3.0), (4.0, 5.0)]
        )
        == 3
    )
    assert (
        g3a_attribution.largest_connected_interval_count([(4.0, 5.0), (0.0, 1.0), (2.0, 3.0)]) == 1
    )


def test_g3a_connected_dependency_groups_do_not_count_aliases_as_independent() -> None:
    intervals = [
        (0.0, 1.0, "price_average_family"),
        (0.9, 2.0, "price_average_family"),
        (1.9, 3.0, "rolling_price_extreme"),
        (4.0, 5.0, "round_number"),
    ]

    assert g3a_attribution.largest_connected_dependency_group_count(intervals) == 2
    assert (
        g3a_attribution.largest_connected_dependency_group_count(
            [
                (0.0, 1.0, "price_average_family"),
                (0.9, 2.0, "price_average_family"),
            ]
        )
        == 1
    )


def test_g3g_independence_uses_frozen_hash_priority() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["BTC/USDT:USDT"] * 3,
            "event_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-04T00:00:00Z"]
            ),
            "selection_hash": ["b", "a", "c"],
        }
    )

    selected = g3g_replay.select_time_independent(frame, hours=48)

    assert selected["selection_hash"].tolist() == ["a", "c"]
    assert selected["independence_hours"].eq(48).all()


def test_g3g_contact_components_use_transitive_price_overlap() -> None:
    peers = [
        {"contacted": True, "lower": 0.0, "upper": 1.0, "aligned_index": 0},
        {"contacted": True, "lower": 0.9, "upper": 2.0, "aligned_index": 1},
        {"contacted": True, "lower": 1.9, "upper": 3.0, "aligned_index": 2},
        {"contacted": True, "lower": 4.0, "upper": 5.0, "aligned_index": 3},
        {"contacted": False, "lower": 0.0, "upper": 5.0, "aligned_index": 4},
    ]

    g3g_replay.assign_contact_components(peers)

    assert [peer["contact_component_id"] for peer in peers] == [0, 0, 0, 1, None]


def test_g3g_acquisition_keeps_disjoint_same_pair_windows_separate() -> None:
    frame = pd.DataFrame(
        {
            "episode_ids": ["old", "new"],
            "cohorts": ["normal", "meme"],
            "pair": ["DOGE/USDT:USDT", "DOGE/USDT:USDT"],
            "parent_contact_hours_utc": ["2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"],
            "causal_anchor_timeframes": ["1h", "4h"],
            "maximum_visible_pre_hours": [24, 72],
            "maximum_visible_post_hours": [12, 48],
            "feature_warmup_hours": [24, 24],
            "visible_start_utc": pd.to_datetime(["2024-12-31", "2025-12-29"], utc=True),
            "visible_end_exclusive_utc": pd.to_datetime(["2025-01-02", "2026-01-03"], utc=True),
            "interval_start_utc": pd.to_datetime(["2024-12-30", "2025-12-28"], utc=True),
            "interval_end_exclusive_utc": pd.to_datetime(["2025-01-02", "2026-01-03"], utc=True),
            "exchange": ["binance", "binance"],
            "market_type": ["futures", "futures"],
            "timeframe": ["1m", "1m"],
            "data_format": ["feather", "feather"],
        }
    )

    intervals = g3g_replay.merge_overlapping_acquisition_intervals(frame)

    assert len(intervals) == 2
    assert intervals["pair"].eq("DOGE/USDT:USDT").all()
    assert intervals["timerange"].str.fullmatch(r"\d{10}-\d{10}").all()


def test_g3g_technical_snapshot_uses_only_fully_completed_bars() -> None:
    dates = pd.date_range("2026-01-01", periods=260, freq="1min", tz="UTC")
    source = pd.DataFrame(
        {
            "date": dates,
            "open": np.arange(260, dtype=float) + 100.0,
            "high": np.arange(260, dtype=float) + 101.0,
            "low": np.arange(260, dtype=float) + 99.0,
            "close": np.arange(260, dtype=float) + 100.5,
            "volume": np.arange(260, dtype=float) + 1.0,
        }
    )
    technical = g3g_analysis.technical_frame(source, timeframe="1m")
    timestamp = dates[250]
    snapshot = g3g_analysis.technical_snapshot({"1m": technical}, timestamp)

    expected = technical.loc[technical["available_at"].le(timestamp)].iloc[-1]
    future = technical.loc[technical["date"].eq(timestamp)].iloc[0]
    assert snapshot["tf_1m__return_1bar"] == pytest.approx(expected["return_1bar"])
    assert snapshot["tf_1m__volume_ratio_20bar"] == pytest.approx(
        expected["volume_ratio_20bar"]
    )
    assert snapshot["tf_1m__volume_ratio_20bar"] != pytest.approx(
        future["volume_ratio_20bar"]
    )
    resampled = g3g_analysis.resample_ohlcv(source, "5m")
    assert len(resampled) == 52
    assert resampled.iloc[0]["volume"] == pytest.approx(source.iloc[:5]["volume"].sum())


def test_g3g_first_direction_uses_exact_minute_order_not_checkpoint_order() -> None:
    path = {
        "checkpoint_minutes": 60,
        "first_up_threshold_minute": 4,
        "first_down_threshold_minute": 17,
        "first_zone_breakout_minute": 8,
        "first_zone_rejection_minute": 21,
    }

    assert g3g_analysis.first_direction_from_path_rows([path]) == "up"
    path["first_up_threshold_minute"] = 9
    path["first_down_threshold_minute"] = 9
    assert g3g_analysis.first_direction_from_path_rows([path]) == "tie"
    path["first_up_threshold_minute"] = None
    path["first_down_threshold_minute"] = None
    assert g3g_analysis.first_direction_from_path_rows([path]) == "unreached"
    assert g3g_analysis.first_zone_resolution_from_path_rows([path]) == "breakout"
    path["first_zone_rejection_minute"] = 5
    assert g3g_analysis.first_zone_resolution_from_path_rows([path]) == "rejection"


def test_g3g_minute_match_state_is_shifted_before_pseudo_contact() -> None:
    dates = pd.date_range("2026-01-01", periods=300, freq="1min", tz="UTC")
    source = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100.0, 130.0, 300),
            "high": np.linspace(100.5, 130.5, 300),
            "low": np.linspace(99.5, 129.5, 300),
            "close": np.linspace(100.1, 130.1, 300),
            "volume": np.linspace(1.0, 300.0, 300),
        }
    )
    matched = g3g_analysis.minute_match_frame(source)
    position = 280
    expected_return = source["close"].iloc[position - 1] / source["close"].iloc[position - 16] - 1.0

    assert matched["match_return_15m"].iloc[position] == pytest.approx(expected_return)


def test_g3g_boundary_first_true_position_is_one_based() -> None:
    values = pd.Series([False, False, True, True])

    assert g3g_analysis.first_true_position(values) == 3
    assert g3g_analysis.first_true_position(pd.Series([False, False])) is None


def test_g3g_expanded_acquisition_doubles_every_episode_window() -> None:
    sample = pd.DataFrame(
        {
            "episode_id": ["episode"],
            "cohort": ["normal"],
            "pair": ["ETH/USDT:USDT"],
            "event_time": pd.to_datetime(["2026-01-10T00:00:00Z"]),
            "cluster_causal_anchor_timeframe": ["1h"],
        }
    )

    default = g3g_analysis.analysis_acquisition_intervals(
        sample, expanded_boundaries=False
    ).iloc[0]
    expanded = g3g_analysis.analysis_acquisition_intervals(
        sample, expanded_boundaries=True
    ).iloc[0]

    assert default["maximum_visible_pre_hours"] == 24
    assert default["maximum_visible_post_hours"] == 12
    assert expanded["maximum_visible_pre_hours"] == 48
    assert expanded["maximum_visible_post_hours"] == 24
    assert expanded["feature_warmup_hours"] == default["feature_warmup_hours"]


def test_g3g_directional_excursion_is_zero_when_price_never_travels_that_way() -> None:
    minute = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:02:00Z",
                ]
            ),
            "open": [100.0, 100.0, 98.0],
            "high": [101.0, 101.0, 99.0],
            "low": [99.0, 80.0, 97.0],
            "close": [100.0, 100.0, 98.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    path = g3g_analysis.checkpoint_paths(
        minute=minute,
        timestamp=pd.Timestamp("2026-01-01T00:01:00Z"),
        reference_price=110.0,
        zone_half_width=1.0,
        level_name="lvn_above",
        checkpoints=(1,),
        episode_id="episode",
        cohort="normal",
        event_kind="actual_cluster_contact",
    )[0]

    assert path["maximum_up_excursion_half_widths"] == 0.0
    assert path["maximum_down_excursion_half_widths"] == pytest.approx(3.0)
    assert path["close_displacement_through_positive_half_widths"] == pytest.approx(-2.0)
    assert path["decision_time_after_contact_bar"] == pd.Timestamp("2026-01-01T00:02:00Z")
    assert path["pressure_change_toward_away"] == pytest.approx(
        -path["pressure_change_toward_through"]
    )


def test_g3a_equivalence_mismatches_handles_tolerance_and_missing_values() -> None:
    left = pd.Series([1.0, np.nan, 3.0, 4.0])
    right = pd.Series([1.0 + 5e-11, np.nan, np.nan, 5.0])

    mismatches, maximum = g3a_attribution.equivalence_mismatches(left, right)

    assert mismatches == 2
    assert maximum == pytest.approx(1.0)


def test_g3a_meme_daily_cache_is_the_declared_eventless_peer_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_event_source(**_: object) -> tuple[Path, Path, dict[str, object]]:
        raise AssertionError("The frozen meme atlas has no 1d event surface.")

    monkeypatch.setattr(g3a_attribution, "event_source", unexpected_event_source)

    reference = g3a_attribution.frozen_event_reference(
        cohort="meme",
        manifest={},
        manifest_path=Path("manifest.json"),
        pair="PUMP/USDT:USDT",
        timeframe="1d",
    )

    assert reference is None


def test_g3b_through_distance_is_relative_to_the_approach_side() -> None:
    assert g3b_obstacles.through_signed_distance(
        "from_below", target=100.0, candidate=101.0, atr=2.0
    ) == pytest.approx(0.5)
    assert g3b_obstacles.through_signed_distance(
        "from_above", target=100.0, candidate=99.0, atr=2.0
    ) == pytest.approx(0.5)
    assert g3b_obstacles.through_signed_distance(
        "from_below", target=100.0, candidate=99.0, atr=2.0
    ) == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="Unsupported approach state"):
        g3b_obstacles.through_signed_distance(
            "already_inside_or_unclear", target=100.0, candidate=101.0, atr=2.0
        )


@pytest.mark.parametrize(
    ("gap", "room_class", "contrast"),
    [
        (None, "no_ahead_higher_tf_level", "open_room"),
        (-0.1, "overlapping_ahead_zone", "close_obstacle"),
        (0.5, "close_obstacle_gap_le_0_5atr", "close_obstacle"),
        (1.5, "intermediate_gap_0_5_to_1_5atr", "intermediate_room"),
        (1.5001, "open_room_gap_gt_1_5atr", "open_room"),
    ],
)
def test_g3b_room_classes_are_frozen_before_outcomes(
    gap: float | None,
    room_class: str,
    contrast: str,
) -> None:
    assert g3b_obstacles.classify_room(gap) == (room_class, contrast)


def test_g3d_classifies_fresh_arrival_only_when_zone_was_known_and_area_was_clear() -> None:
    levels = np.full((10, 3), np.nan)
    widths = np.full((10, 3), np.nan)
    valid = np.zeros((10, 3), dtype=bool)
    levels[:, 0] = 100.0
    widths[:, 0] = 1.0
    valid[:, 0] = True
    zones = g3d_arrival.DensityZoneSet(
        family="confirmed_swing_price_density",
        history_hours=168,
        bin_width_atr=0.25,
        levels=levels,
        half_widths=widths,
        valid=valid,
        support_counts=np.ones_like(levels),
        source_counts=np.full(10, 10.0),
    )
    high = np.full(10, 98.0)
    low = np.full(10, 97.0)
    pre_close = np.full(10, 98.0)

    state, approach, contacts, continuity = g3d_arrival.classify_density_event(
        index=7,
        level=100.0,
        half_width=1.0,
        event_kind="contact",
        zones=zones,
        high=high,
        low=low,
        pre_close=pre_close,
    )

    assert state == "first_arrival_after_outside_interval"
    assert approach == "from_below"
    assert contacts == 0
    assert continuity == 6


def test_g3d_separates_inside_repeat_near_miss_and_unstable_zone() -> None:
    levels = np.full((10, 3), np.nan)
    widths = np.full((10, 3), np.nan)
    valid = np.zeros((10, 3), dtype=bool)
    levels[:, 0] = 100.0
    widths[:, 0] = 1.0
    valid[:, 0] = True
    zones = g3d_arrival.DensityZoneSet(
        family="repeated_close_density",
        history_hours=168,
        bin_width_atr=0.25,
        levels=levels,
        half_widths=widths,
        valid=valid,
        support_counts=np.ones_like(levels),
        source_counts=np.full(10, 10.0),
    )
    high = np.full(10, 98.0)
    low = np.full(10, 97.0)
    pre_close = np.full(10, 98.0)
    high[3] = 100.0

    repeat = g3d_arrival.classify_density_event(
        index=7,
        level=100.0,
        half_width=1.0,
        event_kind="contact",
        zones=zones,
        high=high,
        low=low,
        pre_close=pre_close,
    )
    assert repeat[:3] == ("repeat_contact", "from_below", 1)

    pre_close[7] = 100.0
    inside = g3d_arrival.classify_density_event(
        index=7,
        level=100.0,
        half_width=1.0,
        event_kind="contact",
        zones=zones,
        high=high,
        low=low,
        pre_close=pre_close,
    )
    assert inside[0] == "already_inside"

    pre_close[7] = 98.0
    high[3] = 98.0
    near = g3d_arrival.classify_density_event(
        index=7,
        level=100.0,
        half_width=1.0,
        event_kind="near_miss",
        zones=zones,
        high=high,
        low=low,
        pre_close=pre_close,
    )
    assert near[0] == "near_miss"

    valid[2, 0] = False
    unstable = g3d_arrival.classify_density_event(
        index=7,
        level=100.0,
        half_width=1.0,
        event_kind="contact",
        zones=zones,
        high=high,
        low=low,
        pre_close=pre_close,
    )
    assert unstable[0] == "unstable_or_new_zone"


def test_g3d_counts_supported_cells_once_instead_of_once_per_period() -> None:
    support = pd.DataFrame(
        {
            "density_family": ["swing", "swing", "swing", "close"],
            "history_hours": [168, 168, 168, 720],
            "actual_scope": ["cluster", "cluster", "cluster", "cluster"],
            "both_validation_periods_supported": [True, True, True, False],
        }
    )

    assert g3d_arrival.supported_both_validation_cell_count(support) == 1


def test_g3d_control_source_filter_opens_only_preflight_supported_cells() -> None:
    geometry = pd.DataFrame(
        {
            "density_family": ["swing", "swing", "close"],
            "history_hours": [168, 720, 168],
            "actual_scope": ["cluster", "cluster", "single"],
            "event_time": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
        }
    )

    filtered = g3d_controls.filter_supported_cells(
        geometry,
        (("swing", 168, "cluster"),),
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["history_hours"] == 168


def test_g3d_event_state_controls_use_explicit_geometry_matching() -> None:
    assert g3d_states.matching_geometry_column("near_miss") == "outside_edge_gap_atr"
    assert g3d_states.matching_geometry_column("repeat_contact") == "outside_edge_gap_atr"
    assert g3d_states.matching_geometry_column("already_inside") == "zone_half_width_atr"
    with pytest.raises(ValueError, match="Unknown G3D event-state control"):
        g3d_states.matching_geometry_column("unknown")


def test_g3e_revisit_requires_six_outside_candles_before_first_encounter() -> None:
    high = np.full(20, 98.0)
    low = np.full(20, 97.0)
    high[3] = 100.0
    high[10] = 100.0

    revisit = g3e_origin.first_revisit(
        origin_index=0,
        level=100.0,
        half_width=1.0,
        high=high,
        low=low,
    )

    assert revisit == (10, "contact")


def test_g3e_origin_scope_keeps_existing_level_overlaps_explicit() -> None:
    events = pd.DataFrame(
        {
            "overlap_density_zone_count": [0, 1, 0, 2],
            "overlap_reference_level_count": [0, 0, 1, 2],
        }
    )

    assert g3e_origin.classify_origin_scope(events).tolist() == [
        "isolated_origin",
        "origin_plus_density",
        "origin_plus_reference",
        "origin_plus_density_and_reference",
    ]


def test_g3e_reaction_opens_only_preflight_supported_scopes() -> None:
    geometry = pd.DataFrame(
        {
            "origin_scope": ["isolated_origin", "origin_plus_density", "isolated_origin"],
            "event_kind": ["contact", "near_miss", "repeat_contact"],
        }
    )

    filtered = g3e_reaction.filter_supported_scopes(
        geometry,
        ("isolated_origin",),
    )

    assert filtered[["origin_scope", "event_kind"]].to_dict("records") == [
        {"origin_scope": "isolated_origin", "event_kind": "contact"}
    ]


def test_g3e_reaction_matches_distance_from_outer_zone_edge() -> None:
    events = pd.DataFrame(
        {
            "pre_distance_atr": [0.40, 0.15, 0.05],
            "zone_half_width_atr": [0.10, 0.10, 0.10],
        }
    )

    assert g3e_reaction.outside_edge_gap_atr(events).tolist() == pytest.approx([0.30, 0.05, 0.0])
    assert "state_g3e_origin_impulse_sign" in g3e_reaction.MATCH_STATE_FEATURES


def test_g3e_component_filter_requires_the_exact_density_zone_to_overlap_origin() -> None:
    event_time = pd.Timestamp("2026-01-01T00:00:00Z")
    contacts = pd.DataFrame(
        {
            "event_time": [event_time],
            "base_index": [1],
            "origin_scope": ["origin_plus_density_and_reference"],
            "level_price": [100.0],
            "zone_half_width": [0.5],
            "overlap_density_zone_count": [1],
            "overlap_reference_level_count": [1],
        }
    )
    matches = pd.DataFrame(
        {
            "actual_event_time": [event_time, event_time],
            "actual_base_index": [1, 1],
            "density_family": ["test_density", "test_density"],
            "history_hours": [168, 168],
            "actual_zone_rank": [0, 1],
            "actual_scope": ["density_plus_reference_cluster"] * 2,
        }
    )
    surface = SimpleNamespace(
        family="test_density",
        history_hours=168,
        zones=SimpleNamespace(
            levels=np.asarray([[np.nan, np.nan], [100.4, 103.0]]),
            half_widths=np.asarray([[np.nan, np.nan], [0.2, 0.2]]),
        ),
    )

    selected = g3e_components.origin_overlapping_density_matches(
        matches=matches,
        contacts=contacts,
        surfaces=[surface],
    )

    assert len(selected) == 1
    assert int(selected.iloc[0]["actual_zone_rank"]) == 0
    assert selected.iloc[0]["actual_scope"] == "origin_plus_density_and_reference"
    assert g3e_components.evidence_eligible_count(pd.DataFrame()) == 0


def test_g3e_shuffled_origin_sources_exclude_real_source_timestamps() -> None:
    base = pd.DataFrame({"period": ["early"] * 12 + ["late"] * 12})
    origins = pd.DataFrame({"origin_index": [1, 5, 14, 20]})

    shuffled = g3e_artificial.shifted_origin_indexes(base, origins)

    assert len(shuffled) == 4
    assert len(set(shuffled.tolist())) == 4
    assert set(shuffled.tolist()).isdisjoint({1, 5, 14, 20})


def test_g3f_completed_week_becomes_available_only_after_all_candles_close() -> None:
    dates = pd.date_range("2026-01-05", periods=168, freq="1h", tz="UTC")
    source = pd.DataFrame(
        {
            "date": dates,
            "high": np.arange(168, dtype=float) + 101.0,
            "low": np.arange(168, dtype=float) + 99.0,
        }
    )

    complete = g3f_generic.complete_calendar_periods(source, period="week")
    incomplete = g3f_generic.complete_calendar_periods(
        source.drop(index=50).reset_index(drop=True),
        period="week",
    )

    assert len(complete) == 1
    assert complete.iloc[0]["available_at"] == pd.Timestamp("2026-01-12T00:00:00Z")
    assert int(complete.iloc[0]["candle_count"]) == 168
    assert incomplete.empty


def test_g3f_alignment_does_not_backfill_an_unfinished_source_candle() -> None:
    source = pd.DataFrame(
        {
            "source_open": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T04:00:00Z"]),
            "available_at": pd.to_datetime(["2026-01-01T04:00:00Z", "2026-01-01T08:00:00Z"]),
            "level": [100.0, 200.0],
        }
    )
    event_times = pd.Series(
        pd.to_datetime(
            [
                "2026-01-01T03:00:00Z",
                "2026-01-01T04:00:00Z",
                "2026-01-01T07:00:00Z",
                "2026-01-01T08:00:00Z",
            ]
        )
    )

    aligned = g3f_generic.aligned_source_frame(event_times, source)

    assert np.isnan(aligned.iloc[0]["level"])
    assert aligned["level"].iloc[1:].tolist() == [100.0, 100.0, 200.0]


def test_g3f_keeps_single_same_timeframe_and_cross_timeframe_scopes_separate() -> None:
    assert g3f_generic.classify_scope(0, 0) == "isolated_generic_level"
    assert g3f_generic.classify_scope(2, 0) == "same_timeframe_generic_cluster"
    assert g3f_generic.classify_scope(2, 1) == "cross_timeframe_generic_cluster"


@pytest.mark.parametrize(
    ("other_count", "cross_timeframe_count", "cross_family_count", "expected"),
    [
        (0, 0, 0, "isolated_generic_level"),
        (2, 0, 0, "same_timeframe_same_family"),
        (2, 0, 1, "same_timeframe_cross_family"),
        (2, 1, 0, "cross_timeframe_same_family"),
        (2, 1, 1, "cross_timeframe_cross_family"),
    ],
)
def test_g3f_names_indicator_and_timeframe_crossings_explicitly(
    other_count: int,
    cross_timeframe_count: int,
    cross_family_count: int,
    expected: str,
) -> None:
    assert (
        g3f_generic.classify_relationship(
            other_count=other_count,
            cross_timeframe_count=cross_timeframe_count,
            cross_family_count=cross_family_count,
        )
        == expected
    )


def test_g3f_does_not_count_sma20_and_bollinger_middle_as_two_components() -> None:
    sma = SimpleNamespace(
        family="simple_moving_average",
        name="sma_20",
        source_timeframe="4h",
    )
    middle = SimpleNamespace(
        family="bollinger_band",
        name="bollinger_20_middle",
        source_timeframe="4h",
    )

    assert g3f_generic.formula_alias_group(sma) == g3f_generic.formula_alias_group(middle)
    assert (
        g3f_generic.classify_dependency_relationship(
            other_count=2,
            cross_dependency_group_count=0,
        )
        == "same_dependency_group"
    )


def test_g3f_reaction_filter_requires_the_exact_crossing_relationship() -> None:
    geometry = pd.DataFrame(
        {
            "level_family": ["bollinger_band", "bollinger_band"],
            "level_name": ["bollinger_20_upper", "bollinger_20_upper"],
            "source_timeframe": ["4h", "4h"],
            "generic_scope": [
                "cross_timeframe_generic_cluster",
                "cross_timeframe_generic_cluster",
            ],
            "cluster_relationship": [
                "cross_timeframe_same_family",
                "cross_timeframe_cross_family",
            ],
            "cluster_dependency_relationship": [
                "same_dependency_group",
                "cross_dependency_group",
            ],
        }
    )
    supported = (
        (
            "bollinger_band",
            "bollinger_20_upper",
            "4h",
            "cross_timeframe_generic_cluster",
            "cross_timeframe_cross_family",
            "cross_dependency_group",
        ),
    )

    filtered = g3f_reaction.filter_supported_geometry(geometry, supported)

    assert len(filtered) == 1
    assert filtered.iloc[0]["cluster_relationship"] == "cross_timeframe_cross_family"


def test_g3f_reaction_density_context_counts_nearby_generic_surfaces() -> None:
    base = pd.DataFrame({"pre_close": [100.0], "base_atr": [2.0]})
    levels = np.asarray([[101.0, 103.0, 110.0]])
    widths = np.asarray([[0.2, 0.4, 0.2]])
    valid = np.asarray([[True, True, True]])

    count, coverage = g3f_reaction.generic_density_context(
        base=base,
        level_matrix=levels,
        width_matrix=widths,
        valid_matrix=valid,
    )

    assert int(count[0]) == 2
    assert coverage[0] == pytest.approx((0.4 + 0.8) / 16.0)


def test_g3f_reaction_density_context_deduplicates_formula_aliases() -> None:
    levels = [
        SimpleNamespace(name="sma_20", source_timeframe="1h"),
        SimpleNamespace(name="bollinger_20_middle", source_timeframe="1h"),
        SimpleNamespace(name="ema_12", source_timeframe="1h"),
    ]
    valid = np.ones((2, 3), dtype=bool)

    unique = g3f_reaction.formula_unique_valid_matrix(levels, valid)

    assert unique[:, 0].all()
    assert not unique[:, 1].any()
    assert unique[:, 2].all()


def test_g3f_reaction_maps_named_relationships_to_their_scope() -> None:
    assert (
        g3f_reaction.relationship_scope("same_timeframe_cross_family")
        == "same_timeframe_generic_cluster"
    )
    assert (
        g3f_reaction.relationship_scope("cross_timeframe_same_family")
        == "cross_timeframe_generic_cluster"
    )


def test_g3f_artificial_identity_shuffle_uses_a_fixed_family_rotation() -> None:
    assert (
        g3f_artificial.shuffled_donor_name("simple_moving_average", "sma_20")
        == "sma_50"
    )
    assert (
        g3f_artificial.shuffled_donor_name("bollinger_band", "bollinger_20_upper")
        == "bollinger_20_lower"
    )


def test_g3f_artificial_stale_and_shuffled_levels_remain_causal() -> None:
    dates = pd.Series(pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC"))
    base = pd.DataFrame(
        {
            "date": dates,
            "high": np.arange(48, dtype=float) + 102.0,
            "low": np.arange(48, dtype=float) + 98.0,
            "base_atr": np.ones(48, dtype=float),
        }
    )

    def level(name: str, offset: float) -> g3f_generic.GenericLevel:
        return g3f_generic.GenericLevel(
            family="simple_moving_average",
            name=name,
            source_timeframe="1h",
            level=np.arange(48, dtype=float) + offset,
            valid=np.ones(48, dtype=bool),
            source_available=dates.copy(),
            source_open=dates - pd.Timedelta(hours=1),
            source_age_hours=np.zeros(48, dtype=float),
            lookback_bars=np.full(48, int(name.removeprefix("sma_")), dtype=np.int16),
        )

    target = level("sma_20", 100.0)
    donor = level("sma_50", 200.0)
    other = level("sma_200", 300.0)
    definitions = {
        item.control: item
        for item in g3f_artificial.artificial_level_definitions(
            base=base,
            target=target,
            levels=(target, donor, other),
        )
    }

    stale = definitions[g3f_artificial.STALE_CONTROL].item
    shuffled = definitions[g3f_artificial.SHUFFLED_CONTROL]
    assert np.isnan(stale.level[19])
    assert stale.level[20] == pytest.approx(target.level[0])
    assert stale.source_available.iloc[20] == target.source_available.iloc[0]
    assert shuffled.donor_key == donor.key
    assert np.array_equal(shuffled.item.level, donor.level)


def test_g3f_artificial_geometry_requires_the_same_explicit_relationship() -> None:
    dates = pd.Series(pd.date_range("2026-01-01", periods=32, freq="1h", tz="UTC"))
    high = np.full(32, 99.0)
    low = np.full(32, 98.5)
    high[5] = 101.05
    low[5] = 100.95
    base = pd.DataFrame(
        {
            "date": dates,
            "period": ["early"] * 32,
            "high": high,
            "low": low,
            "pre_close": np.full(32, 99.0),
            "base_atr": np.ones(32),
        }
    )

    def level(
        *, family: str, name: str, timeframe: str, price: float
    ) -> g3f_generic.GenericLevel:
        return g3f_generic.GenericLevel(
            family=family,
            name=name,
            source_timeframe=timeframe,
            level=np.full(32, price),
            valid=np.ones(32, dtype=bool),
            source_available=dates.copy(),
            source_open=dates - pd.Timedelta(hours=1),
            source_age_hours=np.zeros(32),
            lookback_bars=np.full(32, 12, dtype=np.int16),
        )

    target = level(
        family="exponential_moving_average",
        name="ema_12",
        timeframe="1h",
        price=100.0,
    )
    peer = level(
        family="exponential_moving_average",
        name="ema_26",
        timeframe="4h",
        price=101.0,
    )
    artificial = g3f_artificial.ArtificialLevel(
        control="shift_+1atr",
        item=level(
            family="exponential_moving_average",
            name="ema_12",
            timeframe="1h",
            price=101.0,
        ),
        excluded_real_keys=(target.key,),
        donor_key=None,
        construction="test",
    )

    geometry, audit = g3f_artificial.artificial_contact_geometry(
        pair="ETH/USDT:USDT",
        base=base,
        levels=(target, peer),
        artificial=artificial,
    )

    assert audit["target_contamination_drops"] == 0
    assert len(geometry) == 1
    assert geometry.iloc[0]["cluster_relationship"] == "cross_timeframe_same_family"
    assert geometry.iloc[0]["cluster_dependency_relationship"] == "same_dependency_group"

    contaminated_target = level(
        family="exponential_moving_average",
        name="ema_12",
        timeframe="1h",
        price=101.0,
    )
    contaminated, contaminated_audit = g3f_artificial.artificial_contact_geometry(
        pair="ETH/USDT:USDT",
        base=base,
        levels=(contaminated_target, peer),
        artificial=artificial,
    )
    assert contaminated.empty
    assert contaminated_audit["target_contamination_drops"] == 1


def test_g3f_attribution_review_distinguishes_a_pass_from_state_imbalance() -> None:
    claim = pd.Series(
        {
            "scope": "normal_top10",
            "route_id": "g3f|1h|bollinger_band|bollinger_20_upper",
            "density_family": "bollinger_band",
            "history_hours": 20,
            "actual_scope": "isolated_generic_level__isolated_generic_level",
            "response_window": "h1",
            "outcome": "volume_ratio_h1",
        }
    )
    cohort = pd.DataFrame(
        [
            {
                **{key: claim[key] for key in g3f_attribution.CLAIM_KEYS},
                "control": "shift_+1atr",
                "period": period,
                "coverage_gate_passed": True,
                "state_balance_usable": True,
                "independent_event_pairs": 100,
                "equal_coin_delta_median": delta,
                "pooled_event_delta_mean": delta,
                "equal_coin_positive_fraction": 0.8,
            }
            for period, delta in (
                ("validation_early", 0.2),
                ("validation_late", 0.1),
            )
        ]
    )
    leave_one_out = pd.DataFrame(
        [
            {
                **{key: claim[key] for key in g3f_attribution.CLAIM_KEYS},
                "control": "shift_+1atr",
                "period": period,
                "coverage_gate_passed": True,
                "equal_coin_delta_median": 0.1,
            }
            for period in ("validation_early", "validation_late")
            for _ in range(10)
        ]
    )
    rule = g3f_attribution.SCOPE_RULES["normal_top10"]

    passed = g3f_attribution.assess_control(
        claim=claim,
        control="shift_+1atr",
        source_stage="artificial",
        source={"cohort": cohort, "leave_one_out": leave_one_out},
        rule=rule,
    )
    imbalanced_cohort = cohort.copy()
    imbalanced_cohort.loc[
        imbalanced_cohort["period"].eq("validation_late"), "state_balance_usable"
    ] = False
    imbalanced = g3f_attribution.assess_control(
        claim=claim,
        control="shift_+1atr",
        source_stage="artificial",
        source={"cohort": imbalanced_cohort, "leave_one_out": leave_one_out},
        rule=rule,
    )

    assert passed["control_status"] == "passed"
    assert passed["effect_sign"] == "higher"
    assert imbalanced["control_status"] == "state_imbalance"


def test_g3f_review_requires_two_period_member_and_leave_one_out_agreement() -> None:
    identity = {
        "route_id": "g3f|1h|bollinger_band|bollinger_20_upper",
        "density_family": "bollinger_band",
        "history_hours": 20,
        "actual_scope": "isolated_generic_level__isolated_generic_level",
        "control": "same_state_same_density_no_level",
        "response_window": "h1",
        "outcome": "volume_ratio_h1",
    }
    cohort = pd.DataFrame(
        [
            {
                **identity,
                "period": period,
                "evidence_eligible": True,
                "pooled_event_delta_mean": delta,
                "equal_coin_delta_median": delta,
                "equal_coin_positive_fraction": fraction,
                "independent_event_pairs": 100,
            }
            for period, delta, fraction in (
                ("early", 0.2, 0.8),
                ("late", 0.1, 0.7),
            )
        ]
    )
    leave_one_out = pd.DataFrame(
        [
            {
                **identity,
                "period": period,
                "omitted_pair": omitted,
                "coverage_gate_passed": True,
                "equal_coin_delta_median": 0.1,
            }
            for period in ("early", "late")
            for omitted in ("BTC/USDT:USDT", "ETH/USDT:USDT")
        ]
    )

    selected = g3f_review.select_repeated_candidates(
        cohort=cohort,
        leave_one_out=leave_one_out,
        scope="normal_top10",
        periods=("early", "late"),
        minimum_positive_fraction=0.7,
        maximum_positive_fraction=0.3,
    )
    reversed_cohort = cohort.copy()
    reversed_cohort.loc[reversed_cohort["period"].eq("late"), "pooled_event_delta_mean"] = -0.1
    rejected = g3f_review.select_repeated_candidates(
        cohort=reversed_cohort,
        leave_one_out=leave_one_out,
        scope="normal_top10",
        periods=("early", "late"),
        minimum_positive_fraction=0.7,
        maximum_positive_fraction=0.3,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["effect_sign"] == "higher"
    assert rejected.empty


def test_g3h_pressure_regime_is_causal_and_masks_low_coverage(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-01", periods=171, freq="1h", tz="UTC")
    pressure = np.arange(171, dtype=float)
    pressure[168] = 1_000.0
    pressure[169] = 2_000.0
    source = pd.DataFrame(
        {
            "date": dates,
            "source_max_ts": dates - pd.Timedelta(minutes=1),
            "obts_feature_present": 1.0,
            "obts_coverage_ratio": 1.0,
            "obts_pressure_25bps_mean": pressure,
        }
    )
    source.loc[169, "obts_coverage_ratio"] = 0.5
    path = tmp_path / "orderbook.parquet"
    source.to_parquet(path, index=False)

    result = g3h_context.load_causal_pressure_surface(path)

    assert result.loc[167, "pressure_regime"] == "unavailable"
    assert result.loc[168, "pressure_regime"] == "active"
    assert result.loc[168, "pressure_upper_tertile"] < 1_000.0
    assert result.loc[169, "pressure_regime"] == "unavailable"
    assert np.isnan(result.loc[169, "absolute_pressure"])


def test_g3h_active_quiet_orientation_uses_stale_context_as_control() -> None:
    source = pd.DataFrame(
        [
            {
                "outcome": "contact_volume_ratio",
                "raw_delta": 2.0,
                "high_current_regime": "active",
                "low_current_regime": "quiet",
                "high_current_absolute_pressure": 0.9,
                "low_current_absolute_pressure": 0.1,
                "high_stale_absolute_pressure": 0.2,
                "low_stale_absolute_pressure": 0.8,
                "high_stale_usable": True,
                "low_stale_usable": True,
            }
        ]
    )

    result = g3h_context.active_quiet_candidates(source)

    assert result.iloc[0]["current_oriented_delta"] == pytest.approx(2.0)
    assert result.iloc[0]["current_expected_oriented_delta"] == pytest.approx(2.0)
    assert bool(result.iloc[0]["current_expected_sign"])
    assert result.iloc[0]["stale_oriented_delta"] == pytest.approx(-2.0)
    assert not bool(result.iloc[0]["stale_expected_sign"])


def test_g3h_global_independence_rejects_cross_coin_overlap_and_event_reuse() -> None:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    candidates = pd.DataFrame(
        [
            {
                "pair": "ETH/USDT:USDT",
                "period": "validation_early",
                "high_base_index": 1,
                "low_base_index": 2,
                "high_event_time": start,
                "low_event_time": start + pd.Timedelta(hours=100),
                "independence_hours": 6,
                "match_distance": 0.1,
                "independent_selection_order": 1,
            },
            {
                "pair": "SOL/USDT:USDT",
                "period": "validation_early",
                "high_base_index": 3,
                "low_base_index": 4,
                "high_event_time": start + pd.Timedelta(hours=1),
                "low_event_time": start + pd.Timedelta(hours=200),
                "independence_hours": 6,
                "match_distance": 0.2,
                "independent_selection_order": 2,
            },
            {
                "pair": "ETH/USDT:USDT",
                "period": "validation_early",
                "high_base_index": 1,
                "low_base_index": 5,
                "high_event_time": start + pd.Timedelta(hours=300),
                "low_event_time": start + pd.Timedelta(hours=400),
                "independence_hours": 6,
                "match_distance": 0.3,
                "independent_selection_order": 3,
            },
            {
                "pair": "ADA/USDT:USDT",
                "period": "validation_early",
                "high_base_index": 6,
                "low_base_index": 7,
                "high_event_time": start + pd.Timedelta(hours=300),
                "low_event_time": start + pd.Timedelta(hours=400),
                "independence_hours": 6,
                "match_distance": 0.4,
                "independent_selection_order": 4,
            },
        ]
    )

    result = g3h_context.select_globally_independent(candidates)

    assert list(result["pair"]) == ["ETH/USDT:USDT", "ADA/USDT:USDT"]


def test_g4a_global_independence_applies_across_pairs() -> None:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    source = pd.DataFrame(
        [
            {
                "cohort": "normal",
                "pair": "ETH/USDT:USDT",
                "event_time": start,
                "g4a_selection_hash": "a",
            },
            {
                "cohort": "normal",
                "pair": "SOL/USDT:USDT",
                "event_time": start + pd.Timedelta(hours=1),
                "g4a_selection_hash": "b",
            },
            {
                "cohort": "normal",
                "pair": "ADA/USDT:USDT",
                "event_time": start + pd.Timedelta(hours=49),
                "g4a_selection_hash": "c",
            },
        ]
    )

    result = g4a_breadth.select_globally_independent_pool(source)

    assert list(result["pair"]) == ["ETH/USDT:USDT", "ADA/USDT:USDT"]


def test_g4a_cell_quotas_redistribute_a_sparse_stratum() -> None:
    cells = [
        ("development", "lvn_above"),
        ("development", "lvn_below"),
        ("validation_early", "lvn_above"),
        ("validation_early", "lvn_below"),
        ("validation_late", "lvn_above"),
        ("validation_late", "lvn_below"),
    ]
    rows = []
    for cell_index, (period, level_name) in enumerate(cells):
        count = 4 if cell_index == 2 else 20
        rows.extend(
            {"period": period, "level_name": level_name}
            for _ in range(count)
        )
    source = pd.DataFrame(rows)

    result = g4a_breadth.balanced_cell_quotas(source, cells=cells, target=60)

    assert sum(result.values()) == 60
    assert result[("validation_early", "lvn_above")] == 4
    assert all(value >= 10 for cell, value in result.items() if cell != cells[2])


def test_g4a_download_command_is_fixed_to_frozen_binance_futures_surface() -> None:
    result = g4a_download.download_command_args(
        {
            "pair": "DOGE/USDT:USDT",
            "timerange": "1760439600-1761480000",
        }
    )

    assert result[0] == str(g4a_download.CONTROLLER_PYTHON.resolve())
    assert result[result.index("--exchange") + 1] == "binance"
    assert result[result.index("--trading-mode") + 1] == "futures"
    assert result[result.index("-p") + 1] == "DOGE/USDT:USDT"
    assert result[result.index("-t") + 1] == "1m"
    assert result[result.index("--timerange") + 1] == "1760439600-1761480000"


def test_g4a_download_rejects_command_outside_frozen_acquisition() -> None:
    frozen_command = "frozen command"
    acquisition = pd.DataFrame(
        [
            {
                "pair": "DOGE/USDT:USDT",
                "timerange": "1760439600-1761480000",
                "download_command": frozen_command,
            }
        ]
    )
    tampered = pd.DataFrame(
        [
            {
                "pair": "DOGE/USDT:USDT",
                "download_command": "different command",
            }
        ]
    )

    with pytest.raises(ValueError, match="outside the frozen acquisition set"):
        g4a_download.validated_failed_download_rows(tampered, acquisition)


def test_g4a_isolated_repair_requires_complete_contiguous_interval() -> None:
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    complete = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=3, freq="1min"),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 20.0, 30.0],
        }
    )

    accepted = g4a_repair.interval_quality(
        complete, start, start + pd.Timedelta(minutes=3), 3
    )
    rejected = g4a_repair.interval_quality(
        complete.drop(index=1), start, start + pd.Timedelta(minutes=3), 3
    )

    assert accepted["complete"] is True
    assert rejected["complete"] is False
    assert rejected["missing_rows"] == 1
    assert rejected["non_one_minute_gaps"] == 1


def test_g4a_isolated_repair_detects_overlap_value_conflict() -> None:
    existing = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01T00:00:00Z"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        }
    )
    incoming = existing.copy()
    incoming.loc[0, "close"] = 1.6

    assert g4a_repair.count_overlap_conflicts(existing, existing) == 0
    assert g4a_repair.count_overlap_conflicts(existing, incoming) == 1


def test_g4a_confirmation_complete_path_mask_rejects_edges() -> None:
    dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC"))

    result = g4a_confirmation.complete_path_mask(
        dates, before_minutes=2, after_minutes=2
    )

    assert result.tolist() == [False, False, True, True, True, True, True, True, False, False]


def test_g4a_confirmation_vectorized_level_contact_is_causal() -> None:
    dates = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    base = pd.DataFrame({"date": dates, "base_atr": [2.0, 2.0, 2.0]})
    candles = pd.DataFrame(
        {
            "date": dates + pd.Timedelta(minutes=30),
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
        }
    )
    level = SimpleNamespace(
        valid=np.array([True, True, True]),
        level=np.array([100.0, 100.0, 100.0]),
        source_available=pd.Series(
            [dates[0], dates[1] + pd.Timedelta(hours=1), dates[2]]
        ),
    )

    result = g4a_confirmation.vectorized_level_contact_count(
        candles, base=base, aligned_levels=[level]
    )

    assert result.tolist() == [1, 0, 1]


def test_g4a_confirmation_global_time_separation_includes_boundary() -> None:
    reference = pd.Timestamp("2025-01-01", tz="UTC")

    assert not g4a_confirmation.separated_from_times(
        reference + pd.Timedelta(minutes=239), [reference]
    )
    assert g4a_confirmation.separated_from_times(
        reference + pd.Timedelta(minutes=240), [reference]
    )


def test_g4a_expansion_splits_only_noncontiguous_missing_minutes() -> None:
    missing = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:01:00Z",
                "2025-01-01T00:04:00Z",
            ]
        )
    )

    result = g4a_expansion.contiguous_minute_runs(missing)

    assert result == [
        (missing[0], missing[1]),
        (missing[2], missing[2]),
    ]


def test_g4a_unconditioned_geometric_components_use_transitive_overlap() -> None:
    peers = [
        {"lower": 1.0, "upper": 2.0, "aligned_index": 1},
        {"lower": 1.5, "upper": 3.0, "aligned_index": 2},
        {"lower": 2.8, "upper": 4.0, "aligned_index": 3},
        {"lower": 5.0, "upper": 6.0, "aligned_index": 4},
    ]

    result = g4a_unconditioned.connected_geometric_components(peers)

    assert [[row["aligned_index"] for row in group] for group in result] == [
        [1, 2, 3],
        [4],
    ]


def test_g4a_unconditioned_request_forbids_postcontact_selection() -> None:
    result = g4a_unconditioned.request_contract("test", [])

    assert result["selection"]["contact_hour_volume_used"] is False
    assert result["selection"]["cluster_contact_required_after_event"] is False
    assert result["selection"]["future_price_path_used"] is False
    assert result["selection"]["profit_used"] is False


def test_g4a_review_separates_conditional_direction_from_joint_success() -> None:
    actual = pd.DataFrame(
        {
            "pair": ["A", "B", "C"],
            "reaction_60m": [True, True, False],
            "direction_callable": [True, True, True],
            "first_direction": ["up", "down", "down"],
            "simple_trend_direction_correct": [True, False, True],
            "simple_trend_joint_reaction_and_direction": [True, False, False],
            "first_path_through": [True, False, True],
            "first_path_away": [False, True, False],
            "first_zone_breakout": [True, False, True],
            "first_zone_rejection": [False, True, False],
        }
    )

    result = g4a_review.summary_metrics(actual, pd.DataFrame())

    assert result["conditional_direction_accuracy_given_reaction"] == 0.5
    assert result["joint_reaction_direction_rate"] == pytest.approx(1.0 / 3.0)
    assert result["direction_call_coverage"] == 1.0


def test_g4a_review_counts_paired_reaction_discordance() -> None:
    paired = pd.DataFrame(
        {
            "reaction_60m": [True, True, True, False],
            "control_reaction_60m": [False, True, False, True],
            "control_control_state_distance_mean": [0.1, 0.2, 0.3, 0.4],
            "control_control_state_distance_max": [0.5, 0.6, 0.7, 0.8],
        }
    )
    for column in g4a_review.PAIRED_DIFFERENCE_COLUMNS:
        paired[column] = [2.0, 2.0, 2.0, 2.0]
        paired[f"control_{column}"] = [1.0, 1.0, 1.0, 1.0]

    result = g4a_review.paired_metrics(paired)

    assert result["actual_only_reaction_count"] == 2
    assert result["control_only_reaction_count"] == 1
    assert result["paired_reaction_uplift"] == 0.25
    assert result["excursion_strength_60_half_widths_difference_median"] == 1.0
