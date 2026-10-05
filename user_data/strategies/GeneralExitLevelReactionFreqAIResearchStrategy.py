from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from user_data.strategies.GeneralExitTargetQualityFreqAIResearchStrategy import (
    GeneralExitTargetQualityFreqAIResearchStrategy,
    LEVEL_FAMILIES,
    MISSING_DISTANCE,
    _atr,
    _future_extreme,
    _numeric,
)


USER_DATA_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "general_exit_level_reaction_cache"
)
CACHE_FILES = {
    "BTC/USDT:USDT": CACHE_DIR / "btc_general_exit_levels_5tf.parquet",
    "ETH/USDT:USDT": CACHE_DIR / "eth_general_exit_levels_5tf.parquet",
    "SOL/USDT:USDT": CACHE_DIR / "sol_general_exit_levels_5tf.parquet",
}
TIMEFRAMES = ("1h", "4h", "8h", "1d", "3d")
TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "8h": 8, "1d": 24, "3d": 72}
CLUSTER_BANDS = ((0.0025, "25bp"), (0.005, "50bp"), (0.01, "100bp"))
TOUCH_BAND = 0.001


def _future_mean(series: Series, horizon: int) -> Series:
    future = pd.to_numeric(series, errors="coerce").shift(-1).iloc[::-1]
    return future.rolling(horizon, min_periods=horizon).mean().iloc[::-1]


def _future_volume_pressure(dataframe: DataFrame, horizon: int) -> Series:
    high = pd.to_numeric(dataframe["high"], errors="coerce")
    low = pd.to_numeric(dataframe["low"], errors="coerce")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    volume = pd.to_numeric(dataframe["volume"], errors="coerce").clip(lower=0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    pressure = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
    signed_volume = pressure.fillna(0.0) * volume.fillna(0.0)
    future_signed = signed_volume.shift(-1).iloc[::-1].rolling(
        horizon, min_periods=horizon
    ).sum().iloc[::-1]
    future_volume = volume.shift(-1).iloc[::-1].rolling(
        horizon, min_periods=horizon
    ).sum().iloc[::-1]
    return future_signed / future_volume.replace(0.0, np.nan)


def build_level_reaction_targets(dataframe: DataFrame) -> DataFrame:
    """Build continuous 1-4 candle price, volume, and pressure reaction labels."""
    out = dataframe.copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0)
    atr_ratio = (_atr(out) / close).replace(0.0, np.nan)
    prior_volume = volume.rolling(3, min_periods=3).mean().replace(0.0, np.nan)

    for horizon in (1, 2, 3, 4):
        future_high = _future_extreme(high, horizon, "max")
        future_low = _future_extreme(low, horizon, "min")
        upside = (future_high / close) - 1.0
        downside = 1.0 - (future_low / close)
        out[f"&-down_vs_up_advantage_{horizon}h"] = (
            (downside - upside) / atr_ratio
        ).clip(-10.0, 10.0)
        out[f"&-future_volume_ratio_{horizon}h"] = (
            _future_mean(volume, horizon) / prior_volume
        ).clip(0.0, 20.0)
        out[f"&-future_volume_pressure_{horizon}h"] = _future_volume_pressure(
            out, horizon
        )
    return out


class GeneralExitLevelReactionFreqAIResearchStrategy(
    GeneralExitTargetQualityFreqAIResearchStrategy
):
    """Explicit level-identity and level-cluster reaction research."""

    include_single_levels = True
    include_same_asset_clusters = False
    include_hit_response = False
    include_hit_identities = True
    include_btc_levels_and_clusters = False
    _reaction_cache_frames: dict[str, DataFrame] = {}

    @classmethod
    def _canonical_pair(cls, pair: str) -> str:
        normalized = pair.strip().upper()
        if normalized in CACHE_FILES:
            return normalized
        base = normalized.split("/")[0].split(":")[0]
        for candidate in CACHE_FILES:
            if candidate.startswith(f"{base}/"):
                return candidate
        raise ValueError(f"Unsupported level-reaction research pair: {pair!r}")

    @classmethod
    def _load_cache(cls, pair: str) -> DataFrame:
        canonical = cls._canonical_pair(pair)
        cached = cls._reaction_cache_frames.get(canonical)
        if cached is not None:
            return cached
        path = CACHE_FILES[canonical]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        frame = frame.sort_values("date").set_index("date")
        cls._reaction_cache_frames[canonical] = frame
        return frame

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = self._canonical_pair(str(metadata.get("pair") or ""))
        aligned = self._aligned_cache(dataframe, pair)
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)
        states = self._states(aligned, close)

        if self.include_single_levels:
            self._append_single_level_features(dataframe, aligned, close, states)
        if self.include_same_asset_clusters:
            self._append_features(dataframe, self._cluster_features(states))
        if self.include_hit_response:
            self._append_hit_response_features(
                dataframe,
                aligned,
                states,
                prefix="alt",
                include_hit_identities=self.include_hit_identities,
            )

        if self.include_btc_levels_and_clusters:
            btc = self._aligned_cache(dataframe, "BTC/USDT:USDT")
            btc_close = _numeric(btc, "close").replace(0.0, np.nan)
            btc_states = self._states(btc, btc_close)
            self._append_single_level_features(
                dataframe, btc, btc_close, btc_states, prefix="btc"
            )
            self._append_hit_response_features(
                dataframe,
                btc,
                btc_states,
                prefix="btc",
                include_hit_identities=True,
            )
            self._append_cross_asset_features(dataframe, states, btc_states)
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = metadata, kwargs
        return build_level_reaction_targets(dataframe)

    @classmethod
    def _states(cls, aligned: DataFrame, close: Series) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for timeframe in TIMEFRAMES:
            _, state = cls._level_features(aligned, close, timeframe)
            states[timeframe] = state
        return states

    @classmethod
    def _append_single_level_features(
        cls,
        dataframe: DataFrame,
        aligned: DataFrame,
        close: Series,
        states: dict[str, dict[str, Any]],
        *,
        prefix: str = "alt",
    ) -> None:
        for timeframe in TIMEFRAMES:
            state = states[timeframe]
            for side in ("resistance", "support"):
                for family in LEVEL_FAMILIES:
                    dataframe[
                        f"%-{prefix}_{timeframe}_{side}_{family}_distance"
                    ] = state[side][family].fillna(MISSING_DISTANCE)
            source_prefix = f"st_{timeframe}"
            for name, column in (
                ("hvn_above_strength", f"{source_prefix}_vp_hvn_above_strength"),
                ("hvn_below_strength", f"{source_prefix}_vp_hvn_below_strength"),
                ("value_area_width_pct", f"{source_prefix}_vp_value_area_width_pct"),
                ("value_area_position", f"{source_prefix}_vp_value_area_position"),
                ("tlv2_resistance_score", f"{source_prefix}_tlv2_resistance_score_rank0"),
                ("tlv2_support_score", f"{source_prefix}_tlv2_support_score_rank0"),
                ("tlv2_resistance_pivots", f"{source_prefix}_tlv2_resistance_pivot_count_rank0"),
                ("tlv2_support_pivots", f"{source_prefix}_tlv2_support_pivot_count_rank0"),
            ):
                dataframe[f"%-{prefix}_{timeframe}_{name}"] = _numeric(
                    aligned, column, 0.0
                ).fillna(0.0)
            poc = _numeric(aligned, f"{source_prefix}_vp_prior_poc")
            dataframe[f"%-{prefix}_{timeframe}_prior_poc_age"] = cls._stability_age(
                poc, TIMEFRAME_HOURS[timeframe]
            )
        for side in ("resistance", "support"):
            levels = cls._flatten_levels(states, side)
            dataframe[f"%-{prefix}_{side}_nearest_named_level_distance"] = levels.min(
                axis=1, skipna=True
            ).fillna(MISSING_DISTANCE)

    @staticmethod
    def _append_features(dataframe: DataFrame, features: dict[str, Series]) -> None:
        for name, values in features.items():
            dataframe[f"%-{name}"] = values

    @staticmethod
    def _flatten_levels(
        states: dict[str, dict[str, Any]], side: str
    ) -> DataFrame:
        return pd.concat(
            {
                f"{timeframe}__{family}": states[timeframe][side][family]
                for timeframe in TIMEFRAMES
                for family in LEVEL_FAMILIES
            },
            axis=1,
        )

    @classmethod
    def _cluster_features(
        cls,
        states: dict[str, dict[str, Any]],
        *,
        prefix: str = "alt",
        member_flags: bool = True,
    ) -> dict[str, Series]:
        features: dict[str, Series] = {}
        for side in ("resistance", "support"):
            all_levels = cls._flatten_levels(states, side)
            cls._append_group_cluster_features(
                features,
                all_levels,
                name=f"{prefix}_{side}_all_named_levels",
                member_flags=member_flags,
            )
            for timeframe in TIMEFRAMES:
                group = all_levels.loc[
                    :, [column for column in all_levels if column.startswith(f"{timeframe}__")]
                ]
                cls._append_group_cluster_features(
                    features,
                    group,
                    name=f"{prefix}_{side}_same_timeframe_{timeframe}_cross_type",
                    member_flags=False,
                )
            for family in LEVEL_FAMILIES:
                group = all_levels.loc[
                    :, [column for column in all_levels if column.endswith(f"__{family}")]
                ]
                cls._append_group_cluster_features(
                    features,
                    group,
                    name=f"{prefix}_{side}_same_type_{family}_cross_timeframe",
                    member_flags=False,
                )
        return features

    @staticmethod
    def _append_group_cluster_features(
        features: dict[str, Series],
        levels: DataFrame,
        *,
        name: str,
        member_flags: bool,
    ) -> None:
        anchor = levels.min(axis=1, skipna=True)
        features[f"{name}_nearest_distance"] = anchor.fillna(MISSING_DISTANCE)
        for band, label in CLUSTER_BANDS:
            within = levels.sub(anchor, axis=0).abs().le(band) & levels.notna()
            count = within.sum(axis=1).astype(float)
            features[f"{name}_{label}_member_count"] = count
            features[f"{name}_{label}_is_cluster"] = count.ge(2.0).astype(float)
            present = levels.where(within)
            features[f"{name}_{label}_span"] = (
                present.max(axis=1) - present.min(axis=1)
            ).fillna(0.0)
            if label == "50bp" and member_flags:
                for column in levels:
                    features[f"{name}_{label}_contains_{column}"] = within[column].astype(
                        float
                    )

    @classmethod
    def _compact_cluster_features(
        cls, states: dict[str, dict[str, Any]], *, prefix: str
    ) -> dict[str, Series]:
        features: dict[str, Series] = {}
        for side in ("resistance", "support"):
            levels = cls._flatten_levels(states, side)
            anchor = levels.min(axis=1, skipna=True)
            features[f"{prefix}_{side}_nearest_distance"] = anchor.fillna(
                MISSING_DISTANCE
            )
            for band, label in CLUSTER_BANDS:
                features[f"{prefix}_{side}_{label}_member_count"] = levels.sub(
                    anchor, axis=0
                ).abs().le(band).sum(axis=1).astype(float)
            for timeframe in TIMEFRAMES:
                group = levels.loc[
                    :, [column for column in levels if column.startswith(f"{timeframe}__")]
                ]
                group_anchor = group.min(axis=1, skipna=True)
                features[
                    f"{prefix}_{side}_same_timeframe_{timeframe}_50bp_member_count"
                ] = group.sub(group_anchor, axis=0).abs().le(0.005).sum(axis=1).astype(
                    float
                )
            for family in LEVEL_FAMILIES:
                group = levels.loc[
                    :, [column for column in levels if column.endswith(f"__{family}")]
                ]
                group_anchor = group.min(axis=1, skipna=True)
                features[
                    f"{prefix}_{side}_same_type_{family}_50bp_member_count"
                ] = group.sub(group_anchor, axis=0).abs().le(0.005).sum(axis=1).astype(
                    float
                )
        return features

    @classmethod
    def _raw_level_prices(
        cls, aligned: DataFrame, timeframe: str, side: str
    ) -> DataFrame:
        prefix = f"st_{timeframe}"
        if side == "resistance":
            columns = {
                "va_edge": f"{prefix}_vp_prior_vah",
                "prior_poc": f"{prefix}_vp_prior_poc",
                "hvn": f"{prefix}_vp_hvn_above",
                "tlv2": f"{prefix}_tlv2_resistance_line_rank0",
            }
        else:
            columns = {
                "va_edge": f"{prefix}_vp_prior_val",
                "prior_poc": f"{prefix}_vp_prior_poc",
                "hvn": f"{prefix}_vp_hvn_below",
                "tlv2": f"{prefix}_tlv2_support_line_rank0",
            }
        return DataFrame(
            {family: _numeric(aligned, column) for family, column in columns.items()},
            index=aligned.index,
        )

    @classmethod
    def _append_hit_response_features(
        cls,
        dataframe: DataFrame,
        aligned: DataFrame,
        current_states: dict[str, dict[str, Any]],
        *,
        prefix: str,
        include_hit_identities: bool,
    ) -> None:
        prior_aligned = aligned.shift(1)
        prior_close = _numeric(prior_aligned, "close").replace(0.0, np.nan)
        prior_states = cls._states(prior_aligned, prior_close)
        cls._append_features(
            dataframe,
            {
                f"prehit_{name}": values
                for name, values in cls._compact_cluster_features(
                    prior_states, prefix=prefix
                ).items()
            },
        )

        high = _numeric(aligned, "high") if prefix == "btc" else pd.to_numeric(
            dataframe["high"], errors="coerce"
        )
        low = _numeric(aligned, "low") if prefix == "btc" else pd.to_numeric(
            dataframe["low"], errors="coerce"
        )
        close = _numeric(aligned, "close") if prefix == "btc" else pd.to_numeric(
            dataframe["close"], errors="coerce"
        )
        volume = _numeric(aligned, "volume", 0.0) if prefix == "btc" else pd.to_numeric(
            dataframe["volume"], errors="coerce"
        ).clip(lower=0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        dataframe[f"%-{prefix}_hit_candle_volume_vs_prior3"] = (
            volume / volume.shift(1).rolling(3, min_periods=3).mean().replace(0.0, np.nan)
        ).clip(0.0, 20.0)
        dataframe[f"%-{prefix}_hit_candle_close_location_pressure"] = pressure

        if not include_hit_identities:
            return

        for side in ("resistance", "support"):
            touches: dict[str, Series] = {}
            for timeframe in TIMEFRAMES:
                levels = cls._raw_level_prices(prior_aligned, timeframe, side)
                valid = levels.gt(prior_close, axis=0) if side == "resistance" else levels.lt(
                    prior_close, axis=0
                )
                for family in LEVEL_FAMILIES:
                    level = levels[family].where(valid[family])
                    touched = high.ge(level * (1.0 - TOUCH_BAND)) & low.le(
                        level * (1.0 + TOUCH_BAND)
                    )
                    key = f"{timeframe}__{family}"
                    touches[key] = touched.fillna(False)
                    dataframe[f"%-{prefix}_{side}_hit_{key}"] = touches[key].astype(float)
            touch_frame = DataFrame(touches, index=dataframe.index)
            dataframe[f"%-{prefix}_{side}_hit_named_level_count"] = touch_frame.sum(
                axis=1
            ).astype(float)
            dataframe[f"%-{prefix}_{side}_hit_distinct_type_count"] = pd.concat(
                [
                    touch_frame.loc[
                        :, [column for column in touch_frame if column.endswith(f"__{family}")]
                    ].any(axis=1)
                    for family in LEVEL_FAMILIES
                ],
                axis=1,
            ).sum(axis=1).astype(float)
            dataframe[f"%-{prefix}_{side}_hit_distinct_timeframe_count"] = pd.concat(
                [
                    touch_frame.loc[
                        :, [column for column in touch_frame if column.startswith(f"{timeframe}__")]
                    ].any(axis=1)
                    for timeframe in TIMEFRAMES
                ],
                axis=1,
            ).sum(axis=1).astype(float)
            away_pressure = -pressure if side == "resistance" else pressure
            dataframe[f"%-{prefix}_{side}_hit_candle_away_pressure"] = away_pressure.where(
                touch_frame.any(axis=1), 0.0
            )

    @classmethod
    def _append_cross_asset_features(
        cls,
        dataframe: DataFrame,
        alt_states: dict[str, dict[str, Any]],
        btc_states: dict[str, dict[str, Any]],
    ) -> None:
        for side in ("resistance", "support"):
            alt = cls._flatten_levels(alt_states, side)
            btc = cls._flatten_levels(btc_states, side)
            for band, label in CLUSTER_BANDS:
                alt_anchor = alt.min(axis=1, skipna=True)
                btc_anchor = btc.min(axis=1, skipna=True)
                alt_count = alt.sub(alt_anchor, axis=0).abs().le(band).sum(axis=1)
                btc_count = btc.sub(btc_anchor, axis=0).abs().le(band).sum(axis=1)
                dataframe[
                    f"%-cross_asset_{side}_{label}_alt_and_btc_both_clustered"
                ] = (alt_count.ge(2) & btc_count.ge(2)).astype(float)
                dataframe[
                    f"%-cross_asset_{side}_{label}_combined_member_count"
                ] = (alt_count + btc_count).astype(float)


class GeneralExitReactionPriceControlFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    include_single_levels = False


class GeneralExitReactionExplicitSingleLevelsFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    pass


class GeneralExitReactionSameAssetClustersFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    include_same_asset_clusters = True


class GeneralExitReactionSameAssetCandleStateControlFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    include_same_asset_clusters = True
    include_hit_response = True
    include_hit_identities = False


class GeneralExitReactionSameAssetClustersHitResponseFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    include_same_asset_clusters = True
    include_hit_response = True


class GeneralExitReactionSameAssetBtcClustersHitResponseFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    include_same_asset_clusters = True
    include_hit_response = True
    include_btc_levels_and_clusters = True
