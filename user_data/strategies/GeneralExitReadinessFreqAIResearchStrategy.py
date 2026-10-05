from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from user_data.strategies.GeneralExitLevelReactionFreqAIResearchStrategy import (
    LEVEL_FAMILIES,
    TIMEFRAMES,
    TOUCH_BAND,
    GeneralExitLevelReactionFreqAIResearchStrategy,
    build_level_reaction_targets,
)
from user_data.strategies.GeneralExitTargetQualityFreqAIResearchStrategy import (
    MISSING_DISTANCE,
    TIMEFRAME_HOURS,
    _atr,
    _future_extreme,
    _numeric,
)


def build_exit_readiness_targets(dataframe: DataFrame) -> DataFrame:
    """Add a side-neutral magnitude label to the existing reaction labels."""
    out = build_level_reaction_targets(dataframe)
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    atr_ratio = (_atr(out) / close).replace(0.0, np.nan)

    for horizon in (1, 2, 3, 4):
        future_high = _future_extreme(high, horizon, "max")
        future_low = _future_extreme(low, horizon, "min")
        upside = (future_high / close) - 1.0
        downside = 1.0 - (future_low / close)
        out[f"&-future_reaction_magnitude_{horizon}h"] = (
            pd.concat((upside, downside), axis=1).max(axis=1) / atr_ratio
        ).clip(0.0, 20.0)
    return out


class GeneralExitReadinessFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    """Learn level importance first, then test incremental exit-readiness evidence."""

    include_single_levels = False
    include_same_asset_clusters = False
    include_hit_response = False
    include_hit_identities = True
    include_btc_levels_and_clusters = False
    include_candidate_components = False
    include_weakening_components = False

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)
        if not (
            self.include_candidate_components or self.include_weakening_components
        ):
            return dataframe

        pair = self._canonical_pair(str(metadata.get("pair") or ""))
        aligned = self._aligned_cache(dataframe, pair)
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(
            0.0, np.nan
        )
        states = self._states(aligned, close)
        if self.include_candidate_components:
            self._append_candidate_components(
                dataframe,
                aligned,
                dataframe,
                close,
                states,
                prefix="alt",
            )
        if self.include_weakening_components:
            self._append_weakening_components(
                dataframe,
                aligned,
                dataframe,
                prefix="alt",
            )

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = metadata, kwargs
        return build_exit_readiness_targets(dataframe)

    @classmethod
    def _append_candidate_components(
        cls,
        dataframe: DataFrame,
        aligned: DataFrame,
        market_frame: DataFrame,
        close: Series,
        states: dict[str, dict[str, Any]],
        *,
        prefix: str,
    ) -> None:
        atr_pct = (_atr(market_frame) / close).replace(0.0, np.nan)
        for side in ("resistance", "support"):
            levels = cls._flatten_levels(states, side)
            for timeframe in TIMEFRAMES:
                for family in LEVEL_FAMILIES:
                    key = f"{timeframe}__{family}"
                    distance = levels[key]
                    distance_atr = (distance / atr_pct).replace(
                        [np.inf, -np.inf], np.nan
                    )
                    closeness = (1.0 / (1.0 + distance_atr.clip(lower=0.0))).where(
                        distance.notna(), 0.0
                    )
                    base = f"%-{prefix}_{side}_{timeframe}_{family}"
                    dataframe[f"{base}_distance_atr"] = distance_atr.fillna(
                        MISSING_DISTANCE
                    )
                    dataframe[f"{base}_closeness"] = closeness

                    around_candidate = levels.sub(distance, axis=0).abs()
                    for band, label in ((0.005, "50bp"),):
                        members = (
                            around_candidate.le(band)
                            & levels.notna()
                            & distance.notna().to_numpy()[:, None]
                        )
                        dataframe[f"{base}_{label}_all_member_count"] = members.sum(
                            axis=1
                        ).astype(float)
                        dataframe[f"{base}_{label}_same_type_mtf_count"] = members.loc[
                            :,
                            [column for column in levels if column.endswith(f"__{family}")],
                        ].sum(axis=1).astype(float)
                        dataframe[
                            f"{base}_{label}_same_timeframe_cross_type_count"
                        ] = members.loc[
                            :,
                            [
                                column
                                for column in levels
                                if column.startswith(f"{timeframe}__")
                            ],
                        ].sum(axis=1).astype(float)

                    source = f"st_{timeframe}"
                    if family == "hvn":
                        quality = _numeric(
                            aligned,
                            f"{source}_vp_hvn_{'above' if side == 'resistance' else 'below'}_strength",
                            0.0,
                        ).fillna(0.0)
                        dataframe[f"{base}_quality"] = quality
                        dataframe[f"{base}_quality_at_proximity"] = quality * closeness
                    elif family == "tlv2":
                        direction = "resistance" if side == "resistance" else "support"
                        quality = _numeric(
                            aligned, f"{source}_tlv2_{direction}_score_rank0", 0.0
                        ).fillna(0.0)
                        pivots = _numeric(
                            aligned,
                            f"{source}_tlv2_{direction}_pivot_count_rank0",
                            0.0,
                        ).fillna(0.0)
                        dataframe[f"{base}_quality"] = quality
                        dataframe[f"{base}_pivot_count"] = pivots
                        dataframe[f"{base}_quality_at_proximity"] = quality * closeness
                    elif family == "prior_poc":
                        poc = _numeric(aligned, f"{source}_vp_prior_poc")
                        dataframe[f"{base}_age"] = cls._stability_age(
                            poc, TIMEFRAME_HOURS[timeframe]
                        )

    @classmethod
    def _append_weakening_components(
        cls,
        dataframe: DataFrame,
        aligned: DataFrame,
        market_frame: DataFrame,
        *,
        prefix: str,
    ) -> None:
        open_ = _numeric(market_frame, "open")
        high = _numeric(market_frame, "high")
        low = _numeric(market_frame, "low")
        close = _numeric(market_frame, "close")
        volume = _numeric(market_frame, "volume", 0.0).clip(lower=0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        return_1h = close.pct_change()
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_pressure = (((close - low) / candle_range) * 2.0 - 1.0).clip(
            -1.0, 1.0
        )
        prior_return = return_1h.shift(1).rolling(3, min_periods=3).mean()
        prior_pressure = close_pressure.shift(1).rolling(3, min_periods=3).mean()
        volume_ratio = (
            volume
            / volume.shift(1).rolling(3, min_periods=3).mean().replace(0.0, np.nan)
        ).clip(0.0, 20.0)

        prior_aligned = aligned.shift(1)
        prior_close = _numeric(prior_aligned, "close").replace(0.0, np.nan)
        for side, sign in (("resistance", 1.0), ("support", -1.0)):
            raw_levels = pd.concat(
                {
                    f"{timeframe}__{family}": cls._raw_level_prices(
                        prior_aligned, timeframe, side
                    )[family]
                    for timeframe in TIMEFRAMES
                    for family in LEVEL_FAMILIES
                },
                axis=1,
            )
            valid = (
                raw_levels.gt(prior_close, axis=0)
                if side == "resistance"
                else raw_levels.lt(prior_close, axis=0)
            )
            touched = (
                high.to_numpy()[:, None]
                >= raw_levels.to_numpy() * (1.0 - TOUCH_BAND)
            ) & (
                low.to_numpy()[:, None]
                <= raw_levels.to_numpy() * (1.0 + TOUCH_BAND)
            )
            touched &= valid.fillna(False).to_numpy()
            touched_frame = DataFrame(
                touched, index=dataframe.index, columns=raw_levels.columns
            )
            touched_levels = raw_levels.where(touched_frame)
            anchor = (
                touched_levels.min(axis=1, skipna=True)
                if side == "resistance"
                else touched_levels.max(axis=1, skipna=True)
            )
            rejection_distance = (
                (anchor - close) / anchor
                if side == "resistance"
                else (close - anchor) / anchor
            )
            overshoot = (
                (high - anchor) / anchor
                if side == "resistance"
                else (anchor - low) / anchor
            )
            rejection_wick = (
                (high - pd.concat((open_, close), axis=1).max(axis=1)) / candle_range
                if side == "resistance"
                else (pd.concat((open_, close), axis=1).min(axis=1) - low)
                / candle_range
            )
            base = f"%-{prefix}_{side}_weakening"
            dataframe[f"{base}_incoming_return"] = sign * prior_return
            dataframe[f"{base}_current_return"] = sign * return_1h
            dataframe[f"{base}_momentum_change"] = sign * (
                return_1h - prior_return
            )
            dataframe[f"{base}_current_close_pressure"] = sign * close_pressure
            dataframe[f"{base}_pressure_change"] = sign * (
                close_pressure - prior_pressure
            )
            dataframe[f"{base}_away_body_pressure"] = -sign * body_pressure
            dataframe[f"{base}_rejection_wick"] = rejection_wick.clip(0.0, 1.0)
            dataframe[f"{base}_volume_ratio"] = volume_ratio
            dataframe[f"{base}_touch_count"] = touched_frame.sum(axis=1).astype(float)
            dataframe[f"{base}_rejection_distance"] = rejection_distance.fillna(0.0)
            dataframe[f"{base}_overshoot"] = overshoot.fillna(0.0)
            dataframe[f"{base}_closed_beyond_level"] = (
                close.gt(anchor * (1.0 + TOUCH_BAND))
                if side == "resistance"
                else close.lt(anchor * (1.0 - TOUCH_BAND))
            ).fillna(False).astype(float)


class GeneralExitReadinessPriceControlFreqAIResearchStrategy(
    GeneralExitReadinessFreqAIResearchStrategy
):
    pass


class GeneralExitReadinessLevelQualityFreqAIResearchStrategy(
    GeneralExitReadinessFreqAIResearchStrategy
):
    include_single_levels = True


class GeneralExitReadinessProximityAlignmentFreqAIResearchStrategy(
    GeneralExitReadinessFreqAIResearchStrategy
):
    include_single_levels = True
    include_candidate_components = True


class GeneralExitReadinessWeakeningFreqAIResearchStrategy(
    GeneralExitReadinessFreqAIResearchStrategy
):
    include_single_levels = True
    include_hit_response = True
    include_candidate_components = True
    include_weakening_components = True


class GeneralExitReadinessBtcContextFreqAIResearchStrategy(
    GeneralExitReadinessWeakeningFreqAIResearchStrategy
):
    include_btc_levels_and_clusters = True
