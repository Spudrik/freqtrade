from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy


USER_DATA_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "general_exit_level_cache"
)

CACHE_FILES = {
    "BTC/USDT:USDT": CACHE_DIR / "btc_general_exit_levels_1h.parquet",
    "ETH/USDT:USDT": CACHE_DIR / "eth_general_exit_levels_1h.parquet",
    "SOL/USDT:USDT": CACHE_DIR / "sol_general_exit_levels_1h.parquet",
}

TIMEFRAMES = ("1h", "4h", "1d")
TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "8h": 8, "1d": 24, "3d": 72}
LEVEL_FAMILIES = ("va_edge", "prior_poc", "hvn", "tlv2")
MISSING_DISTANCE = 10.0


def _numeric(frame: DataFrame, column: str, default: float = np.nan) -> Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _future_extreme(series: Series, horizon: int, method: str) -> Series:
    future = pd.to_numeric(series, errors="coerce").shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    extreme = rolling.max() if method == "max" else rolling.min()
    return extreme.iloc[::-1]


def _atr(dataframe: DataFrame, period: int = 14) -> Series:
    high = pd.to_numeric(dataframe["high"], errors="coerce")
    low = pd.to_numeric(dataframe["low"], errors="coerce")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    prior_close = close.shift(1)
    true_range = pd.concat(
        ((high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()),
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def build_general_exit_targets(dataframe: DataFrame) -> DataFrame:
    """Build side-neutral path labels. Positive advantage means downside dominated."""
    out = dataframe.copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    atr_ratio = (_atr(out) / close).replace(0.0, np.nan)

    for horizon in (2, 4, 8, 24, 48):
        future_high = _future_extreme(high, horizon, "max")
        future_low = _future_extreme(low, horizon, "min")
        upside = (future_high / close) - 1.0
        downside = 1.0 - (future_low / close)
        advantage = ((downside - upside) / atr_ratio).clip(-10.0, 10.0)
        out[f"&-down_vs_up_advantage_{horizon}h"] = advantage
    return out


class GeneralExitTargetQualityFreqAIResearchStrategy(IStrategy):
    """Entry-independent FreqAI research for general target-zone exit quality."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = False
    trailing_stop = False

    feature_profile = "price_only"
    level_families = LEVEL_FAMILIES
    include_mtf_alignment = True
    include_btc_price_context = False
    include_btc_level_context = False
    include_btc_confluence_summary = False
    btc_level_families = LEVEL_FAMILIES
    _cache_frames: dict[str, DataFrame] = {}

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = period, metadata, kwargs
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = metadata, kwargs
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce").clip(lower=0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)

        for horizon in (1, 3, 6, 12, 24):
            dataframe[f"%-return_{horizon}h"] = close.pct_change(horizon)
        dataframe["%-atr_pct"] = _atr(dataframe) / close
        dataframe["%-body_pressure"] = ((close - open_) / candle_range).clip(-1.0, 1.0)
        dataframe["%-close_location"] = pressure
        dataframe["%-volume_z_24h"] = (
            (volume - volume.rolling(24, min_periods=12).mean())
            / volume.rolling(24, min_periods=12).std(ddof=0).replace(0.0, np.nan)
        )
        signed_volume = pressure.fillna(0.0) * volume.fillna(0.0)
        dataframe["%-volume_pressure_6h"] = (
            signed_volume.rolling(6, min_periods=3).sum()
            / volume.rolling(6, min_periods=3).sum().replace(0.0, np.nan)
        )
        dataframe["%-volume_pressure_24h"] = (
            signed_volume.rolling(24, min_periods=12).sum()
            / volume.rolling(24, min_periods=12).sum().replace(0.0, np.nan)
        )
        prior_high = high.shift(1).rolling(72, min_periods=36).max()
        prior_low = low.shift(1).rolling(72, min_periods=36).min()
        dataframe["%-range_position_72h"] = (
            (close - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)
        )
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = self._canonical_pair(str(metadata.get("pair") or ""))
        aligned = self._aligned_cache(dataframe, pair)
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)

        all_states: dict[str, dict[str, Any]] = {}
        for timeframe in TIMEFRAMES:
            features, state = self._level_features(
                aligned,
                close,
                timeframe,
                families=self.level_families,
            )
            all_states[timeframe] = state
            if self.feature_profile in {
                "level_1h",
                "level_mtf",
                "level_mtf_btc_price",
                "level_mtf_btc_levels",
            } and (timeframe == "1h" or self.feature_profile != "level_1h"):
                for name, values in features.items():
                    dataframe[f"%-{timeframe}_{name}"] = values

        diagnostics = self._aggregate_level_state(all_states)
        for name, values in diagnostics.items():
            dataframe[f"diag_{name}"] = values

        if self.include_mtf_alignment and self.feature_profile in {
            "level_mtf",
            "level_mtf_btc_price",
            "level_mtf_btc_levels",
        }:
            for name, values in self._mtf_alignment_features(
                all_states, families=self.level_families
            ).items():
                dataframe[f"%-mtf_{name}"] = values

        include_btc_price = self.include_btc_price_context or self.feature_profile in {
            "level_mtf_btc_price",
            "level_mtf_btc_levels",
        }
        include_btc_levels = (
            self.include_btc_level_context
            or self.feature_profile == "level_mtf_btc_levels"
        )
        if include_btc_price or include_btc_levels:
            btc = self._aligned_cache(dataframe, "BTC/USDT:USDT")
            if include_btc_price:
                self._append_btc_price_features(dataframe, btc, close)
            if include_btc_levels:
                btc_close = _numeric(btc, "close")
                btc_states: dict[str, dict[str, Any]] = {}
                for timeframe in TIMEFRAMES:
                    features, state = self._level_features(
                        btc,
                        btc_close,
                        timeframe,
                        families=self.btc_level_families,
                    )
                    btc_states[timeframe] = state
                    for name, values in features.items():
                        dataframe[f"%-btc_{timeframe}_{name}"] = values
                for name, values in self._mtf_alignment_features(
                    btc_states, families=self.btc_level_families
                ).items():
                    dataframe[f"%-btc_mtf_{name}"] = values
                if self.include_btc_confluence_summary:
                    for name, values in self._aggregate_level_state(btc_states).items():
                        dataframe[f"%-btc_diag_{name}"] = values
                for timeframe in TIMEFRAMES:
                    alt_state = all_states[timeframe]
                    btc_state = btc_states[timeframe]
                    dataframe[f"%-btc_alt_resistance_distance_gap_{timeframe}"] = (
                        alt_state["nearest_resistance"] - btc_state["nearest_resistance"]
                    ).abs().fillna(MISSING_DISTANCE)
                    dataframe[f"%-btc_alt_support_distance_gap_{timeframe}"] = (
                        alt_state["nearest_support"] - btc_state["nearest_support"]
                    ).abs().fillna(MISSING_DISTANCE)
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = metadata, kwargs
        return build_general_exit_targets(dataframe)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @classmethod
    def _canonical_pair(cls, pair: str) -> str:
        normalized = pair.strip().upper()
        if normalized in CACHE_FILES:
            return normalized
        base = normalized.split("/")[0].split(":")[0]
        for candidate in CACHE_FILES:
            if candidate.startswith(f"{base}/"):
                return candidate
        raise ValueError(f"Unsupported general-exit research pair: {pair!r}")

    @classmethod
    def _load_cache(cls, pair: str) -> DataFrame:
        canonical = cls._canonical_pair(pair)
        cached = cls._cache_frames.get(canonical)
        if cached is not None:
            return cached
        path = CACHE_FILES[canonical]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        frame = frame.sort_values("date").set_index("date")
        cls._cache_frames[canonical] = frame
        return frame

    @classmethod
    def _aligned_cache(cls, dataframe: DataFrame, pair: str) -> DataFrame:
        cache = cls._load_cache(pair)
        decision_dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce") + pd.Timedelta(hours=1)
        aligned = cache.reindex(pd.DatetimeIndex(decision_dates)).reset_index(drop=True)
        aligned.index = dataframe.index
        return aligned

    @staticmethod
    def _level_features(
        aligned: DataFrame,
        close: Series,
        timeframe: str,
        *,
        families: tuple[str, ...] = LEVEL_FAMILIES,
    ) -> tuple[dict[str, Series], dict[str, Any]]:
        prefix = f"st_{timeframe}"
        tf_factor = TIMEFRAME_HOURS[timeframe]

        prior_poc = _numeric(aligned, f"{prefix}_vp_prior_poc")
        prior_vah = _numeric(aligned, f"{prefix}_vp_prior_vah")
        prior_val = _numeric(aligned, f"{prefix}_vp_prior_val")
        hvn_above = _numeric(aligned, f"{prefix}_vp_hvn_above")
        hvn_below = _numeric(aligned, f"{prefix}_vp_hvn_below")
        resistance_line = _numeric(aligned, f"{prefix}_tlv2_resistance_line_rank0")
        support_line = _numeric(aligned, f"{prefix}_tlv2_support_line_rank0")

        def above_distance(level: Series) -> Series:
            distance = (level / close) - 1.0
            return distance.where(distance.ge(0.0))

        def below_distance(level: Series) -> Series:
            distance = 1.0 - (level / close)
            return distance.where(distance.ge(0.0))

        resistance_all = DataFrame(
            {
                "va_edge": above_distance(prior_vah),
                "prior_poc": above_distance(prior_poc),
                "hvn": above_distance(hvn_above),
                "tlv2": above_distance(resistance_line),
            },
            index=aligned.index,
        )
        support_all = DataFrame(
            {
                "va_edge": below_distance(prior_val),
                "prior_poc": below_distance(prior_poc),
                "hvn": below_distance(hvn_below),
                "tlv2": below_distance(support_line),
            },
            index=aligned.index,
        )
        invalid_families = sorted(set(families) - set(LEVEL_FAMILIES))
        if invalid_families:
            raise ValueError(f"Unsupported level families: {invalid_families}")
        if not families:
            raise ValueError("At least one level family is required")
        resistance = resistance_all.loc[:, list(families)]
        support = support_all.loc[:, list(families)]
        nearest_resistance, second_resistance, resistance_family = (
            GeneralExitTargetQualityFreqAIResearchStrategy._nearest_state(resistance)
        )
        nearest_support, second_support, support_family = (
            GeneralExitTargetQualityFreqAIResearchStrategy._nearest_state(support)
        )

        features: dict[str, Series] = {}
        for family in resistance.columns:
            features[f"resistance_{family}_distance"] = resistance[family].fillna(MISSING_DISTANCE)
            features[f"support_{family}_distance"] = support[family].fillna(MISSING_DISTANCE)
            features[f"resistance_{family}_present"] = resistance[family].notna().astype(float)
            features[f"support_{family}_present"] = support[family].notna().astype(float)
            features[f"nearest_resistance_is_{family}"] = resistance_family.eq(family).astype(float)
            features[f"nearest_support_is_{family}"] = support_family.eq(family).astype(float)

        features["nearest_resistance_distance"] = nearest_resistance.fillna(MISSING_DISTANCE)
        features["nearest_support_distance"] = nearest_support.fillna(MISSING_DISTANCE)
        features["second_resistance_gap"] = (second_resistance - nearest_resistance).fillna(MISSING_DISTANCE)
        features["second_support_gap"] = (second_support - nearest_support).fillna(MISSING_DISTANCE)
        for band, label in ((0.0025, "25bp"), (0.005, "50bp"), (0.01, "100bp")):
            features[f"resistance_confluence_{label}"] = resistance.sub(nearest_resistance, axis=0).abs().le(band).sum(axis=1).astype(float)
            features[f"support_confluence_{label}"] = support.sub(nearest_support, axis=0).abs().le(band).sum(axis=1).astype(float)

        if "hvn" in families:
            features["hvn_above_strength"] = _numeric(aligned, f"{prefix}_vp_hvn_above_strength", 0.0).fillna(0.0)
            features["hvn_below_strength"] = _numeric(aligned, f"{prefix}_vp_hvn_below_strength", 0.0).fillna(0.0)
        if "va_edge" in families:
            features["value_area_width_pct"] = _numeric(aligned, f"{prefix}_vp_value_area_width_pct", 0.0).fillna(0.0)
            features["value_area_position"] = _numeric(aligned, f"{prefix}_vp_value_area_position", 0.5).fillna(0.5)
        if set(families) == set(LEVEL_FAMILIES):
            features["vp_score_long"] = _numeric(aligned, f"{prefix}_vp_score_long", 0.0).fillna(0.0)
            features["vp_score_short"] = _numeric(aligned, f"{prefix}_vp_score_short", 0.0).fillna(0.0)
        if "tlv2" in families:
            features["tlv2_resistance_score"] = _numeric(aligned, f"{prefix}_tlv2_resistance_score_rank0", 0.0).fillna(0.0)
            features["tlv2_support_score"] = _numeric(aligned, f"{prefix}_tlv2_support_score_rank0", 0.0).fillna(0.0)
            features["tlv2_resistance_pivots"] = _numeric(aligned, f"{prefix}_tlv2_resistance_pivot_count_rank0", 0.0).fillna(0.0)
            features["tlv2_support_pivots"] = _numeric(aligned, f"{prefix}_tlv2_support_pivot_count_rank0", 0.0).fillna(0.0)
            features["tlv2_resistance_stability_age"] = GeneralExitTargetQualityFreqAIResearchStrategy._stability_age(resistance_line, tf_factor)
            features["tlv2_support_stability_age"] = GeneralExitTargetQualityFreqAIResearchStrategy._stability_age(support_line, tf_factor)
        if "prior_poc" in families:
            features["poc_stability_age"] = GeneralExitTargetQualityFreqAIResearchStrategy._stability_age(prior_poc, tf_factor)

        state = {
            "resistance": resistance,
            "support": support,
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
        }
        return features, state

    @staticmethod
    def _nearest_state(candidates: DataFrame) -> tuple[Series, Series, Series]:
        values = candidates.to_numpy(dtype="float64")
        ordered = np.sort(np.where(np.isfinite(values), values, np.nan), axis=1)
        nearest = pd.Series(ordered[:, 0], index=candidates.index, dtype="float64")
        second_values = (
            ordered[:, 1]
            if ordered.shape[1] > 1
            else np.full(len(candidates), np.nan, dtype="float64")
        )
        second = pd.Series(second_values, index=candidates.index, dtype="float64")
        finite = np.isfinite(values)
        nearest_index = np.argmin(np.where(finite, values, np.inf), axis=1)
        names = np.asarray(candidates.columns, dtype="object")[nearest_index]
        names[~finite.any(axis=1)] = ""
        family = pd.Series(names, index=candidates.index, dtype="object")
        return nearest, second, family

    @staticmethod
    def _stability_age(level: Series, timeframe_hours: int) -> Series:
        numeric = pd.to_numeric(level, errors="coerce")
        relative_change = numeric.pct_change(fill_method=None).abs()
        changed = numeric.isna() | numeric.shift(1).isna() | relative_change.gt(0.001)
        age_hours = numeric.groupby(changed.cumsum()).cumcount().astype(float)
        return (age_hours / max(1, timeframe_hours)).where(numeric.notna(), 0.0)

    @staticmethod
    def _aggregate_level_state(states: dict[str, dict[str, Any]]) -> dict[str, Series]:
        resistance = pd.concat(
            [state["resistance"].add_prefix(f"{tf}_") for tf, state in states.items()],
            axis=1,
        )
        support = pd.concat(
            [state["support"].add_prefix(f"{tf}_") for tf, state in states.items()],
            axis=1,
        )
        nearest_resistance = resistance.min(axis=1, skipna=True)
        nearest_support = support.min(axis=1, skipna=True)
        return {
            "nearest_resistance_pct": nearest_resistance.fillna(MISSING_DISTANCE),
            "nearest_support_pct": nearest_support.fillna(MISSING_DISTANCE),
            "resistance_confluence_50bp": resistance.sub(nearest_resistance, axis=0).abs().le(0.005).sum(axis=1).astype(float),
            "support_confluence_50bp": support.sub(nearest_support, axis=0).abs().le(0.005).sum(axis=1).astype(float),
        }

    @staticmethod
    def _mtf_alignment_features(
        states: dict[str, dict[str, Any]], *, families: tuple[str, ...] = LEVEL_FAMILIES
    ) -> dict[str, Series]:
        features: dict[str, Series] = {}
        pairs = (("1h", "4h"), ("1h", "1d"), ("4h", "1d"))
        for left, right in pairs:
            features[f"resistance_alignment_{left}_{right}"] = (
                states[left]["nearest_resistance"] - states[right]["nearest_resistance"]
            ).abs().fillna(MISSING_DISTANCE)
            features[f"support_alignment_{left}_{right}"] = (
                states[left]["nearest_support"] - states[right]["nearest_support"]
            ).abs().fillna(MISSING_DISTANCE)
            for family in families:
                features[f"resistance_{family}_alignment_{left}_{right}"] = (
                    states[left]["resistance"][family] - states[right]["resistance"][family]
                ).abs().fillna(MISSING_DISTANCE)
                features[f"support_{family}_alignment_{left}_{right}"] = (
                    states[left]["support"][family] - states[right]["support"][family]
                ).abs().fillna(MISSING_DISTANCE)
        resistance_nearest = pd.concat(
            [states[tf]["nearest_resistance"].rename(tf) for tf in TIMEFRAMES], axis=1
        )
        support_nearest = pd.concat(
            [states[tf]["nearest_support"].rename(tf) for tf in TIMEFRAMES], axis=1
        )
        features["resistance_timeframe_spread"] = (
            resistance_nearest.max(axis=1) - resistance_nearest.min(axis=1)
        ).fillna(MISSING_DISTANCE)
        features["support_timeframe_spread"] = (
            support_nearest.max(axis=1) - support_nearest.min(axis=1)
        ).fillna(MISSING_DISTANCE)
        return features

    @staticmethod
    def _append_btc_price_features(dataframe: DataFrame, btc: DataFrame, alt_close: Series) -> None:
        btc_close = _numeric(btc, "close").replace(0.0, np.nan)
        btc_volume = _numeric(btc, "volume", 0.0).clip(lower=0.0)
        for horizon in (1, 6, 24):
            btc_return = btc_close.pct_change(horizon)
            dataframe[f"%-btc_return_{horizon}h"] = btc_return
            dataframe[f"%-btc_alt_return_gap_{horizon}h"] = (
                alt_close.pct_change(horizon) - btc_return
            )
        dataframe["%-btc_volume_z_24h"] = (
            (btc_volume - btc_volume.rolling(24, min_periods=12).mean())
            / btc_volume.rolling(24, min_periods=12).std(ddof=0).replace(0.0, np.nan)
        )
        dataframe["%-btc_alt_return_corr_72h"] = alt_close.pct_change().rolling(
            72, min_periods=36
        ).corr(btc_close.pct_change())


class GeneralExitPriceOnlyFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "price_only"


class GeneralExitLevel1hFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_1h"


class GeneralExitLevelMtfFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"


class GeneralExitLevelMtfRawFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"
    include_mtf_alignment = False


class GeneralExitVaMtfFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"
    level_families = ("va_edge",)


class GeneralExitPocMtfFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"
    level_families = ("prior_poc",)


class GeneralExitHvnMtfFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"
    level_families = ("hvn",)


class GeneralExitTlv2MtfFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"
    level_families = ("tlv2",)


class GeneralExitVaTlv2MtfFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf"
    level_families = ("va_edge", "tlv2")


class GeneralExitLevelMtfBtcPriceFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf_btc_price"


class GeneralExitLevelMtfBtcLevelsFreqAIResearchStrategy(GeneralExitTargetQualityFreqAIResearchStrategy):
    feature_profile = "level_mtf_btc_levels"


class GeneralExitLevelMtfBtcPocConfluenceFreqAIResearchStrategy(
    GeneralExitTargetQualityFreqAIResearchStrategy
):
    feature_profile = "level_mtf"
    include_btc_level_context = True
    include_btc_confluence_summary = True
    btc_level_families = ("prior_poc",)


class GeneralExitLevelMtfBtcHvnConfluenceFreqAIResearchStrategy(
    GeneralExitTargetQualityFreqAIResearchStrategy
):
    feature_profile = "level_mtf"
    include_btc_level_context = True
    include_btc_confluence_summary = True
    btc_level_families = ("hvn",)


class GeneralExitLevelMtfBtcPocHvnConfluenceFreqAIResearchStrategy(
    GeneralExitTargetQualityFreqAIResearchStrategy
):
    feature_profile = "level_mtf"
    include_btc_level_context = True
    include_btc_confluence_summary = True
    btc_level_families = ("prior_poc", "hvn")


class GeneralExitLevelMtfBtcPocHvnVaConfluenceFreqAIResearchStrategy(
    GeneralExitTargetQualityFreqAIResearchStrategy
):
    feature_profile = "level_mtf"
    include_btc_level_context = True
    include_btc_confluence_summary = True
    btc_level_families = ("prior_poc", "hvn", "va_edge")
