from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.strategies.GeneralExitLevelReactionFreqAIResearchStrategy import (
    GeneralExitLevelReactionFreqAIResearchStrategy,
)
from user_data.strategies.sieve3_event_reaction_targets import (
    EXIT_ROLE_HORIZONS,
    EXIT_ROLE_SEQUENCE_HORIZONS,
    TARGET_HORIZONS,
    TOUCH_HORIZONS,
    _atr,
    build_sieve3_exit_role_targets,
    build_sieve3_reaction_targets,
)


USER_DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "round1"
    / "cache"
)
def _pair_token(pair: str) -> str:
    base = pair.strip().upper().split("/")[0].split(":")[0]
    if not base or not base.replace("-", "").isalnum():
        raise ValueError(f"Unsupported Sieve3 event-reaction pair: {pair!r}")
    return base.lower()


class Sieve3EventReactionFreqAIResearchStrategy(
    GeneralExitLevelReactionFreqAIResearchStrategy
):
    """Research-only model ladder for exact Sieve3 events and nearby levels."""

    startup_candle_count = 2200
    include_sieve_events = False
    include_sieve_active_events = False
    include_fixed_indicators = False
    include_level_context = False
    include_single_levels = False
    include_same_asset_clusters = False
    include_hit_response = False
    include_hit_identities = False
    include_btc_levels_and_clusters = False
    _event_cache_frames: dict[str, DataFrame] = {}

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        dataframe = super().feature_engineering_expand_basic(
            dataframe, metadata, **kwargs
        )
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce").clip(lower=0.0)
        atr = _atr(dataframe).replace(0.0, np.nan)
        candle_range = (high - low).replace(0.0, np.nan)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)

        for lookback in (24, 168, 720, 2160):
            prior_high = high.shift(1).rolling(lookback, min_periods=max(12, lookback // 2)).max()
            prior_low = low.shift(1).rolling(lookback, min_periods=max(12, lookback // 2)).min()
            width = (prior_high - prior_low).replace(0.0, np.nan)
            dataframe[f"%-range_position_{lookback}h"] = (close - prior_low) / width
            dataframe[f"%-distance_prior_high_{lookback}h_atr"] = (
                (prior_high - close) / atr
            ).clip(-20.0, 20.0)
            dataframe[f"%-distance_prior_low_{lookback}h_atr"] = (
                (close - prior_low) / atr
            ).clip(-20.0, 20.0)
            dataframe[f"%-break_above_prior_high_{lookback}h"] = close.gt(prior_high).astype(float)
            dataframe[f"%-break_below_prior_low_{lookback}h"] = close.lt(prior_low).astype(float)

        for lookback in (6, 24, 168):
            dataframe[f"%-volume_ratio_{lookback}h"] = (
                volume / volume.shift(1).rolling(lookback, min_periods=max(3, lookback // 2)).mean()
            ).clip(0.0, 20.0)
            pressure = close_location.fillna(0.0) * volume.fillna(0.0)
            dataframe[f"%-pressure_{lookback}h"] = (
                pressure.rolling(lookback, min_periods=max(3, lookback // 2)).sum()
                / volume.rolling(lookback, min_periods=max(3, lookback // 2)).sum().replace(0.0, np.nan)
            )
        dataframe["%-pressure_change_6h_vs_24h"] = (
            dataframe["%-pressure_6h"] - dataframe["%-pressure_24h"]
        )
        dataframe["%-range_expansion_vs_24h"] = (
            candle_range / candle_range.shift(1).rolling(24, min_periods=12).mean().replace(0.0, np.nan)
        ).clip(0.0, 20.0)
        if self.include_fixed_indicators:
            dataframe = self._add_fixed_indicator_control(dataframe)
        return dataframe

    @staticmethod
    def _add_fixed_indicator_control(dataframe: DataFrame) -> DataFrame:
        """Add the frozen, timestamp-safe Generation 1 indicator control."""
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        atr = _atr(dataframe).replace(0.0, np.nan)

        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        dataframe["%-fixed_rsi_14"] = (100.0 - (100.0 / (1.0 + relative_strength))) / 100.0

        bollinger_mid = close.rolling(20, min_periods=20).mean()
        bollinger_std = close.rolling(20, min_periods=20).std(ddof=0)
        dataframe["%-fixed_bollinger_position_20"] = (
            (close - bollinger_mid) / bollinger_std.replace(0.0, np.nan)
        ).clip(-10.0, 10.0)
        dataframe["%-fixed_bollinger_width_20_atr"] = (
            (4.0 * bollinger_std) / atr
        ).clip(0.0, 50.0)

        ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
        dataframe["%-fixed_macd_12_26_atr"] = (macd / atr).clip(-20.0, 20.0)
        dataframe["%-fixed_macd_histogram_atr"] = (
            (macd - macd_signal) / atr
        ).clip(-20.0, 20.0)

        sma_20 = close.rolling(20, min_periods=20).mean()
        ema_20 = close.ewm(span=20, min_periods=20, adjust=False).mean()
        dataframe["%-fixed_sma_20_distance_atr"] = ((close - sma_20) / atr).clip(
            -20.0, 20.0
        )
        dataframe["%-fixed_sma_20_slope_5_atr"] = (sma_20.diff(5) / atr).clip(
            -20.0, 20.0
        )
        dataframe["%-fixed_ema_20_distance_atr"] = ((close - ema_20) / atr).clip(
            -20.0, 20.0
        )
        dataframe["%-fixed_ema_20_slope_5_atr"] = (ema_20.diff(5) / atr).clip(
            -20.0, 20.0
        )

        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        upward = high.diff()
        downward = -low.diff()
        plus_dm = upward.where((upward > downward) & (upward > 0.0), 0.0)
        minus_dm = downward.where((downward > upward) & (downward > 0.0), 0.0)
        smoothed_tr = true_range.ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean()
        plus_di = 100.0 * plus_dm.ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean() / smoothed_tr.replace(0.0, np.nan)
        minus_di = 100.0 * minus_dm.ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean() / smoothed_tr.replace(0.0, np.nan)
        directional_sum = (plus_di + minus_di).replace(0.0, np.nan)
        directional_index = 100.0 * (plus_di - minus_di).abs() / directional_sum
        dataframe["%-fixed_adx_14"] = directional_index.ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean() / 100.0
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        if self.include_level_context:
            dataframe = super().feature_engineering_standard(
                dataframe, metadata, **kwargs
            )
        if self.include_sieve_events:
            pair = str(metadata.get("pair") or "")
            aligned = self._aligned_event_cache(dataframe, pair)
            settings = self.config.get("sieve3_event_reaction", {})
            configured_columns = settings.get("event_feature_columns")
            configured_prefixes = settings.get("event_feature_prefixes")
            if configured_columns:
                requested = [str(column) for column in configured_columns]
                missing = [column for column in requested if column not in aligned]
                if missing:
                    raise ValueError(
                        f"Configured Sieve3 event features are missing for {pair}: {missing}"
                    )
                feature_columns = requested
            elif configured_prefixes:
                prefixes = tuple(str(prefix) for prefix in configured_prefixes)
                feature_columns = [
                    column for column in aligned.columns if column.startswith(prefixes)
                ]
            else:
                feature_columns = [
                    column
                    for column in aligned.columns
                    if column.startswith("entry_onset__")
                    or (
                        self.include_sieve_active_events
                        and column.startswith("entry_active__")
                    )
                    or column.startswith("summary__")
                ]
            if not feature_columns:
                raise ValueError(f"No Sieve3 event feature columns were found for {pair}")
            for column in feature_columns:
                dataframe[f"%-sieve3_{column}"] = pd.to_numeric(
                    aligned[column], errors="coerce"
                ).fillna(0.0)
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = metadata, kwargs
        selected = self.config.get("sieve3_event_reaction", {}).get("target_names")
        if selected and any(str(target).startswith("&-exit_") for target in selected):
            labelled = build_sieve3_exit_role_targets(dataframe)
        else:
            labelled = build_sieve3_reaction_targets(dataframe)
        if not selected:
            return labelled
        selected_targets = {str(target) for target in selected}
        for horizon in TARGET_HORIZONS:
            reaction_target = f"&-future_reaction_magnitude_{horizon}h_atr"
            if reaction_target not in selected_targets:
                continue
            upside = f"&-future_upside_{horizon}h_atr"
            downside = f"&-future_downside_{horizon}h_atr"
            if upside not in labelled or downside not in labelled:
                raise ValueError(
                    f"Cannot derive configured reaction-magnitude target: {reaction_target}"
                )
            labelled[reaction_target] = pd.concat(
                [
                    pd.to_numeric(labelled[upside], errors="coerce"),
                    pd.to_numeric(labelled[downside], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1, skipna=False)
        available = {column for column in labelled.columns if column.startswith("&-")}
        missing = selected_targets - available
        if missing:
            raise ValueError(f"Configured Sieve3 reaction targets are missing: {sorted(missing)}")
        unused = available - selected_targets
        return labelled.drop(columns=sorted(unused))

    def _event_cache_dir(self) -> Path:
        settings = self.config.get("sieve3_event_reaction", {})
        configured = settings.get("event_cache_dir")
        path = Path(configured) if configured else DEFAULT_EVENT_CACHE_DIR
        if not path.is_absolute():
            path = USER_DATA_DIR / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()

    def _load_event_cache(self, pair: str) -> DataFrame:
        path = self._event_cache_dir() / f"{_pair_token(pair)}_sieve3_events_1h.parquet"
        cache_key = str(path)
        cached = self._event_cache_frames.get(cache_key)
        if cached is not None:
            return cached
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["decision_time"] = pd.to_datetime(
            frame["decision_time"], utc=True, errors="coerce"
        )
        frame = (
            frame.dropna(subset=["decision_time"])
            .drop_duplicates("decision_time", keep="last")
            .sort_values("decision_time")
            .set_index("decision_time")
        )
        self._event_cache_frames[cache_key] = frame
        return frame

    def _aligned_event_cache(self, dataframe: DataFrame, pair: str) -> DataFrame:
        cache = self._load_event_cache(pair)
        settings = self.config.get("sieve3_event_reaction", {})
        shift_hours = int(settings.get("event_feature_shift_hours", 0) or 0)
        if shift_hours:
            cache = cache.copy()
            cache.index = cache.index + pd.Timedelta(hours=shift_hours)
        decision_dates = pd.to_datetime(
            dataframe["date"], utc=True, errors="coerce"
        ) + pd.Timedelta(hours=1)
        valid_dates = decision_dates.dropna()
        if valid_dates.empty:
            raise ValueError(f"No valid decision dates for {pair}")
        if valid_dates.min() < cache.index.min():
            raise ValueError(
                f"Sieve3 event cache does not cover {pair} model window: "
                f"{valid_dates.min()}..{valid_dates.max()} vs "
                f"{cache.index.min()}..{cache.index.max()}"
            )
        aligned = cache.reindex(pd.DatetimeIndex(decision_dates)).reset_index(drop=True)
        aligned.index = dataframe.index
        coverage = (
            pd.to_numeric(aligned["coverage_present"], errors="coerce").eq(1.0)
            if "coverage_present" in aligned
            else pd.Series(False, index=dataframe.index)
        )
        missing = ~coverage
        if missing.any():
            scope_end_raw = settings.get("event_scope_end")
            scope_end = pd.to_datetime(scope_end_raw, utc=True, errors="coerce")
            if pd.isna(scope_end):
                raise ValueError(
                    f"Sieve3 event cache has missing hourly coverage for {pair} and "
                    "no valid event_scope_end was configured"
                )
            out_of_scope_padding = decision_dates.ge(scope_end)
            invalid_missing = missing & ~out_of_scope_padding
            if invalid_missing.any():
                invalid_dates = decision_dates[invalid_missing]
                raise ValueError(
                    f"Sieve3 event cache has {int(invalid_missing.sum())} in-scope "
                    f"coverage gaps for {pair}: {invalid_dates.min()}..{invalid_dates.max()}"
                )
        return aligned


class Sieve3EventReactionPriceControlFreqAIResearchStrategy(
    Sieve3EventReactionFreqAIResearchStrategy
):
    pass


class Sieve3EventReactionLevelContextFreqAIResearchStrategy(
    Sieve3EventReactionFreqAIResearchStrategy
):
    include_level_context = True
    include_single_levels = True


class Sieve3EventReactionIdentityFreqAIResearchStrategy(
    Sieve3EventReactionFreqAIResearchStrategy
):
    include_sieve_events = True


class Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy(
    Sieve3EventReactionFreqAIResearchStrategy
):
    include_fixed_indicators = True


class Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy(
    Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy
):
    include_sieve_events = True


class Sieve3EventReactionFixedIndicatorIdentityActiveFreqAIResearchStrategy(
    Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy
):
    include_sieve_active_events = True


class Sieve3EventReactionFixedIndicatorShiftedOnsetFreqAIResearchStrategy(
    Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy
):
    """Fixed-indicator onset-only placebo using an explicitly configured past shift."""

    pass


class Sieve3EventReactionFixedIndicatorShiftedIdentityFreqAIResearchStrategy(
    Sieve3EventReactionFixedIndicatorIdentityActiveFreqAIResearchStrategy
):
    """Fixed-indicator onset-plus-active placebo using a configured past shift."""

    pass


class Sieve3EventReactionShiftedIdentityFreqAIResearchStrategy(
    Sieve3EventReactionIdentityFreqAIResearchStrategy
):
    """Event-identity placebo; the configured cache shift is recorded per run."""

    pass


class Sieve3EventReactionEventLevelInteractionFreqAIResearchStrategy(
    Sieve3EventReactionFreqAIResearchStrategy
):
    include_sieve_events = True
    include_level_context = True
    include_single_levels = True
    include_same_asset_clusters = True
    include_hit_response = True
    include_hit_identities = True
