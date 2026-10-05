from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import talib
import talib.abstract as ta
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[5]
BASE_STRATEGY_PATH = (
    REPO_ROOT
    / "user_data"
    / "strategies"
    / "Archive"
    / "non_sieve3_top_level_strategies_archived_20260612"
    / "ContextFreqAIResearchStrategy.py"
)
spec = importlib.util.spec_from_file_location("_context_freqai_research_base", BASE_STRATEGY_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load base FreqAI strategy from {BASE_STRATEGY_PATH}")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
ContextFreqAIResearchStrategy = base_module.ContextFreqAIResearchStrategy


class ThreeKiHistoricalNewsFreqAIBaseStrategy(ContextFreqAIResearchStrategy):
    """
    FreqAI-only historical GDELT/GKG research setup for 2020-2022.

    This class does not run its own ML. It only supplies timestamp-safe historical
    news/context features to FreqAI and defines the labels FreqAI should learn.
    """

    timeframe = "1h"
    startup_candle_count = 360
    can_short = False
    feature_mode = "combined"
    include_trader_event_labels = False
    include_orderbook_context = False
    include_structure_context = False
    include_confluence_event_features = False
    include_confluence_event_targets = False
    include_trader_confluence_context = False
    include_technical_context_features = False
    include_ta_confluence_features = False
    return_label_horizons = (1, 3, 6, 24, 72)
    path_label_horizons = (3, 6, 24, 72)
    context_max_age_hours = 0
    gdelt_context_db = r"C:\FreqTradeStuff\user_data\research_news_data\gdelt\gdelt_context.sqlite"
    historical_news_sources = ("gdelt", "gkg")
    gdelt_feature_set = "all"
    gdelt_feature_style = "standard"
    trader_state_profile = "none"
    ta_confluence_profile = "none"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50, 100, 200)
    reward_danger_threshold = 0.03
    _historical_news_cache: dict[tuple[str, tuple[str, ...], str, str], DataFrame] = {}

    @classmethod
    def _load_context_features(cls) -> DataFrame:
        db_path = Path(cls.gdelt_context_db)
        sources = tuple(str(item).lower() for item in cls.historical_news_sources)
        cache_key = (str(db_path), sources, str(cls.gdelt_feature_set), str(cls.gdelt_feature_style))
        cached = cls._historical_news_cache.get(cache_key)
        if cached is not None:
            return cached
        if not db_path.exists():
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._historical_news_cache[cache_key] = empty
            return empty

        frames: list[DataFrame] = []
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            if "gdelt" in sources:
                frames.append(cls._load_gdelt_event_features(conn))
            if "gkg" in sources:
                frames.append(cls._load_gkg_document_features(conn))
        if not frames:
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._historical_news_cache[cache_key] = empty
            return empty

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")
        merged = merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        merged["max_source_available_at"] = merged["date"]
        merged = cls._select_context_style_features(merged)
        cls._historical_news_cache[cache_key] = merged
        return merged

    @classmethod
    def _select_context_style_features(cls, context: DataFrame) -> DataFrame:
        style = str(cls.gdelt_feature_style)
        if style == "standard" or context.empty:
            return context
        keep = ["date"]
        if "max_source_available_at" in context:
            keep.append("max_source_available_at")
        feature_columns = [column for column in context.columns if column not in {"date", "max_source_available_at"}]
        style_patterns = {
            "level": lambda column: not any(token in column for token in ("_6h_sum", "_24h_sum", "_72h_sum", "_z_7d", "_rising_")),
            "change": lambda column: any(token in column for token in ("_6h_sum", "_24h_sum", "_72h_sum")),
            "shock": lambda column: "_z_7d" in column,
            "persistence": lambda column: "_72h_sum" in column or "_rising_" in column,
            "short_persistence": lambda column: "_6h_sum" in column or "_24h_sum" in column or "_rising_" in column,
            "deep_persistence": lambda column: "_72h_sum" in column or "_rising_" in column,
            "shock_persistence": lambda column: "_z_7d" in column or "_72h_sum" in column or "_rising_" in column,
        }
        selector = style_patterns.get(style)
        if selector is None:
            raise ValueError(f"Unknown GDELT feature style: {style}")
        selected = keep + [column for column in feature_columns if selector(column)]
        return context.loc[:, [column for column in selected if column in context]].copy()

    @staticmethod
    def _prefix_columns(frame: DataFrame, prefix: str) -> DataFrame:
        renamed = {
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column not in {"date", "parse_error"}
        }
        return frame.rename(columns=renamed)

    @classmethod
    def _load_gdelt_event_features(cls, conn: sqlite3.Connection) -> DataFrame:
        selected_columns = cls._gdelt_selected_columns()
        frame = pd.read_sql_query(
            f"""
            SELECT date,
                   {", ".join(selected_columns)}
            FROM gdelt_hourly_features
            WHERE parse_error IS NULL
            ORDER BY date
            """,
            conn,
        )
        if frame.empty:
            return DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce") + pd.Timedelta(hours=1)
        frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        frame = cls._prefix_columns(frame, "gdelt")
        return cls._add_rolling_news_features(frame, "gdelt")

    @classmethod
    def _gdelt_selected_columns(cls) -> list[str]:
        groups = {
            "attention": [
                "event_count",
                "num_mentions_sum",
                "num_sources_sum",
                "num_articles_sum",
            ],
            "tone_impact": [
                "avg_tone_weighted",
                "goldstein_weighted",
            ],
            "conflict_risk": [
                "conflict_event_count",
                "protest_event_count",
                "coercion_event_count",
                "sanctions_trade_url_count",
            ],
            "market_topics": [
                "oil_energy_url_count",
                "banking_credit_url_count",
                "macro_url_count",
                "crypto_url_count",
            ],
        }
        feature_set = str(cls.gdelt_feature_set)
        if feature_set == "all":
            selected: list[str] = []
            for values in groups.values():
                selected.extend(values)
            return selected
        if feature_set not in groups:
            raise ValueError(f"Unknown GDELT feature set: {feature_set}")
        return groups[feature_set]

    @classmethod
    def _load_gkg_document_features(cls, conn: sqlite3.Connection) -> DataFrame:
        frame = pd.read_sql_query(
            """
            SELECT date,
                   document_count,
                   source_count,
                   word_count_sum,
                   tone_sum,
                   positive_tone_sum,
                   negative_tone_sum,
                   polarity_sum,
                   activity_sum,
                   theme_count_sum,
                   unique_theme_count,
                   top_theme_doc_count,
                   crypto_doc_count,
                   bitcoin_doc_count,
                   ethereum_doc_count,
                   stablecoin_liquidity_doc_count,
                   regulation_doc_count,
                   etf_institutional_doc_count,
                   security_exploit_doc_count,
                   macro_economic_doc_count,
                   central_bank_doc_count,
                   rates_doc_count,
                   inflation_doc_count,
                   recession_growth_doc_count,
                   banking_credit_doc_count,
                   oil_energy_doc_count,
                   sanctions_trade_doc_count,
                   war_geopolitics_doc_count
            FROM gdelt_gkg_file_features
            WHERE parse_error IS NULL
            ORDER BY date
            """,
            conn,
        )
        if frame.empty:
            return DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"])
        frame["date"] = frame["date"].dt.floor("h") + pd.Timedelta(hours=1)
        agg = {}
        for column in frame.columns:
            if column == "date":
                continue
            agg[column] = "mean" if column in {"source_count", "unique_theme_count"} else "sum"
        hourly = frame.groupby("date", as_index=False).agg(agg)
        hourly = cls._prefix_columns(hourly, "gkg")
        return cls._add_rolling_news_features(hourly, "gkg")

    @staticmethod
    def _add_rolling_news_features(frame: DataFrame, prefix: str) -> DataFrame:
        frame = frame.sort_values("date").reset_index(drop=True)
        numeric_columns = [column for column in frame.columns if column != "date"]
        additions = {}
        for column in numeric_columns:
            series = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            frame[column] = series
            for window in (6, 24, 72):
                rolling = series.rolling(window, min_periods=max(2, window // 3)).sum()
                additions[f"{column}_{window}h_sum"] = rolling
            mean_168 = series.rolling(168, min_periods=48).mean()
            std_168 = series.rolling(168, min_periods=48).std()
            additions[f"{column}_z_7d"] = (series - mean_168) / std_168.replace(0, pd.NA)
        key_count = f"{prefix}_event_count" if prefix == "gdelt" else f"{prefix}_document_count"
        if key_count in frame:
            count = pd.to_numeric(frame[key_count], errors="coerce").fillna(0.0)
            additions[f"{prefix}_attention_rising_24h"] = (
                count.rolling(6, min_periods=2).sum() > count.rolling(24, min_periods=8).mean()
            ).astype(float)
        if additions:
            frame = pd.concat([frame, DataFrame(additions, index=frame.index)], axis=1)
        return frame.copy()

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe = super().feature_engineering_expand_basic(dataframe, metadata, **kwargs)
        if self.include_technical_context_features and self._include_price_features():
            dataframe = self._append_technical_context_features(dataframe)
        if self.include_ta_confluence_features and self._include_price_features():
            dataframe = self._append_ta_confluence_features(dataframe)
        return dataframe

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.replace(0, pd.NA)

    def _append_technical_context_features(self, dataframe: DataFrame) -> DataFrame:
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce").fillna(0.0)

        candle_range = (high - low).replace(0, pd.NA)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0)
        signed_volume = body_pressure * volume
        typical_price = (high + low + close) / 3.0
        additions: dict[str, pd.Series] = {}

        volume_24h = volume.rolling(24, min_periods=12).sum()
        volume_7d_mean = volume_24h.rolling(168, min_periods=48).mean()
        volume_7d_std = volume_24h.rolling(168, min_periods=48).std()
        volume_24h_z = (volume_24h - volume_7d_mean) / volume_7d_std.replace(0, pd.NA)
        additions["%-tech_volume_24h_z_7d"] = volume_24h_z

        for window in (24, 72, 168):
            min_periods = max(12, window // 2)
            volume_sum = volume.rolling(window, min_periods=min_periods).sum()
            vwap = self._safe_divide((typical_price * volume).rolling(window, min_periods=min_periods).sum(), volume_sum)
            weighted_var = self._safe_divide(
                (((typical_price - vwap) ** 2) * volume).rolling(window, min_periods=min_periods).sum(),
                volume_sum,
            )
            vp_std = weighted_var.clip(lower=0.0) ** 0.5
            value_high = (vwap + vp_std).shift(1)
            value_low = (vwap - vp_std).shift(1)
            vwap_level = vwap.shift(1)
            value_width = (value_high - value_low).replace(0, pd.NA)

            additions[f"%-tech_vp_dist_poc_{window}h"] = self._safe_divide(close - vwap_level, close)
            additions[f"%-tech_vp_value_pos_{window}h"] = (close - value_low) / value_width
            additions[f"%-tech_vp_width_{window}h"] = self._safe_divide(value_width, close)
            additions[f"%-tech_vp_above_value_{window}h"] = (close > value_high).astype(float)
            additions[f"%-tech_vp_below_value_{window}h"] = (close < value_low).astype(float)
            additions[f"%-tech_vp_reclaim_{window}h"] = ((close > value_low) & (close.shift(1) <= value_low.shift(1))).astype(float)
            additions[f"%-tech_vp_reject_high_{window}h"] = ((high > value_high) & (close < value_high)).astype(float)
            additions[f"%-tech_vp_reject_low_{window}h"] = ((low < value_low) & (close > value_low)).astype(float)

            rolling_high = high.shift(1).rolling(window, min_periods=min_periods).max()
            rolling_low = low.shift(1).rolling(window, min_periods=min_periods).min()
            rolling_range = (rolling_high - rolling_low).replace(0, pd.NA)
            range_ratio = self._safe_divide(rolling_range, close)
            range_base = range_ratio.rolling(168, min_periods=48).mean()
            range_base_std = range_ratio.rolling(168, min_periods=48).std()
            compression_z = (range_ratio - range_base) / range_base_std.replace(0, pd.NA)
            additions[f"%-tech_range_pos_{window}h"] = (close - rolling_low) / rolling_range
            additions[f"%-tech_compression_z_{window}h"] = compression_z
            additions[f"%-tech_compressed_{window}h"] = (compression_z < -0.75).astype(float)
            additions[f"%-tech_range_expanding_{window}h"] = (
                range_ratio > range_ratio.shift(6) * 1.15
            ).astype(float)
            additions[f"%-tech_trendline_res_slope_{window}h"] = self._safe_divide(
                rolling_high.diff(window) / float(window),
                close,
            )
            additions[f"%-tech_trendline_sup_slope_{window}h"] = self._safe_divide(
                rolling_low.diff(window) / float(window),
                close,
            )
            additions[f"%-tech_resistance_break_{window}h"] = (close > rolling_high).astype(float)
            additions[f"%-tech_support_break_{window}h"] = (close < rolling_low).astype(float)
            additions[f"%-tech_resistance_reject_{window}h"] = ((high > rolling_high) & (close <= rolling_high)).astype(float)
            additions[f"%-tech_support_reclaim_{window}h"] = ((low < rolling_low) & (close >= rolling_low)).astype(float)

        bb_mid = close.rolling(24, min_periods=12).mean()
        bb_std = close.rolling(24, min_periods=12).std()
        bb_upper = (bb_mid + (2.0 * bb_std)).shift(1)
        bb_lower = (bb_mid - (2.0 * bb_std)).shift(1)
        bb_width = self._safe_divide(bb_upper - bb_lower, close)
        bb_width_mean = bb_width.rolling(168, min_periods=48).mean()
        pressure_24h = self._safe_divide(signed_volume.rolling(24, min_periods=12).sum(), volume_24h)
        range_24h = (
            high.shift(1).rolling(24, min_periods=12).max()
            - low.shift(1).rolling(24, min_periods=12).min()
        ).replace(0, pd.NA)

        additions["%-tech_bb_width_24h"] = bb_width
        additions["%-tech_bb_pos_24h"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, pd.NA)
        additions["%-tech_bb_squeeze_24h"] = (bb_width < bb_width_mean * 0.75).astype(float)
        additions["%-tech_bb_upper_break_24h"] = (close > bb_upper).astype(float)
        additions["%-tech_bb_lower_break_24h"] = (close < bb_lower).astype(float)
        additions["%-tech_pressure_24h"] = pressure_24h
        additions["%-tech_pressure_expansion_up_24h"] = (
            (pressure_24h > 0.15) & (volume_24h_z > 0.75) & (close > close.shift(6))
        ).astype(float)
        additions["%-tech_pressure_expansion_down_24h"] = (
            (pressure_24h < -0.15) & (volume_24h_z > 0.75) & (close < close.shift(6))
        ).astype(float)
        additions["%-tech_breakout_after_compression_24h"] = (
            (additions["%-tech_compressed_72h"].shift(1) > 0.0)
            & (additions["%-tech_resistance_break_24h"] > 0.0)
            & (pressure_24h > 0.0)
        ).astype(float)
        additions["%-tech_breakdown_after_compression_24h"] = (
            (additions["%-tech_compressed_72h"].shift(1) > 0.0)
            & (additions["%-tech_support_break_24h"] > 0.0)
            & (pressure_24h < 0.0)
        ).astype(float)
        additions["%-tech_candle_range_expansion_24h"] = self._safe_divide(candle_range, range_24h)

        return pd.concat([dataframe, DataFrame(additions, index=dataframe.index)], axis=1)

    @staticmethod
    def _cross_above(left: pd.Series, right: pd.Series) -> pd.Series:
        return (left.gt(right) & left.shift(1).le(right.shift(1))).astype(float)

    @staticmethod
    def _cross_below(left: pd.Series, right: pd.Series) -> pd.Series:
        return (left.lt(right) & left.shift(1).ge(right.shift(1))).astype(float)

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
        mean = series.rolling(window, min_periods=min_periods).mean()
        std = series.rolling(window, min_periods=min_periods).std()
        return (series - mean) / std.replace(0, pd.NA)

    @staticmethod
    def _add_ta_value(raw: dict[str, pd.Series], name: str, fn) -> None:
        try:
            value = fn()
        except Exception:
            return
        if isinstance(value, DataFrame):
            for column in value.columns:
                raw[f"{name}_{str(column).lower()}"] = pd.to_numeric(value[column], errors="coerce")
        else:
            raw[name] = pd.to_numeric(value, errors="coerce")

    def _append_ta_confluence_features(self, dataframe: DataFrame) -> DataFrame:
        profile = str(self.ta_confluence_profile)
        if profile == "none":
            return dataframe

        inputs = dataframe[["open", "high", "low", "close", "volume"]].copy()
        for column in inputs.columns:
            inputs[column] = pd.to_numeric(inputs[column], errors="coerce")
        open_ = inputs["open"]
        high = inputs["high"]
        low = inputs["low"]
        close = inputs["close"]
        volume = inputs["volume"].fillna(0.0)
        candle_range = (high - low).replace(0, pd.NA)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0)

        raw: dict[str, pd.Series] = {}
        for period in self.ta_indicator_periods:
            self._add_ta_value(raw, f"rsi_{period}", lambda p=period: ta.RSI(inputs, timeperiod=p))
            self._add_ta_value(raw, f"ema_{period}", lambda p=period: ta.EMA(inputs, timeperiod=p))
            self._add_ta_value(raw, f"sma_{period}", lambda p=period: ta.SMA(inputs, timeperiod=p))
            self._add_ta_value(raw, f"tema_{period}", lambda p=period: ta.TEMA(inputs, timeperiod=p))
            self._add_ta_value(raw, f"roc_{period}", lambda p=period: ta.ROC(inputs, timeperiod=p))
        for name, fn in {
            "adx_14": lambda: ta.ADX(inputs, timeperiod=14),
            "plus_di_14": lambda: ta.PLUS_DI(inputs, timeperiod=14),
            "minus_di_14": lambda: ta.MINUS_DI(inputs, timeperiod=14),
            "cci_20": lambda: ta.CCI(inputs, timeperiod=20),
            "mfi_14": lambda: ta.MFI(inputs, timeperiod=14),
            "willr_14": lambda: ta.WILLR(inputs, timeperiod=14),
            "obv": lambda: ta.OBV(inputs),
            "adosc": lambda: ta.ADOSC(inputs),
            "trix_30": lambda: ta.TRIX(inputs, timeperiod=30),
            "ultosc": lambda: ta.ULTOSC(inputs),
            "cmo_14": lambda: ta.CMO(inputs, timeperiod=14),
            "natr_14": lambda: ta.NATR(inputs, timeperiod=14),
            "ppo": lambda: ta.PPO(inputs),
            "macd": lambda: ta.MACD(inputs),
            "stoch": lambda: ta.STOCH(inputs),
            "bbands": lambda: ta.BBANDS(inputs, timeperiod=20),
        }.items():
            self._add_ta_value(raw, name, fn)

        grouped: dict[str, dict[str, pd.Series]] = {
            "ma": {},
            "macd": {},
            "oscillator": {},
            "bollinger": {},
            "volume": {},
            "candle": {},
            "trend": {},
            "range": {},
            "vp": {},
            "confluence": {},
        }

        for name, series in raw.items():
            if name.startswith(("ema_", "sma_", "tema_")):
                grouped["ma"][name] = series
            elif name.startswith(("macd_", "ppo", "trix_", "roc_")):
                grouped["macd"][name] = series
            elif name.startswith(("rsi_", "cci_", "mfi_", "willr_", "stoch_", "ultosc", "cmo_")):
                grouped["oscillator"][name] = series
            elif name.startswith("bbands_"):
                grouped["bollinger"][name] = series
            elif name in {"obv", "adosc"}:
                grouped["volume"][name] = series
            elif name.startswith(("adx_", "plus_di_", "minus_di_")):
                grouped["trend"][name] = series

        for fast, slow in ((8, 21), (10, 30), (20, 50), (50, 200)):
            ema_fast = raw.get(f"ema_{fast}")
            ema_slow = raw.get(f"ema_{slow}")
            sma_slow = raw.get(f"sma_{slow}")
            if ema_fast is not None and ema_slow is not None:
                grouped["ma"][f"ema_{fast}_above_ema_{slow}"] = ema_fast.gt(ema_slow).astype(float)
                grouped["ma"][f"ema_{fast}_cross_above_ema_{slow}"] = self._cross_above(ema_fast, ema_slow)
                grouped["ma"][f"ema_{fast}_cross_below_ema_{slow}"] = self._cross_below(ema_fast, ema_slow)
            if ema_fast is not None and sma_slow is not None:
                grouped["ma"][f"ema_{fast}_above_sma_{slow}"] = ema_fast.gt(sma_slow).astype(float)

        macd_line = raw.get("macd_macd")
        macd_signal = raw.get("macd_macdsignal")
        macd_hist = raw.get("macd_macdhist")
        if macd_line is not None and macd_signal is not None:
            grouped["macd"]["macd_above_signal"] = macd_line.gt(macd_signal).astype(float)
            grouped["macd"]["macd_cross_above_signal"] = self._cross_above(macd_line, macd_signal)
            grouped["macd"]["macd_cross_below_signal"] = self._cross_below(macd_line, macd_signal)
        if macd_hist is not None:
            grouped["macd"]["macd_hist_rising_3bar"] = macd_hist.diff().gt(0).rolling(3, min_periods=3).sum().eq(3).astype(float)
            grouped["macd"]["macd_hist_falling_3bar"] = macd_hist.diff().lt(0).rolling(3, min_periods=3).sum().eq(3).astype(float)

        rsi_14 = raw.get("rsi_14")
        if rsi_14 is not None:
            grouped["oscillator"]["rsi_14_oversold_reclaim"] = (rsi_14.shift(1).lt(30) & rsi_14.ge(30)).astype(float)
            grouped["oscillator"]["rsi_14_overbought_loss"] = (rsi_14.shift(1).gt(70) & rsi_14.le(70)).astype(float)
            grouped["oscillator"]["rsi_14_rising_3bar"] = rsi_14.diff().gt(0).rolling(3, min_periods=3).sum().eq(3).astype(float)
            grouped["oscillator"]["rsi_14_falling_3bar"] = rsi_14.diff().lt(0).rolling(3, min_periods=3).sum().eq(3).astype(float)

        bb_upper = raw.get("bbands_upperband")
        bb_middle = raw.get("bbands_middleband")
        bb_lower = raw.get("bbands_lowerband")
        if bb_upper is not None and bb_middle is not None and bb_lower is not None:
            bb_width = self._safe_divide(bb_upper - bb_lower, bb_middle)
            grouped["bollinger"]["bb_width"] = bb_width
            grouped["bollinger"]["bb_squeeze_120"] = bb_width.le(bb_width.rolling(120, min_periods=40).quantile(0.20)).astype(float)
            grouped["bollinger"]["close_cross_above_bb_upper"] = self._cross_above(close, bb_upper)
            grouped["bollinger"]["close_cross_below_bb_lower"] = self._cross_below(close, bb_lower)
            grouped["bollinger"]["bb_mid_reclaim"] = self._cross_above(close, bb_middle)
            grouped["bollinger"]["bb_mid_loss"] = self._cross_below(close, bb_middle)

        volume_24 = volume.rolling(24, min_periods=12).sum()
        signed_volume = body_pressure * volume
        grouped["volume"]["body_pressure"] = body_pressure
        grouped["volume"]["volume_z_24"] = self._rolling_zscore(volume, 24, 12)
        grouped["volume"]["volume_z_168"] = self._rolling_zscore(volume, 168, 48)
        grouped["volume"]["signed_volume_pressure_24"] = self._safe_divide(
            signed_volume.rolling(24, min_periods=12).sum(),
            volume_24,
        )
        grouped["volume"]["obv_rising_12bar"] = raw.get("obv", pd.Series(index=dataframe.index, dtype=float)).diff().gt(0).rolling(12, min_periods=6).mean()
        grouped["volume"]["adosc_rising_6bar"] = raw.get("adosc", pd.Series(index=dataframe.index, dtype=float)).diff().gt(0).rolling(6, min_periods=3).mean()

        grouped["trend"]["adx_14_high"] = raw.get("adx_14", pd.Series(index=dataframe.index, dtype=float)).gt(25).astype(float)
        grouped["trend"]["plus_di_above_minus_di"] = raw.get("plus_di_14", pd.Series(index=dataframe.index, dtype=float)).gt(
            raw.get("minus_di_14", pd.Series(index=dataframe.index, dtype=float))
        ).astype(float)

        prior_high_24 = high.shift(1).rolling(24, min_periods=12).max()
        prior_low_24 = low.shift(1).rolling(24, min_periods=12).min()
        prior_high_72 = high.shift(1).rolling(72, min_periods=36).max()
        prior_low_72 = low.shift(1).rolling(72, min_periods=36).min()
        range_24 = (prior_high_24 - prior_low_24).replace(0, pd.NA)
        range_72 = (prior_high_72 - prior_low_72).replace(0, pd.NA)
        range_ratio_72 = self._safe_divide(range_72, close)
        grouped["range"]["range_pos_24"] = (close - prior_low_24) / range_24
        grouped["range"]["range_pos_72"] = (close - prior_low_72) / range_72
        grouped["range"]["range_compression_72"] = self._rolling_zscore(range_ratio_72, 168, 48).lt(-0.75).astype(float)
        grouped["range"]["resistance_break_24"] = close.gt(prior_high_24).astype(float)
        grouped["range"]["support_break_24"] = close.lt(prior_low_24).astype(float)
        grouped["range"]["higher_high_3bar"] = high.gt(high.shift(1)).rolling(3, min_periods=3).sum().eq(3).astype(float)
        grouped["range"]["lower_low_3bar"] = low.lt(low.shift(1)).rolling(3, min_periods=3).sum().eq(3).astype(float)

        bullish_count = pd.Series(0.0, index=dataframe.index)
        bearish_count = pd.Series(0.0, index=dataframe.index)
        for pattern in talib.get_function_groups().get("Pattern Recognition", []):
            try:
                values = getattr(ta, pattern)(inputs)
            except Exception:
                continue
            signed = pd.to_numeric(values, errors="coerce").fillna(0.0) / 100.0
            name = pattern.lower().replace("cdl", "candle_")
            grouped["candle"][name] = signed
            bullish_count = bullish_count + signed.gt(0).astype(float)
            bearish_count = bearish_count + signed.lt(0).astype(float)
        grouped["candle"]["bullish_candle_pattern_count"] = bullish_count
        grouped["candle"]["bearish_candle_pattern_count"] = bearish_count
        grouped["candle"]["bullish_candle_patterns_3bar"] = bullish_count.rolling(3, min_periods=1).sum()
        grouped["candle"]["bearish_candle_patterns_3bar"] = bearish_count.rolling(3, min_periods=1).sum()

        ema_20 = raw.get("ema_20")
        ema_50 = raw.get("ema_50")
        ema_200 = raw.get("ema_200")
        trend_up = ema_20.gt(ema_50).astype(float) if ema_20 is not None and ema_50 is not None else pd.Series(0.0, index=dataframe.index)
        trend_down = ema_20.lt(ema_50).astype(float) if ema_20 is not None and ema_50 is not None else pd.Series(0.0, index=dataframe.index)
        long_ma_regime = (
            ema_50.gt(ema_200).astype(float)
            if ema_50 is not None and ema_200 is not None
            else pd.Series(0.0, index=dataframe.index)
        )
        squeeze = grouped["bollinger"].get("bb_squeeze_120", pd.Series(0.0, index=dataframe.index))
        pressure = grouped["volume"].get("signed_volume_pressure_24", pd.Series(0.0, index=dataframe.index))
        vol_z = grouped["volume"].get("volume_z_24", pd.Series(0.0, index=dataframe.index))
        macd_up = grouped["macd"].get("macd_hist_rising_3bar", pd.Series(0.0, index=dataframe.index))
        macd_down = grouped["macd"].get("macd_hist_falling_3bar", pd.Series(0.0, index=dataframe.index))
        grouped["confluence"]["squeeze_breakout_macd_volume"] = (
            squeeze.gt(0) & grouped["range"]["resistance_break_24"].gt(0) & macd_up.gt(0) & vol_z.gt(0.5)
        ).astype(float)
        grouped["confluence"]["squeeze_breakdown_macd_volume"] = (
            squeeze.gt(0) & grouped["range"]["support_break_24"].gt(0) & macd_down.gt(0) & vol_z.gt(0.5)
        ).astype(float)
        grouped["confluence"]["trend_pullback_reclaim"] = (
            trend_up.gt(0) & long_ma_regime.gt(0) & grouped["bollinger"].get("bb_mid_reclaim", pd.Series(0.0, index=dataframe.index)).gt(0)
        ).astype(float)
        grouped["confluence"]["trend_pullback_reject"] = (
            trend_down.gt(0) & grouped["bollinger"].get("bb_mid_loss", pd.Series(0.0, index=dataframe.index)).gt(0)
        ).astype(float)
        grouped["confluence"]["rsi_reclaim_macd_volume"] = (
            grouped["oscillator"].get("rsi_14_oversold_reclaim", pd.Series(0.0, index=dataframe.index)).gt(0)
            & macd_up.gt(0)
            & pressure.gt(0)
        ).astype(float)
        grouped["confluence"]["rsi_loss_macd_volume"] = (
            grouped["oscillator"].get("rsi_14_overbought_loss", pd.Series(0.0, index=dataframe.index)).gt(0)
            & macd_down.gt(0)
            & pressure.lt(0)
        ).astype(float)
        grouped["confluence"]["bull_candles_plus_higher_high"] = (
            grouped["candle"]["bullish_candle_patterns_3bar"].ge(3) & grouped["range"]["higher_high_3bar"].gt(0)
        ).astype(float)
        grouped["confluence"]["bear_candles_plus_lower_low"] = (
            grouped["candle"]["bearish_candle_patterns_3bar"].ge(3) & grouped["range"]["lower_low_3bar"].gt(0)
        ).astype(float)

        if self.include_technical_context_features:
            vp_above = pd.to_numeric(dataframe.get("%-tech_vp_above_value_24h", 0.0), errors="coerce").fillna(0.0)
            vp_below = pd.to_numeric(dataframe.get("%-tech_vp_below_value_24h", 0.0), errors="coerce").fillna(0.0)
            vp_reject_high = pd.to_numeric(dataframe.get("%-tech_vp_reject_high_24h", 0.0), errors="coerce").fillna(0.0)
            vp_reject_low = pd.to_numeric(dataframe.get("%-tech_vp_reject_low_24h", 0.0), errors="coerce").fillna(0.0)
            grouped["vp"]["vp_above_value_macd_volume"] = (vp_above.gt(0) & macd_up.gt(0) & pressure.gt(0)).astype(float)
            grouped["vp"]["vp_below_value_macd_volume"] = (vp_below.gt(0) & macd_down.gt(0) & pressure.lt(0)).astype(float)
            grouped["vp"]["vp_high_reject_oscillator_loss"] = (
                vp_reject_high.gt(0) & grouped["oscillator"].get("rsi_14_overbought_loss", pd.Series(0.0, index=dataframe.index)).gt(0)
            ).astype(float)
            grouped["vp"]["vp_low_reclaim_oscillator_recovery"] = (
                vp_reject_low.gt(0) & grouped["oscillator"].get("rsi_14_oversold_reclaim", pd.Series(0.0, index=dataframe.index)).gt(0)
            ).astype(float)

        profile_groups = {
            "macd_bb_volume": {"macd", "bollinger", "volume", "confluence"},
            "macd_ma_trend": {"macd", "ma", "trend"},
            "rsi_macd_recovery": {"oscillator", "macd", "volume", "confluence"},
            "bollinger_squeeze_break": {"bollinger", "range", "volume", "confluence"},
            "ma_stack_trend": {"ma", "trend", "range"},
            "oscillator_reversion": {"oscillator", "bollinger", "range"},
            "candle_momentum": {"candle", "range", "volume", "confluence"},
            "candle_oscillator": {"candle", "oscillator", "confluence"},
            "adx_trend_pressure": {"trend", "ma", "volume"},
            "volume_price_pressure": {"volume", "range", "macd"},
            "range_compression_expansion": {"range", "bollinger", "volume", "confluence"},
            "vp_bb_confluence": {"vp", "bollinger", "volume", "confluence"},
            "vp_ma_confluence": {"vp", "ma", "trend", "volume"},
            "vp_oscillator_rejection": {"vp", "oscillator", "candle"},
            "mtf_trend_agreement": {"ma", "trend", "range"},
            "mtf_momentum_agreement": {"macd", "oscillator", "volume"},
            "mtf_daily_regime": {"ma", "trend", "bollinger"},
            "breakout_continuation": {"range", "bollinger", "macd", "volume", "confluence"},
            "breakdown_continuation": {"range", "bollinger", "macd", "volume", "confluence"},
            "all_ta_balanced": set(grouped),
        }
        selected_groups = profile_groups.get(profile)
        if selected_groups is None:
            raise ValueError(f"Unknown TA confluence profile: {profile}")

        additions: dict[str, pd.Series] = {}
        for group_name in selected_groups:
            for name, series in grouped.get(group_name, {}).items():
                additions[f"%-ta_{group_name}_{name}"] = pd.to_numeric(series, errors="coerce").astype("float32")
        return pd.concat([dataframe, DataFrame(additions, index=dataframe.index)], axis=1)

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)
        if self._include_context_features() and "%-ctx_feature_present" in dataframe:
            present = pd.to_numeric(dataframe["%-ctx_feature_present"], errors="coerce").fillna(0.0) >= 1.0
            context_columns = [
                column
                for column in dataframe.columns
                if column.startswith("%-ctx_")
                and column not in {
                    "%-ctx_feature_present",
                    "%-ctx_feature_missing",
                    "%-ctx_feature_age_hours",
                }
            ]
            if context_columns:
                dataframe.loc[~present, context_columns] = pd.NA
            dataframe = dataframe.drop(
                columns=[
                    column
                    for column in ("%-ctx_feature_missing", "%-ctx_feature_age_hours")
                    if column in dataframe
                ],
                errors="ignore",
            )
        if self.trader_state_profile != "none":
            dataframe = self._append_trader_state_context_features(dataframe)
        return dataframe

    def _append_trader_state_context_features(self, dataframe: DataFrame) -> DataFrame:
        profile = str(self.trader_state_profile)
        context_present = pd.to_numeric(dataframe.get("%-ctx_feature_present", 0.0), errors="coerce").fillna(0.0) >= 1.0

        def num(column: str) -> pd.Series:
            if column not in dataframe:
                return pd.Series(0.0, index=dataframe.index)
            return pd.to_numeric(dataframe[column], errors="coerce")

        attention = (
            num("%-ctx_gdelt_event_count_z_7d")
            + num("%-ctx_gdelt_num_mentions_sum_z_7d")
            + num("%-ctx_gdelt_num_articles_sum_z_7d")
        ) / 3.0
        conflict = (
            num("%-ctx_gdelt_conflict_event_count_z_7d")
            + num("%-ctx_gdelt_coercion_event_count_z_7d")
            + num("%-ctx_gdelt_protest_event_count_z_7d")
            + num("%-ctx_gdelt_sanctions_trade_url_count_z_7d")
        ) / 4.0
        market_topic = (
            num("%-ctx_gdelt_crypto_url_count_z_7d")
            + num("%-ctx_gdelt_macro_url_count_z_7d")
            + num("%-ctx_gdelt_banking_credit_url_count_z_7d")
            + num("%-ctx_gdelt_oil_energy_url_count_z_7d")
        ) / 4.0
        tone = num("%-ctx_gdelt_avg_tone_weighted_z_7d")
        goldstein = num("%-ctx_gdelt_goldstein_weighted_z_7d")

        compressed = num("%-tech_compressed_72h") > 0.0
        breakout = num("%-tech_resistance_break_24h") > 0.0
        breakdown = num("%-tech_support_break_24h") > 0.0
        pressure_up = num("%-tech_pressure_expansion_up_24h") > 0.0
        pressure_down = num("%-tech_pressure_expansion_down_24h") > 0.0
        upper_break = num("%-tech_bb_upper_break_24h") > 0.0
        lower_break = num("%-tech_bb_lower_break_24h") > 0.0
        vp_above = num("%-tech_vp_above_value_24h") > 0.0
        vp_below = num("%-tech_vp_below_value_24h") > 0.0

        quiet_news = context_present & attention.lt(-0.35) & conflict.lt(0.25)
        risk_rising = context_present & conflict.gt(0.75)
        attention_spike = context_present & attention.gt(1.0)
        tone_worse = context_present & tone.lt(-0.5)
        topic_spike = context_present & market_topic.gt(0.75)

        additions = {
            "%-state_quiet_technical_breakout": (quiet_news & compressed & breakout & pressure_up).astype(float),
            "%-state_quiet_technical_breakdown": (quiet_news & compressed & breakdown & pressure_down).astype(float),
            "%-state_bull_breakout_risk_disagreement": ((breakout | upper_break | vp_above) & risk_rising).astype(float),
            "%-state_bear_breakdown_risk_agreement": ((breakdown | lower_break | vp_below) & (risk_rising | tone_worse)).astype(float),
            "%-state_downside_pressure_news_shock": (pressure_down & (risk_rising | topic_spike)).astype(float),
            "%-state_upside_pressure_market_topic": (pressure_up & topic_spike & ~risk_rising).astype(float),
            "%-state_extended_attention_spike": ((upper_break | lower_break | vp_above | vp_below) & attention_spike).astype(float),
            "%-state_gdelt_attention_x_pressure": attention.where(context_present) * num("%-tech_pressure_24h"),
            "%-state_gdelt_conflict_x_breakdown": conflict.where(context_present) * (breakdown | pressure_down).astype(float),
            "%-state_gdelt_topic_x_breakout": market_topic.where(context_present) * (breakout | pressure_up).astype(float),
            "%-state_gdelt_tone_x_trend": tone.where(context_present) * num("%-tech_pressure_24h"),
            "%-state_gdelt_goldstein_x_range_pos": goldstein.where(context_present) * num("%-tech_range_pos_24h"),
        }
        if profile == "quiet_breakout":
            keep = {key: value for key, value in additions.items() if "quiet" in key or "topic_x_breakout" in key}
        elif profile == "risk_disagreement":
            keep = {key: value for key, value in additions.items() if "risk" in key or "conflict" in key or "tone" in key}
        elif profile == "news_exhaustion":
            keep = {key: value for key, value in additions.items() if "attention" in key or "extended" in key}
        elif profile == "all_states":
            keep = additions
        else:
            raise ValueError(f"Unknown trader state profile: {profile}")
        return pd.concat([dataframe, DataFrame(keep, index=dataframe.index)], axis=1)

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe = super().set_freqai_targets(dataframe, metadata, **kwargs)
        trend_thresholds = {24: 0.03, 72: 0.05}
        for horizon, threshold in trend_thresholds.items():
            return_column = f"&-future_return_{horizon}h"
            if return_column not in dataframe:
                continue
            future_return = dataframe[return_column]
            dataframe[f"&-future_down_{horizon}h"] = self._binary_label(
                future_return,
                future_return < 0,
            )
            dataframe[f"&-trend_up_{horizon}h"] = self._binary_label(
                future_return,
                future_return >= threshold,
            )
            dataframe[f"&-trend_down_{horizon}h"] = self._binary_label(
                future_return,
                future_return <= -threshold,
            )
            upside_column = f"&-future_max_upside_{horizon}h"
            drawdown_column = f"&-future_max_drawdown_{horizon}h"
            if upside_column in dataframe and drawdown_column in dataframe:
                upside = dataframe[upside_column]
                drawdown = dataframe[drawdown_column]
                dataframe[f"&-long_exit_risk_{horizon}h"] = self._binary_label(
                    drawdown,
                    drawdown <= -float(self.reward_danger_threshold),
                )
                dataframe[f"&-short_exit_risk_{horizon}h"] = self._binary_label(
                    upside,
                    upside >= float(self.reward_danger_threshold),
                )
                dataframe[f"&-long_giveback_after_profit_{horizon}h"] = self._binary_label(
                    upside,
                    (upside >= float(self.reward_danger_threshold)) & (future_return <= 0.005),
                )
                dataframe[f"&-short_giveback_after_profit_{horizon}h"] = self._binary_label(
                    drawdown,
                    (drawdown <= -float(self.reward_danger_threshold)) & (future_return >= -0.005),
                )
                dataframe[f"&-long_hold_quality_{horizon}h"] = self._binary_label(
                    upside,
                    (upside >= threshold) & (drawdown > -0.025),
                )
                dataframe[f"&-short_hold_quality_{horizon}h"] = self._binary_label(
                    drawdown,
                    (drawdown <= -threshold) & (upside < 0.025),
                )
                ordered = self._ordered_target_stop_labels(dataframe, horizon)
                for column, values in ordered.items():
                    dataframe[f"&-{column}_{horizon}h"] = values
        return dataframe

    def _ordered_target_stop_labels(self, dataframe: DataFrame, horizon: int) -> dict[str, pd.Series]:
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        reward_threshold = float(self.reward_danger_threshold)
        danger_threshold = -float(self.reward_danger_threshold)
        valid_source = close.shift(-horizon)

        long_reward_first: pd.Series | None = None
        long_danger_first: pd.Series | None = None
        short_reward_first: pd.Series | None = None
        short_danger_first: pd.Series | None = None
        unresolved = pd.Series(True, index=dataframe.index)

        for step in range(1, horizon + 1):
            future_high = high.shift(-step)
            future_low = low.shift(-step)
            up_hit = (future_high / close - 1.0) >= reward_threshold
            down_hit = (future_low / close - 1.0) <= danger_threshold
            if long_reward_first is None:
                long_reward_first = pd.Series(False, index=dataframe.index)
                long_danger_first = pd.Series(False, index=dataframe.index)
                short_reward_first = pd.Series(False, index=dataframe.index)
                short_danger_first = pd.Series(False, index=dataframe.index)
            long_reward_first = long_reward_first | (unresolved & up_hit & ~down_hit)
            long_danger_first = long_danger_first | (unresolved & down_hit)
            short_reward_first = short_reward_first | (unresolved & down_hit & ~up_hit)
            short_danger_first = short_danger_first | (unresolved & up_hit)
            unresolved = unresolved & ~up_hit & ~down_hit

        assert long_reward_first is not None
        assert long_danger_first is not None
        assert short_reward_first is not None
        assert short_danger_first is not None
        return {
            "long_reward_before_danger": self._binary_label(valid_source, long_reward_first),
            "long_danger_before_reward": self._binary_label(valid_source, long_danger_first),
            "short_reward_before_danger": self._binary_label(valid_source, short_reward_first),
            "short_danger_before_reward": self._binary_label(valid_source, short_danger_first),
        }


class ThreeKiHistoricalNewsPriceOnlyFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "price_only"
    historical_news_sources: tuple[str, ...] = ()


class ThreeKiHistoricalPriceTechFreqAIStrategy(ThreeKiHistoricalNewsPriceOnlyFreqAIStrategy):
    include_technical_context_features = True


class ThreeKiHistoricalNewsGdeltOnlyFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "context_only"
    historical_news_sources = ("gdelt",)


class ThreeKiHistoricalNewsGkgOnlyFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "context_only"
    historical_news_sources = ("gkg",)


class ThreeKiHistoricalNewsOnlyFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "context_only"
    historical_news_sources = ("gdelt", "gkg")


class ThreeKiHistoricalPriceNewsFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "combined"
    historical_news_sources = ("gdelt", "gkg")


class ThreeKiHistoricalPriceGdeltFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "combined"
    historical_news_sources = ("gdelt",)


class ThreeKiHistoricalPriceTechGdeltFreqAIStrategy(ThreeKiHistoricalPriceGdeltFreqAIStrategy):
    include_technical_context_features = True


class ThreeKiHistoricalPriceGdeltAttentionFreqAIStrategy(ThreeKiHistoricalPriceGdeltFreqAIStrategy):
    gdelt_feature_set = "attention"


class ThreeKiHistoricalPriceTechGdeltAttentionFreqAIStrategy(ThreeKiHistoricalPriceTechGdeltFreqAIStrategy):
    gdelt_feature_set = "attention"


class ThreeKiHistoricalPriceGdeltToneImpactFreqAIStrategy(ThreeKiHistoricalPriceGdeltFreqAIStrategy):
    gdelt_feature_set = "tone_impact"


class ThreeKiHistoricalPriceTechGdeltToneImpactFreqAIStrategy(ThreeKiHistoricalPriceTechGdeltFreqAIStrategy):
    gdelt_feature_set = "tone_impact"


class ThreeKiHistoricalPriceGdeltConflictRiskFreqAIStrategy(ThreeKiHistoricalPriceGdeltFreqAIStrategy):
    gdelt_feature_set = "conflict_risk"


class ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy(ThreeKiHistoricalPriceTechGdeltFreqAIStrategy):
    gdelt_feature_set = "conflict_risk"


class ThreeKiHistoricalPriceGdeltMarketTopicsFreqAIStrategy(ThreeKiHistoricalPriceGdeltFreqAIStrategy):
    gdelt_feature_set = "market_topics"


class ThreeKiHistoricalPriceTechGdeltMarketTopicsFreqAIStrategy(ThreeKiHistoricalPriceTechGdeltFreqAIStrategy):
    gdelt_feature_set = "market_topics"


class ThreeKiHistoricalPriceTechGdeltAttentionPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltAttentionFreqAIStrategy
):
    gdelt_feature_style = "persistence"


class ThreeKiHistoricalPriceTechGdeltToneImpactShockFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltToneImpactFreqAIStrategy
):
    gdelt_feature_style = "shock"


class ThreeKiHistoricalPriceTechGdeltConflictRiskShockFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    gdelt_feature_style = "shock"


class ThreeKiHistoricalPriceTechGdeltConflictRiskPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    gdelt_feature_style = "persistence"


class ThreeKiHistoricalPriceTechGdeltMarketTopicsChangeFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltMarketTopicsFreqAIStrategy
):
    gdelt_feature_style = "change"


class ThreeKiHistoricalPriceTechGdeltAllShockFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltFreqAIStrategy
):
    gdelt_feature_style = "shock"


class ThreeKiHistoricalPriceTechGdeltAllPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltFreqAIStrategy
):
    gdelt_feature_style = "persistence"


class ThreeKiHistoricalPriceTechGdeltAllRiskDisagreementFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltFreqAIStrategy
):
    trader_state_profile = "risk_disagreement"


class ThreeKiHistoricalPriceTechGdeltAllQuietBreakoutFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltFreqAIStrategy
):
    trader_state_profile = "quiet_breakout"


class ThreeKiHistoricalPriceTechGdeltAllNewsExhaustionFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltFreqAIStrategy
):
    trader_state_profile = "news_exhaustion"


class ThreeKiHistoricalPriceTechGdeltConflictRiskStateFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    trader_state_profile = "risk_disagreement"


class ThreeKiHistoricalPriceTechGdeltMarketTopicsQuietStateFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltMarketTopicsFreqAIStrategy
):
    trader_state_profile = "quiet_breakout"


class ThreeKiHistoricalPriceTechGdeltConflictRiskThreshold2FreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    reward_danger_threshold = 0.02


class ThreeKiHistoricalPriceTechGdeltConflictRiskThreshold5FreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    reward_danger_threshold = 0.05


class ThreeKiHistoricalPriceGkgFreqAIStrategy(ThreeKiHistoricalNewsFreqAIBaseStrategy):
    feature_mode = "combined"
    historical_news_sources = ("gkg",)


class ThreeKiHistoricalPriceTechGkgFreqAIStrategy(ThreeKiHistoricalPriceGkgFreqAIStrategy):
    include_technical_context_features = True


class ThreeKiHistoricalPriceTechGkgPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGkgFreqAIStrategy
):
    gdelt_feature_style = "persistence"


class ThreeKiHistoricalPriceTechGkgShockPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGkgFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"


class ThreeKiHistoricalPriceTechGkgAttentionExhaustionFreqAIStrategy(
    ThreeKiHistoricalPriceTechGkgFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"
    trader_state_profile = "news_exhaustion"


class ThreeKiHistoricalPriceTechGdeltConflictRiskShortPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    gdelt_feature_style = "short_persistence"


class ThreeKiHistoricalPriceTechGdeltConflictRiskDeepPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    gdelt_feature_style = "deep_persistence"


class ThreeKiHistoricalPriceTechGdeltConflictRiskShockPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"


class ThreeKiHistoricalPriceTechGdeltConflictRiskConfluenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltConflictRiskFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"
    trader_state_profile = "all_states"


class ThreeKiHistoricalPriceTechGdeltMarketTopicsShockPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltMarketTopicsFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"


class ThreeKiHistoricalPriceTechGdeltAttentionShockPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltAttentionFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"


class ThreeKiHistoricalPriceTechGdeltAllShockPersistenceFreqAIStrategy(
    ThreeKiHistoricalPriceTechGdeltFreqAIStrategy
):
    gdelt_feature_style = "shock_persistence"


class ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy(ThreeKiHistoricalNewsPriceOnlyFreqAIStrategy):
    include_ta_confluence_features = True


class ThreeKiHistoricalTaMacdBbVolumeFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "macd_bb_volume"


class ThreeKiHistoricalTaMacdMaTrendFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "macd_ma_trend"


class ThreeKiHistoricalTaRsiMacdRecoveryFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "rsi_macd_recovery"


class ThreeKiHistoricalTaBollingerSqueezeBreakFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "bollinger_squeeze_break"


class ThreeKiHistoricalTaMaStackTrendFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "ma_stack_trend"


class ThreeKiHistoricalTaOscillatorReversionFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "oscillator_reversion"


class ThreeKiHistoricalTaCandleMomentumFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "candle_momentum"


class ThreeKiHistoricalTaCandleOscillatorFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "candle_oscillator"


class ThreeKiHistoricalTaAdxTrendPressureFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "adx_trend_pressure"


class ThreeKiHistoricalTaVolumePricePressureFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "volume_price_pressure"


class ThreeKiHistoricalTaRangeCompressionExpansionFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "range_compression_expansion"


class ThreeKiHistoricalTaVpBbConfluenceFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    include_technical_context_features = True
    ta_confluence_profile = "vp_bb_confluence"


class ThreeKiHistoricalTaVpMaConfluenceFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    include_technical_context_features = True
    ta_confluence_profile = "vp_ma_confluence"


class ThreeKiHistoricalTaVpOscillatorRejectionFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    include_technical_context_features = True
    ta_confluence_profile = "vp_oscillator_rejection"


class ThreeKiHistoricalTaMtfTrendAgreementFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "mtf_trend_agreement"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50)


class ThreeKiHistoricalTaMtfMomentumAgreementFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "mtf_momentum_agreement"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50)


class ThreeKiHistoricalTaMtfDailyRegimeFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "mtf_daily_regime"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50)


class ThreeKiHistoricalTaBreakoutContinuationFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "breakout_continuation"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50)


class ThreeKiHistoricalTaBreakdownContinuationFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "breakdown_continuation"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50)


class ThreeKiHistoricalTaAllBalancedFreqAIStrategy(ThreeKiHistoricalTaConfluenceBaseFreqAIStrategy):
    ta_confluence_profile = "all_ta_balanced"
    ta_indicator_periods = (5, 8, 10, 14, 20, 21, 30, 50)
