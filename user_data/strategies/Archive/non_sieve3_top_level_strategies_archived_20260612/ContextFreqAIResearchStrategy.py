from __future__ import annotations

from functools import reduce
from pathlib import Path
import sqlite3
import sys

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class ContextFreqAIResearchStrategy(IStrategy):
    """
    Research-only FreqAI strategy for numeric external context features.

    It reads the derived context_features_1h SQL table. It does not scrape,
    parse article text, call APIs, or alter any production strategy behavior.
    """

    minimal_roi = {"0": 0.0}
    stoploss = -0.99
    timeframe = "1h"
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    feature_mode = "combined"
    include_orderbook_context = False
    include_structure_context = False
    include_confluence_event_features = False
    include_confluence_event_targets = False
    include_trader_confluence_context = False
    trader_confluence_feature_profile = "delta_with_flags"
    require_orderbook_present_for_confluence_targets = False
    orderbook_use_compacted_features = True
    orderbook_compact_feature_profile = "full"
    orderbook_compact_feature_file = ""
    structure_feature_profile = "structure_vp"
    structure_feature_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\structural_cache\btc_structural_features_1h_latest.parquet"
    )
    context_feature_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\exports\context_features_1h_latest.parquet"
    )
    trader_confluence_feature_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\confluence_cache\trader_confluence_1h_latest.parquet"
    )
    context_max_age_hours = 48
    orderbook_bar_timeframe_seconds = 60
    orderbook_min_coverage_ratio = 0.50
    orderbook_primary_market_key = "binance_usdm_futures"
    orderbook_feature_columns = (
        "obctx_ready",
        "obctx_quality_score",
        "obctx_coverage_ratio",
        "obctx_gap_flag",
        "obctx_risk_score",
        "obctx_risk_state",
        "obctx_spread_bps",
        "obctx_score_long",
        "obctx_score_short",
        "obctx_score_abs",
        "obctx_state",
        "obctx_book_bias_score",
        "obctx_pressure_score",
        "obctx_pressure_state",
        "obctx_wall_score",
        "obctx_wall_state",
        "obctx_wall_support_score",
        "obctx_wall_resistance_score",
        "obctx_market_ready_ratio",
        "obctx_market_pressure_score",
        "obctx_market_agreement_score",
    )
    return_label_horizons = (6, 168)
    path_label_horizons = (72, 720)
    include_regime_label = False
    include_trader_event_labels = False

    _context_cache: dict[tuple[str, str], DataFrame] = {}
    _orderbook_compact_cache: dict[tuple[str, str, str, str], DataFrame] = {}
    _structure_cache: dict[tuple[str, str, str, str], DataFrame] = {}
    _trader_confluence_cache: dict[tuple[str, str, str], DataFrame] = {}

    @classmethod
    def _context_db_path(cls) -> Path:
        user_data_dir = Path(__file__).resolve().parents[1]
        return user_data_dir / "research_news_data" / "context_features" / "context_features.sqlite"

    @classmethod
    def _context_feature_file_path(cls) -> Path | None:
        if not cls.context_feature_file:
            return None
        return Path(cls.context_feature_file)

    @classmethod
    def _orderbook_db_path(cls) -> Path:
        user_data_dir = Path(__file__).resolve().parents[1]
        return user_data_dir / "orderbook_data" / "live" / "orderbook_events.sqlite"

    @classmethod
    def _orderbook_compact_feature_file_path(cls) -> Path | None:
        if not cls.orderbook_compact_feature_file:
            return None
        return Path(cls.orderbook_compact_feature_file)

    @classmethod
    def _structure_feature_file_path(cls) -> Path | None:
        if not cls.structure_feature_file:
            return None
        return Path(cls.structure_feature_file)

    @classmethod
    def _trader_confluence_feature_file_path(cls) -> Path | None:
        if not cls.trader_confluence_feature_file:
            return None
        return Path(cls.trader_confluence_feature_file)

    @staticmethod
    def _canonical_pair(pair: str | None) -> str:
        text = str(pair or "BTC/USDT").strip().upper()
        return text.split(":", 1)[0]

    @staticmethod
    def _add_orderbook_context_features(dataframe: DataFrame, **kwargs) -> DataFrame:
        indicator_dir = Path(__file__).resolve().parents[1] / "Indicators" / "work_in_progress"
        if str(indicator_dir) not in sys.path:
            sys.path.insert(0, str(indicator_dir))
        from external_orderbook_context_features import add_orderbook_context_features

        return add_orderbook_context_features(dataframe, **kwargs)

    @classmethod
    def _load_context_features(cls) -> DataFrame:
        feature_file = cls._context_feature_file_path()
        source_path = feature_file or cls._context_db_path()
        cache_key = (cls.__name__, str(source_path))
        cached = cls._context_cache.get(cache_key)
        if cached is not None:
            return cached
        if feature_file is not None:
            if not feature_file.exists():
                frame = DataFrame()
            else:
                frame = pd.read_parquet(feature_file)
        else:
            db_path = cls._context_db_path()
            if not db_path.exists():
                frame = DataFrame()
            else:
                with sqlite3.connect(str(db_path), timeout=10.0) as conn:
                    frame = pd.read_sql_query("SELECT * FROM context_features_1h ORDER BY date", conn)
        if frame.empty:
            empty = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._context_cache[cache_key] = empty
            return empty
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        skip_columns = {
            "date",
            "generated_at",
            "min_source_available_at",
            "schema_version",
            "feature_config_hash",
            "news_rows_24h",
            "web_rows_24h",
            "global_metrics_available",
            "missing_available_at_rows",
            "feature_history_hours",
        }
        feature_columns = [column for column in frame.columns if column not in skip_columns]
        cleaned = frame[["date", *feature_columns]].dropna(subset=["date"]).copy()
        for column in feature_columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned = cleaned.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        cls._context_cache[cache_key] = cleaned
        return cleaned

    @classmethod
    def _load_orderbook_compact_features(cls, canonical_pair: str) -> DataFrame:
        feature_file = cls._orderbook_compact_feature_file_path()
        cache_key = (
            cls.__name__,
            canonical_pair,
            str(feature_file or cls._orderbook_db_path()),
            str(cls.orderbook_compact_feature_profile),
        )
        cached = cls._orderbook_compact_cache.get(cache_key)
        if cached is not None:
            return cached
        if feature_file is not None:
            if not feature_file.exists():
                raise FileNotFoundError(feature_file)
            frame = pd.read_parquet(feature_file)
            if "canonical_pair" in frame.columns:
                frame = frame[
                    frame["canonical_pair"].astype(str).str.upper() == str(canonical_pair).upper()
                ].copy()
            cleaned = cls._clean_orderbook_compact_frame(frame)
            cls._orderbook_compact_cache[cache_key] = cleaned
            return cleaned
        db_path = cls._orderbook_db_path()
        if not db_path.exists():
            frame = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._orderbook_compact_cache[cache_key] = frame
            return frame
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'orderbook_features_1h'"
            ).fetchone()
            if not exists:
                frame = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
                cls._orderbook_compact_cache[cache_key] = frame
                return frame
            frame = pd.read_sql_query(
                "SELECT * FROM orderbook_features_1h WHERE canonical_pair = ? ORDER BY date",
                conn,
                params=(canonical_pair,),
            )
        if frame.empty:
            cleaned = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._orderbook_compact_cache[cache_key] = cleaned
            return cleaned
        cleaned = cls._clean_orderbook_compact_frame(frame)
        cls._orderbook_compact_cache[cache_key] = cleaned
        return cleaned

    @staticmethod
    def _clean_orderbook_compact_frame(frame: DataFrame) -> DataFrame:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        skip_columns = {
            "date",
            "canonical_pair",
            "market_key",
            "generated_at",
            "source_min_ts",
            "source_max_ts",
            "obts_schema_version",
            "feature_schema_version",
            "feature_config_hash",
        }
        feature_columns = [column for column in frame.columns if column not in skip_columns]
        cleaned = frame[["date", *feature_columns]].dropna(subset=["date"]).copy()
        for column in feature_columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned = cleaned.sort_values("date").reset_index(drop=True)
        return cleaned

    @classmethod
    def _load_structure_features(cls, canonical_pair: str) -> DataFrame:
        feature_file = cls._structure_feature_file_path()
        cache_key = (
            cls.__name__,
            canonical_pair,
            str(feature_file or ""),
            str(cls.structure_feature_profile),
        )
        cached = cls._structure_cache.get(cache_key)
        if cached is not None:
            return cached
        if feature_file is None:
            frame = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            cls._structure_cache[cache_key] = frame
            return frame
        if not feature_file.exists():
            raise FileNotFoundError(feature_file)
        frame = pd.read_parquet(feature_file)
        if "canonical_pair" in frame.columns:
            frame = frame[
                frame["canonical_pair"].astype(str).str.upper() == str(canonical_pair).upper()
            ].copy()
        cleaned = cls._clean_structure_frame(frame)
        cls._structure_cache[cache_key] = cleaned
        return cleaned

    @classmethod
    def _clean_structure_frame(cls, frame: DataFrame) -> DataFrame:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        skip_columns = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "canonical_pair",
            "generated_at",
            "schema_version",
            "feature_config_hash",
        }
        feature_columns = [
            column
            for column in cls._select_structure_columns(frame)
            if column in frame.columns and column not in skip_columns
        ]
        cleaned = frame[["date", *feature_columns]].dropna(subset=["date"]).copy()
        for column in feature_columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        return cleaned.sort_values("date").reset_index(drop=True)

    @classmethod
    def _load_trader_confluence_features(cls) -> DataFrame:
        feature_file = cls._trader_confluence_feature_file_path()
        if feature_file is None:
            return DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
        cache_key = (cls.__name__, str(feature_file), str(cls.trader_confluence_feature_profile))
        cached = cls._trader_confluence_cache.get(cache_key)
        if cached is not None:
            return cached
        if not feature_file.exists():
            raise FileNotFoundError(feature_file)
        frame = pd.read_parquet(feature_file)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        selected = cls._select_trader_confluence_columns(frame)
        cleaned = frame[["date", *selected]].dropna(subset=["date"]).copy()
        for column in selected:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned = cleaned.sort_values("date").reset_index(drop=True)
        cls._trader_confluence_cache[cache_key] = cleaned
        return cleaned

    @classmethod
    def _select_trader_confluence_columns(cls, frame: DataFrame) -> list[str]:
        exact = {
            "conf_price_compression_24h",
            "conf_price_range_expansion_24h",
            "conf_range_position_high_24h",
            "conf_range_position_low_24h",
            "conf_range_position_high_72h",
            "conf_range_position_low_72h",
            "conf_range_position_high_stack",
            "conf_range_position_low_stack",
            "conf_volume_bullish_impulse_short",
            "conf_volume_bearish_impulse_short",
            "conf_ob_support_persistence_24h",
            "conf_ob_resistance_persistence_24h",
            "conf_ob_support_persistence_weak",
            "conf_ob_resistance_persistence_weak",
            "conf_ob_wall_distance_stretched",
            "conf_ob_upside_vacuum",
            "conf_ob_downside_vacuum",
            "conf_ob_upside_vacuum_after_resistance_removed",
            "conf_ob_downside_vacuum_after_support_removed",
            "conf_ob_bullish_pressure_agreement",
            "conf_ob_bearish_pressure_agreement",
            "conf_ob_pressure_agreement_ratio",
            "conf_ob_venue_count_present",
            "context_present",
            "structure_present",
            "orderbook_present",
        }
        profile = str(cls.trader_confluence_feature_profile)
        selected = []
        for column in frame.columns:
            if column in exact:
                selected.append(column)
                continue
            if profile.startswith("epsilon") and column.startswith("conf_epsilon_"):
                if profile == "epsilon_components_only":
                    if "_cmp_" in column:
                        selected.append(column)
                    continue
                if profile == "epsilon_without_flags":
                    if column.endswith(("_setup", "_trigger", "_score", "_component_score", "_component_count")):
                        continue
                selected.append(column)
                continue
            if not column.startswith("conf_delta_"):
                continue
            if profile == "delta_components_only":
                if "_cmp_" in column:
                    selected.append(column)
                continue
            if profile == "delta_without_flags":
                if column.endswith(("_setup", "_trigger", "_score", "_component_score", "_component_count")):
                    continue
            selected.append(column)
        return selected

    @classmethod
    def _select_structure_columns(cls, frame: DataFrame) -> list[str]:
        if cls.structure_feature_profile == "full":
            return [column for column in frame.columns if column != "date"]
        if cls.structure_feature_profile != "structure_vp":
            raise ValueError(f"Unknown structure feature profile: {cls.structure_feature_profile}")
        exact = {
            "st_price_return_1h",
            "st_price_return_3h",
            "st_price_return_6h",
            "st_volume_pressure_6h",
            "st_volume_pressure_24h",
            "st_volume_z_24h",
            "st_range_position_24h",
            "st_range_position_72h",
            "st_breakout_structure_setup",
            "st_breakdown_structure_setup",
            "st_failed_breakout_structure_risk",
            "st_failed_breakdown_structure_risk",
            "st_near_tlv2_resistance_1h",
            "st_near_tlv2_support_1h",
        }
        selected = set(exact)
        for timeframe in ("1h", "4h", "1d"):
            for prefix in (
                f"st_{timeframe}_vp_",
                f"st_{timeframe}_tlv2_",
                f"st_{timeframe}_ms_",
            ):
                selected.update(column for column in frame.columns if column.startswith(prefix))
        raw_level_suffixes = (
            "_prior_poc",
            "_prior_vah",
            "_prior_val",
            "_poc",
            "_vah",
            "_val",
            "_line_rank0",
            "_last_swing_high",
            "_last_swing_low",
            "_break_level",
            "_invalidation_level",
            "_bullish_break_level",
            "_bearish_break_level",
        )
        return [
            column
            for column in frame.columns
            if column in selected and not column.endswith(raw_level_suffixes)
        ]

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        if not self._include_price_features():
            return dataframe
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        if not self._include_price_features():
            return dataframe
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-volume_change"] = dataframe["volume"].pct_change()
        volume_24h = dataframe["volume"].rolling(24, min_periods=12).sum()
        prior_volume_24h = volume_24h.shift(24)
        dataframe["%-volume_24h_change"] = volume_24h / prior_volume_24h.replace(0, pd.NA) - 1.0
        dataframe["%-volume_24h_rising"] = (volume_24h > prior_volume_24h).astype(float)
        volume_mean_7d = volume_24h.rolling(168, min_periods=48).mean()
        volume_std_7d = volume_24h.rolling(168, min_periods=48).std()
        dataframe["%-volume_24h_z_7d"] = (volume_24h - volume_mean_7d) / volume_std_7d.replace(0, pd.NA)

        candle_range = (dataframe["high"] - dataframe["low"]).replace(0, pd.NA)
        body_pressure = ((dataframe["close"] - dataframe["open"]) / candle_range).clip(-1.0, 1.0).fillna(0.0)
        signed_volume = body_pressure * dataframe["volume"]
        up_volume = dataframe["volume"].where(dataframe["close"] > dataframe["open"], 0.0)
        for window in (6, 24):
            min_periods = max(3, window // 2)
            volume_sum = dataframe["volume"].rolling(window, min_periods=min_periods).sum()
            dataframe[f"%-volume_pressure_{window}h"] = (
                signed_volume.rolling(window, min_periods=min_periods).sum() / volume_sum.replace(0, pd.NA)
            )
            dataframe[f"%-up_volume_share_{window}h"] = (
                up_volume.rolling(window, min_periods=min_periods).sum() / volume_sum.replace(0, pd.NA)
            )

        higher_high = (dataframe["high"] > dataframe["high"].shift(1)).astype(float)
        higher_low = (dataframe["low"] > dataframe["low"].shift(1)).astype(float)
        lower_high = (dataframe["high"] < dataframe["high"].shift(1)).astype(float)
        lower_low = (dataframe["low"] < dataframe["low"].shift(1)).astype(float)
        for window in (6, 24):
            min_periods = max(3, window // 2)
            dataframe[f"%-hh_count_{window}h"] = higher_high.rolling(window, min_periods=min_periods).sum()
            dataframe[f"%-hl_count_{window}h"] = higher_low.rolling(window, min_periods=min_periods).sum()
            dataframe[f"%-lh_count_{window}h"] = lower_high.rolling(window, min_periods=min_periods).sum()
            dataframe[f"%-ll_count_{window}h"] = lower_low.rolling(window, min_periods=min_periods).sum()
            dataframe[f"%-structure_pressure_{window}h"] = (
                (higher_high + higher_low - lower_high - lower_low).rolling(window, min_periods=min_periods).mean()
            )

        for window in (24, 72):
            min_periods = window // 2
            prior_high = dataframe["high"].shift(1).rolling(window, min_periods=min_periods).max()
            prior_low = dataframe["low"].shift(1).rolling(window, min_periods=min_periods).min()
            prior_range = (prior_high - prior_low).replace(0, pd.NA)
            dataframe[f"%-close_range_pos_{window}h"] = (dataframe["close"] - prior_low) / prior_range
            dataframe[f"%-close_breakout_{window}h"] = (dataframe["close"] > prior_high).astype(float)
            dataframe[f"%-close_breakdown_{window}h"] = (dataframe["close"] < prior_low).astype(float)
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        if self._include_price_features():
            dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
            dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        if self._include_context_features():
            context = self._load_context_features()
        else:
            context = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
        if not context.empty:
            left = dataframe[["date"]].copy()
            left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")
            left["_rowid"] = range(len(left))
            context_for_merge = context.copy()
            context_for_merge["_context_feature_date"] = context_for_merge["date"]
            if "max_source_available_at" in context_for_merge:
                context_for_merge["_context_source_available_at"] = pd.to_datetime(
                    context_for_merge["max_source_available_at"],
                    utc=True,
                    errors="coerce",
                )
            else:
                context_for_merge["_context_source_available_at"] = context_for_merge["date"]
            merged = pd.merge_asof(
                left.sort_values("date"),
                context_for_merge,
                on="date",
                direction="backward",
                tolerance=pd.Timedelta(hours=float(self.context_max_age_hours)),
            ).sort_values("_rowid")
            matched_date = pd.to_datetime(merged.get("_context_feature_date"), utc=True, errors="coerce")
            left_date = pd.to_datetime(merged["date"], utc=True, errors="coerce")
            source_available_at = pd.to_datetime(merged.get("_context_source_available_at"), utc=True, errors="coerce")
            context_age_hours = (left_date.reset_index(drop=True) - source_available_at.reset_index(drop=True)).dt.total_seconds() / 3600.0
            source_fresh = source_available_at.notna() & context_age_hours.ge(0.0) & context_age_hours.le(float(self.context_max_age_hours))
            context_present = (matched_date.notna() & source_fresh).astype(float)
            context_features = {}
            context_features["%-ctx_feature_present"] = context_present.to_numpy()
            context_features["%-ctx_feature_missing"] = (1.0 - context_present).to_numpy()
            context_features["%-ctx_feature_age_hours"] = context_age_hours.fillna(float(self.context_max_age_hours) + 1.0).to_numpy()
            for column in context.columns:
                if column in {"date", "max_source_available_at"}:
                    continue
                context_features[f"%-ctx_{column}"] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).to_numpy()
            if context_features:
                dataframe = pd.concat([dataframe, DataFrame(context_features, index=dataframe.index)], axis=1)
            if self._include_price_features():
                breakout = dataframe.get("%-close_breakout_24h")
                pressure = dataframe.get("%-volume_pressure_24h")
                rising_volume = dataframe.get("%-volume_24h_rising")
                for source_column, output_column in (
                    ("%-ctx_article_count_24h", "%-ctx_article_breakout_pressure_24h"),
                    ("%-ctx_gdelt_event_count_24h", "%-ctx_gdelt_breakout_pressure_24h"),
                ):
                    source = dataframe.get(source_column)
                    if source is not None and breakout is not None and pressure is not None:
                        dataframe[output_column] = source.fillna(0.0) * breakout.fillna(0.0) * pressure.fillna(0.0)
                    if source is not None and rising_volume is not None:
                        dataframe[f"{output_column}_volume_rising"] = source.fillna(0.0) * rising_volume.fillna(0.0)
        if self._include_orderbook_features():
            dataframe = self._append_orderbook_features(dataframe, metadata)
            dataframe = self._append_orderbook_price_interactions(dataframe)
        if self._include_structure_features():
            dataframe = self._append_structure_features(dataframe, metadata)
        if self.include_confluence_event_features:
            dataframe = self._append_structure_confluence_event_features(dataframe)
        if self.include_trader_confluence_context:
            dataframe = self._append_trader_confluence_features(dataframe)
        return dataframe

    def _include_price_features(self) -> bool:
        return self.feature_mode in {"combined", "price_only"}

    def _include_context_features(self) -> bool:
        return self.feature_mode in {"combined", "context_only"}

    def _include_orderbook_features(self) -> bool:
        return bool(self.include_orderbook_context)

    def _include_structure_features(self) -> bool:
        return bool(self.include_structure_context)

    def _append_trader_confluence_features(self, dataframe: DataFrame) -> DataFrame:
        features = self._load_trader_confluence_features()
        if features.empty:
            return dataframe
        left = dataframe[["date"]].copy()
        left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")
        merge_features = features.copy()
        merge_features["__tc_feature_present"] = 1.0
        merged = left.merge(merge_features, on="date", how="left")
        output = {}
        present = pd.to_numeric(merged["__tc_feature_present"], errors="coerce").fillna(0.0)
        output["%-tc_feature_present"] = present.to_numpy()
        output["%-tc_feature_missing"] = present.eq(0.0).astype(float).to_numpy()
        for column in features.columns:
            if column == "date":
                continue
            output[f"%-tc_{column}"] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).to_numpy()
        if output:
            dataframe = pd.concat([dataframe, DataFrame(output, index=dataframe.index)], axis=1)
        return dataframe

    def _append_orderbook_features(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.orderbook_use_compacted_features:
            canonical_pair = self._canonical_pair(metadata.get("pair"))
            compact = self._load_orderbook_compact_features(canonical_pair)
            if compact.empty:
                return dataframe
            compact = self._select_orderbook_compact_columns(compact)
            left = dataframe[["date"]].copy()
            left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")
            merged = left.merge(compact, on="date", how="left", sort=False)
            orderbook_present = self._orderbook_present_from_merge(merged)
            compact_features = {}
            dataframe["orderbook_present"] = orderbook_present.to_numpy()
            compact_features["%-orderbook_present"] = orderbook_present.to_numpy()
            compact_features["%-orderbook_missing"] = (1.0 - orderbook_present).to_numpy()
            for column in compact.columns:
                if column == "date":
                    continue
                compact_features[f"%-{column}"] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).to_numpy()
            if not compact_features:
                return dataframe
            return pd.concat([dataframe, DataFrame(compact_features, index=dataframe.index)], axis=1)
        with_context = self._add_orderbook_context_features(
            dataframe,
            pair=metadata.get("pair"),
            db_path=self._orderbook_db_path(),
            bar_timeframe_seconds=self.orderbook_bar_timeframe_seconds,
            primary_market_key=self.orderbook_primary_market_key,
            min_coverage_ratio=self.orderbook_min_coverage_ratio,
            include_market_context=False,
            include_diagnostics=False,
            allow_missing=True,
        )
        orderbook_features = {}
        if "raw_tick_rows" in with_context:
            base_present = pd.to_numeric(with_context["raw_tick_rows"], errors="coerce").fillna(0.0).gt(0.0)
        else:
            feature_presence_columns = [column for column in self.orderbook_feature_columns if column in with_context]
            base_present = with_context[feature_presence_columns].notna().any(axis=1) if feature_presence_columns else pd.Series(False, index=dataframe.index)
        if "obctx_coverage_ratio" in with_context:
            coverage_ok = pd.to_numeric(with_context["obctx_coverage_ratio"], errors="coerce").fillna(0.0).ge(float(self.orderbook_min_coverage_ratio))
        else:
            coverage_ok = pd.Series(True, index=dataframe.index)
        orderbook_present = (base_present & coverage_ok).astype(float)
        dataframe["orderbook_present"] = orderbook_present.to_numpy()
        orderbook_features["%-orderbook_present"] = orderbook_present.to_numpy()
        orderbook_features["%-orderbook_missing"] = (1.0 - orderbook_present).to_numpy()
        for column in self.orderbook_feature_columns:
            if column in with_context:
                output_column = f"%-{column}"
                orderbook_features[output_column] = pd.to_numeric(with_context[column], errors="coerce").fillna(0.0).to_numpy()
        if not orderbook_features:
            return dataframe
        return pd.concat([dataframe, DataFrame(orderbook_features, index=dataframe.index)], axis=1)

    def _orderbook_present_from_merge(self, merged: DataFrame) -> pd.Series:
        if "obts_feature_present" in merged:
            base_present = pd.to_numeric(merged["obts_feature_present"], errors="coerce").fillna(0.0).gt(0.0)
            if "obts_coverage_ratio" in merged:
                coverage_ok = pd.to_numeric(merged["obts_coverage_ratio"], errors="coerce").fillna(0.0).ge(float(self.orderbook_min_coverage_ratio))
                base_present = base_present & coverage_ok
            return base_present.astype(float)
        if "raw_tick_rows" in merged:
            base_present = pd.to_numeric(merged["raw_tick_rows"], errors="coerce").fillna(0.0).gt(0.0)
            if "coverage_ratio" in merged:
                coverage_ok = pd.to_numeric(merged["coverage_ratio"], errors="coerce").fillna(0.0).ge(float(self.orderbook_min_coverage_ratio))
                base_present = base_present & coverage_ok
            return base_present.astype(float)
        candidate_columns = [
            column
            for column in merged.columns
            if column != "date" and (column.startswith("obts_") or column.startswith("ob1h_"))
        ]
        if not candidate_columns:
            return pd.Series(0.0, index=merged.index)
        base_present = merged[candidate_columns].notna().any(axis=1)
        if "obts_coverage_ratio" in merged:
            coverage_ok = pd.to_numeric(merged["obts_coverage_ratio"], errors="coerce").fillna(0.0).ge(float(self.orderbook_min_coverage_ratio))
            base_present = base_present & coverage_ok
        return base_present.astype(float)

    def _append_structure_features(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        canonical_pair = self._canonical_pair(metadata.get("pair"))
        structure = self._load_structure_features(canonical_pair)
        if structure.empty:
            return dataframe
        left = dataframe[["date"]].copy()
        left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")
        merged = left.merge(structure, on="date", how="left", sort=False)
        structure_features = {}
        for column in structure.columns:
            if column == "date":
                continue
            structure_features[f"%-{column}"] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).to_numpy()
        if not structure_features:
            return dataframe
        return pd.concat([dataframe, DataFrame(structure_features, index=dataframe.index)], axis=1)

    def _select_orderbook_compact_columns(self, compact: DataFrame) -> DataFrame:
        if self.orderbook_compact_feature_profile == "full":
            return compact
        if self.orderbook_compact_feature_profile == "bybit_refined":
            selected = {
                "date",
                "raw_tick_rows",
                "ob1h_bybit_spot_tick_rows",
                "ob1h_bybit_spot_coverage_ratio",
                "ob1h_bybit_spot_valid_book_ratio",
                "ob1h_bybit_spot_ready",
                "ob1h_bybit_spot_nearest_ask_wall_distance_bps_min",
                "ob1h_bybit_spot_nearest_bid_wall_distance_bps_min",
                "ob1h_bybit_spot_spread_bps_last",
                "ob1h_bybit_spot_spread_bps_p95",
                "ob1h_bybit_spot_nearest_ask_wall_notional_p95",
                "ob1h_bybit_spot_nearest_bid_wall_notional_p95",
                "ob1h_bybit_spot_nearest_ask_wall_score_p95",
                "ob1h_bybit_spot_nearest_bid_wall_score_p95",
                "ob1h_bybit_spot_bid_liquidity_5bps_min",
                "ob1h_bybit_spot_ask_liquidity_5bps_min",
                "ob1h_bybit_spot_depth_thinness_score",
                "ob1h_bybit_spot_total_depth_top20_mean",
                "ob1h_bybit_spot_imbalance_top20_std",
                "ob1h_bybit_spot_pressure_flip_count",
            }
            refined_roots = (
                "ob1h_bybit_spot_nearest_ask_wall_distance_bps_min",
                "ob1h_bybit_spot_nearest_bid_wall_distance_bps_min",
                "ob1h_bybit_spot_spread_bps_last",
                "ob1h_bybit_spot_spread_bps_p95",
                "ob1h_bybit_spot_nearest_ask_wall_notional_p95",
                "ob1h_bybit_spot_nearest_bid_wall_notional_p95",
                "ob1h_bybit_spot_nearest_ask_wall_score_p95",
                "ob1h_bybit_spot_nearest_bid_wall_score_p95",
                "ob1h_bybit_spot_bid_liquidity_5bps_min",
                "ob1h_bybit_spot_ask_liquidity_5bps_min",
                "ob1h_bybit_spot_depth_thinness_score",
                "ob1h_bybit_spot_total_depth_top20_mean",
                "ob1h_bybit_spot_imbalance_top20_std",
                "ob1h_bybit_spot_pressure_flip_count",
            )
            for root in refined_roots:
                selected.update({f"{root}_delta_1h", f"{root}_delta_6h", f"{root}_z_24h", f"{root}_z_7d"})
            available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
            return compact.loc[:, available]
        if self.orderbook_compact_feature_profile == "bybit_behaviour":
            selected = {
                "date",
                "raw_tick_rows",
                "ob1h_bybit_spot_tick_rows",
                "ob1h_bybit_spot_coverage_ratio",
                "ob1h_bybit_spot_valid_book_ratio",
                "ob1h_bybit_spot_ready",
                "ob1h_bybit_spot_nearest_ask_wall_distance_bps_min",
                "ob1h_bybit_spot_nearest_bid_wall_distance_bps_min",
                "ob1h_bybit_spot_spread_bps_p95",
                "ob1h_bybit_spot_total_depth_top20_mean",
            }
            selected.update(
                column
                for column in compact.columns
                if column.startswith("ob1h_bybit_spot_beh_")
            )
            available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
            return compact.loc[:, available]
        if self.orderbook_compact_feature_profile in {"bybit_zone_compression", "bybit_rejection_state"}:
            selected = {
                "date",
                "raw_tick_rows",
                "ob1h_bybit_spot_tick_rows",
                "ob1h_bybit_spot_coverage_ratio",
                "ob1h_bybit_spot_valid_book_ratio",
                "ob1h_bybit_spot_ready",
                "ob1h_bybit_spot_nearest_ask_wall_distance_bps_min",
                "ob1h_bybit_spot_nearest_bid_wall_distance_bps_min",
                "ob1h_bybit_spot_spread_bps_p95",
                "ob1h_bybit_spot_total_depth_top20_mean",
                "ob1h_bybit_spot_beh_zone_compression_score",
                "ob1h_bybit_spot_beh_persistent_ask_zone_6h",
                "ob1h_bybit_spot_beh_persistent_bid_zone_6h",
                "ob1h_bybit_spot_beh_persistent_ask_zone_24h",
                "ob1h_bybit_spot_beh_persistent_bid_zone_24h",
                "ob1h_bybit_spot_beh_spread_shock_24h",
                "ob1h_bybit_spot_beh_resistance_removed_score",
                "ob1h_bybit_spot_beh_support_removed_score",
                "ob1h_bybit_spot_beh_bullish_impulse_score",
                "ob1h_bybit_spot_beh_bearish_impulse_score",
                "ob1h_bybit_spot_beh_post_breakout_support_rebuild",
                "ob1h_bybit_spot_beh_post_breakdown_resistance_rebuild",
            }
            available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
            return compact.loc[:, available]
        if self.orderbook_compact_feature_profile == "bybit_objective_alpha":
            selected = {
                "date",
                "raw_tick_rows",
                "ob1h_bybit_spot_tick_rows",
                "ob1h_bybit_spot_coverage_ratio",
                "ob1h_bybit_spot_valid_book_ratio",
                "ob1h_bybit_spot_ready",
            }
            selected.update(column for column in compact.columns if column.startswith("ob1h_alpha_"))
            selected.update(column for column in compact.columns if column.startswith("ob1h_bybit_spot_alpha_"))
            available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
            return compact.loc[:, available]
        if self.orderbook_compact_feature_profile == "bybit_trader_state":
            selected = {"date"}
            selected.update(
                column
                for column in compact.columns
                if column.startswith("obts_") and column not in {"obts_schema_version"}
            )
            available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
            return compact.loc[:, available]
        if self.orderbook_compact_feature_profile == "bybit_trader_refined":
            selected = {
                "date",
                "obts_feature_present",
                "obts_market_id",
                "obts_sample_rows_1m",
                "obts_coverage_ratio",
                "obts_zone_compression_score",
                "obts_resistance_zone_distance_bps",
                "obts_support_zone_distance_bps",
                "obts_resistance_zone_strength",
                "obts_support_zone_strength",
                "obts_resistance_zone_stability_score",
                "obts_support_zone_stability_score",
                "obts_resistance_zone_removed_strength_1h",
                "obts_support_zone_removed_strength_1h",
                "obts_resistance_zone_added_strength_1h",
                "obts_support_zone_added_strength_1h",
                "obts_pressure_direction",
                "obts_pressure_duration_hours",
                "obts_pressure_price_agreement",
                "obts_pressure_divergence",
                "obts_upside_liquidity_vacuum_near",
                "obts_downside_liquidity_vacuum_near",
                "obts_breakout_failure_score",
                "obts_breakdown_failure_score",
                "obts_ask_absorption_score",
                "obts_bid_absorption_score",
                "obts_pre_breakout_failure_risk_score",
                "obts_resistance_rejection_resolved_score",
                "obts_resistance_cleared_score",
                "obts_post_failure_chop_score",
                "obts_ask_absorption_before_reaction_score",
                "obts_pre_breakdown_failure_risk_score",
                "obts_support_bounce_resolved_score",
                "obts_support_cleared_score",
                "obts_post_breakdown_chop_score",
                "obts_bid_absorption_before_reaction_score",
                "obts_range_position_24h",
                "obts_range_position_72h",
                "obts_price_trend_state_24h",
                "obts_volatility_expansion_state",
            }
            available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
            return compact.loc[:, available]
        if self.orderbook_compact_feature_profile not in {"headline", "bybit_headline"}:
            raise ValueError(f"Unknown orderbook compact feature profile: {self.orderbook_compact_feature_profile}")
        markets = ("bybit_spot",) if self.orderbook_compact_feature_profile == "bybit_headline" else (
            "binance_spot",
            "binance_usdm_futures",
            "bybit_spot",
            "bybit_linear",
        )
        selected = {
            "date",
            "raw_tick_rows",
            "market_context_rows",
            "ob1h_x_market_ready_count",
            "ob1h_x_market_ready_ratio",
            "ob1h_x_coverage_mean",
            "ob1h_x_coverage_min",
            "ob1h_x_pressure_delta",
            "ob1h_x_pressure_delta_std",
            "ob1h_x_pressure_delta_agreement",
            "ob1h_x_imbalance_top20_mean",
            "ob1h_x_imbalance_top20_mean_std",
            "ob1h_x_imbalance_top20_mean_agreement",
            "ob1h_x_spread_bps_mean",
            "ob1h_x_spread_bps_mean_std",
            "ob1h_x_total_depth_top20_mean",
            "ob1h_x_wall_support_resistance_delta",
            "ob1h_x_confluence_long_score",
            "ob1h_x_confluence_short_score",
            "ob1h_x_confluence_delta",
        }
        cross_roots = (
            "ob1h_x_pressure_delta",
            "ob1h_x_imbalance_top20_mean",
            "ob1h_x_spread_bps_mean",
            "ob1h_x_total_depth_top20_mean",
            "ob1h_x_confluence_long_score",
            "ob1h_x_confluence_short_score",
        )
        market_suffixes = (
            "tick_rows",
            "coverage_ratio",
            "valid_book_ratio",
            "ready",
            "max_gap_seconds",
            "message_count_sum",
            "spread_bps_mean",
            "spread_bps_p95",
            "spread_bps_max",
            "microprice_offset_bps_mean",
            "imbalance_top20_mean",
            "imbalance_10bps_mean",
            "total_depth_top20_mean",
            "depth_top20_bid_share",
            "depth_thinness_score",
            "pressure_delta",
            "pressure_flip_count",
            "pressure_agreement_ratio",
            "wall_support_resistance_delta",
            "near_bid_wall_ratio_10bps",
            "near_ask_wall_ratio_10bps",
            "near_bid_wall_ratio_25bps",
            "near_ask_wall_ratio_25bps",
            "ctx_funding_rate_last",
            "ctx_open_interest_last",
            "ctx_open_interest_delta",
            "ctx_taker_pressure",
        )
        for root in cross_roots:
            selected.update({f"{root}_delta_1h", f"{root}_delta_6h", f"{root}_z_24h", f"{root}_z_7d"})
        for market in markets:
            safe_market = market.replace("-", "_")
            for suffix in market_suffixes:
                selected.add(f"ob1h_{safe_market}_{suffix}")
            for root in ("pressure_delta", "imbalance_top20_mean", "spread_bps_mean", "total_depth_top20_mean", "wall_support_resistance_delta"):
                base = f"ob1h_{safe_market}_{root}"
                selected.update({f"{base}_delta_1h", f"{base}_delta_6h", f"{base}_z_24h", f"{base}_z_7d"})
        available = ["date", *[column for column in compact.columns if column in selected and column != "date"]]
        return compact.loc[:, available]

    def _append_orderbook_price_interactions(self, dataframe: DataFrame) -> DataFrame:
        if self.orderbook_compact_feature_profile not in {"bybit_zone_compression", "bybit_rejection_state"}:
            return dataframe
        zone = dataframe.get("%-ob1h_bybit_spot_beh_zone_compression_score")
        if zone is None:
            return dataframe
        zone = pd.to_numeric(zone, errors="coerce").fillna(0.0)
        interactions = {}
        for source_column, output_suffix in (
            ("%-close_range_pos_24h", "range_pos_24h"),
            ("%-close_range_pos_72h", "range_pos_72h"),
            ("%-close_breakout_24h", "breakout_24h"),
            ("%-close_breakout_72h", "breakout_72h"),
            ("%-close_breakdown_24h", "breakdown_24h"),
            ("%-close_breakdown_72h", "breakdown_72h"),
            ("%-volume_pressure_24h", "volume_pressure_24h"),
            ("%-up_volume_share_24h", "up_volume_share_24h"),
        ):
            source = dataframe.get(source_column)
            if source is not None:
                interactions[f"%-ob_zone_compression_x_{output_suffix}"] = zone * pd.to_numeric(source, errors="coerce").fillna(0.0)
        spread_shock = dataframe.get("%-ob1h_bybit_spot_beh_spread_shock_24h")
        if spread_shock is not None:
            interactions["%-ob_zone_compression_x_spread_shock_24h"] = (
                zone * pd.to_numeric(spread_shock, errors="coerce").fillna(0.0)
            )
        if self.orderbook_compact_feature_profile == "bybit_rejection_state":
            def numeric_feature(column: str, default: float = 0.0) -> pd.Series:
                source = dataframe.get(column)
                if source is None:
                    return pd.Series(default, index=dataframe.index, dtype="float64")
                return pd.to_numeric(source, errors="coerce").fillna(default)

            range_pos_72h = numeric_feature("%-close_range_pos_72h", 0.5)
            resistance_removed = numeric_feature("%-ob1h_bybit_spot_beh_resistance_removed_score")
            support_removed = numeric_feature("%-ob1h_bybit_spot_beh_support_removed_score")
            bullish_impulse = numeric_feature("%-ob1h_bybit_spot_beh_bullish_impulse_score")
            bearish_impulse = numeric_feature("%-ob1h_bybit_spot_beh_bearish_impulse_score")
            support_rebuild = numeric_feature("%-ob1h_bybit_spot_beh_post_breakout_support_rebuild")
            resistance_rebuild = numeric_feature("%-ob1h_bybit_spot_beh_post_breakdown_resistance_rebuild")
            volume_pressure = numeric_feature("%-volume_pressure_24h")
            upside_pressure = volume_pressure.clip(lower=0.0)
            downside_pressure = (-volume_pressure).clip(lower=0.0)
            interactions["%-ob_state_compressed_near_resistance_score"] = zone * range_pos_72h
            interactions["%-ob_state_compressed_near_support_score"] = zone * (1.0 - range_pos_72h)
            interactions["%-ob_state_failed_upside_acceptance_score"] = (
                zone * resistance_removed * bullish_impulse * upside_pressure
            )
            interactions["%-ob_state_post_breakout_rejection_score"] = support_rebuild * range_pos_72h
            interactions["%-ob_state_downside_risk_score"] = (
                zone * support_removed * bearish_impulse * downside_pressure * (1.0 - range_pos_72h)
            )
            interactions["%-ob_state_post_breakdown_continuation_score"] = resistance_rebuild * (1.0 - range_pos_72h)
        if interactions:
            dataframe = pd.concat([dataframe, DataFrame(interactions, index=dataframe.index)], axis=1)
        return dataframe

    def _append_structure_confluence_event_features(self, dataframe: DataFrame) -> DataFrame:
        def num_feature(column: str) -> pd.Series:
            source = dataframe.get(column)
            if source is None:
                return pd.Series(0.0, index=dataframe.index, dtype="float64")
            return pd.to_numeric(source, errors="coerce").fillna(0.0)

        def bool_feature(column: str) -> pd.Series:
            return num_feature(column).gt(0.0)

        vol_bull = num_feature("%-st_volume_pressure_6h").gt(0.0)
        vol_bear = num_feature("%-st_volume_pressure_6h").lt(0.0)
        range_low = num_feature("%-st_range_position_24h").le(0.35)
        range_high = num_feature("%-st_range_position_24h").ge(0.65)
        support_score = num_feature("%-st_1h_tlv2_support_score_rank0").ge(0.50)
        resistance_score = num_feature("%-st_1h_tlv2_resistance_score_rank0").ge(0.50)
        near_support = num_feature("%-st_near_tlv2_support_1h").gt(0.0)
        near_resistance = num_feature("%-st_near_tlv2_resistance_1h").gt(0.0)
        lower_rejection = bool_feature("%-st_1h_vp_lower_rejection_with_pressure")
        upper_rejection = bool_feature("%-st_1h_vp_upper_rejection_with_pressure")
        hvn_reclaim = bool_feature("%-st_1h_vp_hvn_below_reclaim")
        hvn_reject = bool_feature("%-st_1h_vp_hvn_above_reject")
        below_value = bool_feature("%-st_1h_vp_below_value_area")
        above_value = bool_feature("%-st_1h_vp_above_value_area")
        node_hold_long = bool_feature("%-st_1h_vp_node_hold_long")
        node_hold_short = bool_feature("%-st_1h_vp_node_hold_short")
        vp_long = num_feature("%-st_1h_vp_score_long").ge(0.25)
        vp_short = num_feature("%-st_1h_vp_score_short").ge(0.25)
        value_pos = num_feature("%-st_1h_vp_value_area_position")
        vah_breakout = bool_feature("%-st_1h_vp_vah_breakout_with_pressure")
        ms_bull = bool_feature("%-st_1h_ms_bos_to_bull") | bool_feature("%-st_1h_ms_choch_to_bull")
        ms_bear = bool_feature("%-st_1h_ms_bos_to_bear") | bool_feature("%-st_1h_ms_choch_to_bear")

        events = {
            "%-so_event_vah_rejection": (
                range_high
                & (above_value | value_pos.ge(0.85) | node_hold_short | upper_rejection | hvn_reject)
                & vp_short
            ),
            "%-so_event_tlv2_resistance_reject": near_resistance & resistance_score & (vol_bear | vp_short),
            "%-so_event_vp_lower_rejection": (
                (below_value | value_pos.le(0.15) | node_hold_long | lower_rejection | hvn_reclaim)
                & vp_long
            ),
            "%-so_event_support_fakeout": num_feature("%-st_failed_breakdown_structure_risk").ge(2.0),
            "%-so_event_crash_detection": num_feature("%-st_breakdown_structure_setup").ge(2.0),
            "%-so_event_breakout_acceptance": (
                range_high
                & (above_value | vah_breakout | ms_bull | num_feature("%-st_1h_vp_score_long").ge(0.35))
                & vol_bull
            ),
            "%-so_event_pressure_bearish_context": vol_bear | ms_bear,
        }
        event_features = {column: mask.astype(float).to_numpy() for column, mask in events.items()}
        return pd.concat([dataframe, DataFrame(event_features, index=dataframe.index)], axis=1)

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        for horizon in self.return_label_horizons:
            dataframe[f"&-future_return_{horizon}h"] = dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
            dataframe[f"&-future_up_{horizon}h"] = self._binary_label(
                dataframe[f"&-future_return_{horizon}h"],
                dataframe[f"&-future_return_{horizon}h"] > 0,
            )
        for horizon in self.path_label_horizons:
            future_max = self._future_close_extreme(dataframe["close"], horizon, "max")
            future_min = self._future_close_extreme(dataframe["close"], horizon, "min")
            dataframe[f"&-future_max_upside_{horizon}h"] = future_max / dataframe["close"] - 1.0
            dataframe[f"&-future_max_drawdown_{horizon}h"] = future_min / dataframe["close"] - 1.0
        shock_thresholds = {3: -0.02, 6: -0.025, 24: -0.04, 72: -0.05}
        for horizon, threshold in shock_thresholds.items():
            drawdown_column = f"&-future_max_drawdown_{horizon}h"
            if drawdown_column in dataframe:
                dataframe[f"&-shock_down_{horizon}h"] = self._binary_label(
                    dataframe[drawdown_column],
                    dataframe[drawdown_column] <= threshold,
                )
        breakout_thresholds = {3: 0.02, 6: 0.025, 24: 0.04, 72: 0.05}
        for horizon, threshold in breakout_thresholds.items():
            upside_column = f"&-future_max_upside_{horizon}h"
            if upside_column in dataframe:
                dataframe[f"&-breakout_up_{horizon}h"] = self._binary_label(
                    dataframe[upside_column],
                    dataframe[upside_column] >= threshold,
                )
        if "&-future_max_upside_720h" in dataframe:
            dataframe["&-breakout_up_720h"] = self._binary_label(
                dataframe["&-future_max_upside_720h"],
                dataframe["&-future_max_upside_720h"] >= 0.10,
            )
        if self.include_regime_label and "&-future_return_2160h" in dataframe:
            dataframe["&-regime_up_2160h"] = self._binary_label(
                dataframe["&-future_return_2160h"],
                dataframe["&-future_return_2160h"] >= 0.20,
            )
        if self.include_trader_event_labels:
            dataframe = self._append_trader_event_targets(dataframe)
        if self.include_confluence_event_targets:
            dataframe = self._append_structure_confluence_event_targets_for_freqai(dataframe)
        return dataframe

    @staticmethod
    def _future_close_extreme(close: pd.Series, horizon: int, method: str) -> pd.Series:
        future = close.shift(-1).iloc[::-1]
        rolling = future.rolling(horizon, min_periods=horizon)
        extreme = rolling.max() if method == "max" else rolling.min()
        return extreme.iloc[::-1]

    @staticmethod
    def _binary_label(source: pd.Series, condition: pd.Series) -> pd.Series:
        return condition.astype(float).where(source.notna())

    def _append_trader_event_targets(self, dataframe: DataFrame) -> DataFrame:
        prior_high_24h = dataframe["high"].shift(1).rolling(24, min_periods=12).max()
        prior_low_24h = dataframe["low"].shift(1).rolling(24, min_periods=12).min()
        future_high_6h = self._future_price_extreme(dataframe["high"], 6, "max")
        future_low_6h = self._future_price_extreme(dataframe["low"], 6, "min")
        future_low_3h = self._future_price_extreme(dataframe["low"], 3, "min")
        future_high_24h = self._future_price_extreme(dataframe["high"], 24, "max")
        future_low_24h = self._future_price_extreme(dataframe["low"], 24, "min")
        future_close_3h = dataframe["close"].shift(-3)
        future_close_6h = dataframe["close"].shift(-6)
        future_close_24h = dataframe["close"].shift(-24)
        breakout_attempt_6h = future_high_6h >= prior_high_24h * 1.001
        breakdown_attempt_6h = future_low_6h <= prior_low_24h * 0.999
        breakout_attempt_24h = future_high_24h >= prior_high_24h * 1.001
        breakdown_attempt_24h = future_low_24h <= prior_low_24h * 0.999
        current_resistance_setup = (
            prior_high_24h.notna()
            & (
                dataframe["high"].ge(prior_high_24h * 0.995)
                | dataframe["close"].ge(prior_high_24h * 0.990)
            )
        )
        current_support_setup = (
            prior_low_24h.notna()
            & (
                dataframe["low"].le(prior_low_24h * 1.005)
                | dataframe["close"].le(prior_low_24h * 1.010)
            )
        )
        dataframe["&-breakout_attempt_next_6h"] = breakout_attempt_6h.astype(float).where(prior_high_24h.notna())
        dataframe["&-breakout_success_next_6h"] = (
            breakout_attempt_6h & (future_close_6h > prior_high_24h * 1.001)
        ).astype(float).where(prior_high_24h.notna())
        dataframe["&-breakout_failure_next_6h"] = (
            breakout_attempt_6h & (future_close_6h < prior_high_24h * 0.999)
        ).astype(float).where(prior_high_24h.notna())
        dataframe["&-breakdown_attempt_next_6h"] = breakdown_attempt_6h.astype(float).where(prior_low_24h.notna())
        dataframe["&-breakdown_success_next_6h"] = (
            breakdown_attempt_6h & (future_close_6h < prior_low_24h * 0.999)
        ).astype(float).where(prior_low_24h.notna())
        dataframe["&-breakdown_failure_next_6h"] = (
            breakdown_attempt_6h & (future_close_6h > prior_low_24h * 1.001)
        ).astype(float).where(prior_low_24h.notna())
        dataframe["&-downside_continuation_next_3h"] = (
            future_low_3h.le(dataframe["close"] * 0.985) & future_close_3h.lt(dataframe["close"] * 0.995)
        ).astype(float).where(future_low_3h.notna() & future_close_3h.notna())
        dataframe["&-downside_continuation_next_6h"] = (
            future_low_6h.le(dataframe["close"] * 0.980) & future_close_6h.lt(dataframe["close"] * 0.995)
        ).astype(float).where(future_low_6h.notna() & future_close_6h.notna())
        dataframe["&-downside_exhaustion_next_6h"] = (
            future_low_6h.gt(dataframe["close"] * 0.990) | future_close_6h.gt(dataframe["close"] * 1.003)
        ).astype(float).where(future_low_6h.notna() & future_close_6h.notna())
        dataframe["&-support_reclaim_next_6h"] = (
            dataframe["close"].lt(prior_low_24h * 1.001) & future_close_6h.gt(prior_low_24h * 1.001)
        ).astype(float).where(prior_low_24h.notna() & future_close_6h.notna())
        dataframe["&-failed_breakout_next_24h"] = (
            breakout_attempt_24h & (future_close_24h < prior_high_24h)
        ).astype(float).where(prior_high_24h.notna())
        dataframe["&-failed_breakdown_next_24h"] = (
            breakdown_attempt_24h & (future_close_24h > prior_low_24h)
        ).astype(float).where(prior_low_24h.notna())
        dataframe["&-fakeout_next_24h"] = (
            (breakout_attempt_24h & (future_close_24h < prior_high_24h))
            | (breakdown_attempt_24h & (future_close_24h > prior_low_24h))
        ).astype(float).where(prior_high_24h.notna() & prior_low_24h.notna())
        dataframe["&-current_resistance_setup"] = current_resistance_setup.astype(float).where(prior_high_24h.notna())
        dataframe["&-current_support_setup"] = current_support_setup.astype(float).where(prior_low_24h.notna())
        dataframe["&-breakout_success_from_current_setup_6h"] = (
            current_resistance_setup & breakout_attempt_6h & future_close_6h.gt(prior_high_24h * 1.001)
        ).astype(float).where(current_resistance_setup)
        dataframe["&-breakout_failure_from_current_setup_6h"] = (
            current_resistance_setup & breakout_attempt_6h & future_close_6h.lt(prior_high_24h * 0.999)
        ).astype(float).where(current_resistance_setup)
        dataframe["&-breakdown_success_from_current_setup_6h"] = (
            current_support_setup & breakdown_attempt_6h & future_close_6h.lt(prior_low_24h * 0.999)
        ).astype(float).where(current_support_setup)
        dataframe["&-breakdown_failure_from_current_setup_6h"] = (
            current_support_setup & breakdown_attempt_6h & future_close_6h.gt(prior_low_24h * 1.001)
        ).astype(float).where(current_support_setup)
        dataframe["&-fakeout_from_current_setup_24h"] = (
            (current_resistance_setup & breakout_attempt_24h & future_close_24h.lt(prior_high_24h))
            | (current_support_setup & breakdown_attempt_24h & future_close_24h.gt(prior_low_24h))
        ).astype(float).where(current_resistance_setup | current_support_setup)
        return dataframe

    def _append_structure_confluence_event_targets_for_freqai(self, dataframe: DataFrame) -> DataFrame:
        if "%-so_event_vah_rejection" not in dataframe:
            raise ValueError("Confluence event target requested before %-so_event_vah_rejection was built.")
        if "&-breakout_failure_next_6h" not in dataframe:
            raise ValueError("Confluence event target requested before &-breakout_failure_next_6h was built.")
        event = pd.to_numeric(dataframe["%-so_event_vah_rejection"], errors="coerce").fillna(0.0).gt(0.0)
        if self.require_orderbook_present_for_confluence_targets:
            if "orderbook_present" not in dataframe:
                raise ValueError("Orderbook-present target mask requested before orderbook_present was built.")
            event = event & pd.to_numeric(dataframe["orderbook_present"], errors="coerce").fillna(0.0).gt(0.0)
        source = pd.to_numeric(dataframe["&-breakout_failure_next_6h"], errors="coerce")
        dataframe["&-so_vah_rejection_breakout_failure_next_6h"] = source.where(event)
        setup_source_column = "&-breakout_failure_from_current_setup_6h"
        if setup_source_column in dataframe:
            setup_source = pd.to_numeric(dataframe[setup_source_column], errors="coerce")
            dataframe["&-so_vah_rejection_breakout_failure_from_current_setup_6h"] = setup_source.where(event)
        return dataframe

    @staticmethod
    def _future_price_extreme(series: pd.Series, horizon: int, method: str) -> pd.Series:
        future = series.shift(-1).iloc[::-1]
        rolling = future.rolling(horizon, min_periods=horizon)
        extreme = rolling.max() if method == "max" else rolling.min()
        return extreme.iloc[::-1]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return_6h"] > 0,
        ]
        if long_conditions:
            dataframe.loc[
                reduce(lambda left, right: left & right, long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "context_freqai_research_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return_6h"] < 0,
        ]
        if exit_conditions:
            dataframe.loc[
                reduce(lambda left, right: left & right, exit_conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "context_freqai_research_exit")
        return dataframe


class ContextFreqAIResearchPriceOnlyStrategy(ContextFreqAIResearchStrategy):
    feature_mode = "price_only"


class ContextFreqAIResearchContextOnlyStrategy(ContextFreqAIResearchStrategy):
    feature_mode = "context_only"


class ContextFreqAIResearchContextEventStrategy(ContextFreqAIResearchStrategy):
    feature_mode = "context_only"
    include_trader_event_labels = True
    return_label_horizons = (1, 3, 6, 24)
    path_label_horizons = (3, 6, 24, 72)


class ContextFreqAIResearchPriceContextEventStrategy(ContextFreqAIResearchContextEventStrategy):
    feature_mode = "combined"


class ContextFreqAIResearchFullHorizonStrategy(ContextFreqAIResearchStrategy):
    return_label_horizons = (1, 6, 24, 72, 168, 336, 720, 2160)
    path_label_horizons = (72, 168, 720)
    include_regime_label = True


class ContextOrderbookFreqAIResearchStrategy(ContextFreqAIResearchStrategy):
    include_orderbook_context = True
    startup_candle_count = 96
    return_label_horizons = (1, 6, 24)
    path_label_horizons = (24, 72)


class ContextOrderbookPriceOnlyFreqAIResearchStrategy(ContextOrderbookFreqAIResearchStrategy):
    include_orderbook_context = False
    feature_mode = "price_only"


class ContextBybitOrderbookBehaviourPriceOnlyFreqAIResearchStrategy(ContextOrderbookPriceOnlyFreqAIResearchStrategy):
    return_label_horizons = (1, 3, 6, 24)
    path_label_horizons = (3, 6, 24, 72)


class ContextBybitOrderbookTraderEventPriceOnlyFreqAIResearchStrategy(ContextBybitOrderbookBehaviourPriceOnlyFreqAIResearchStrategy):
    include_trader_event_labels = True


class ContextOrderbookOnlyFreqAIResearchStrategy(ContextOrderbookFreqAIResearchStrategy):
    feature_mode = "price_only"


class ContextOrderbookOnlyParquetFreqAIResearchStrategy(ContextOrderbookOnlyFreqAIResearchStrategy):
    orderbook_compact_feature_file = (
        r"C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_20260523_232420.parquet"
    )


class ContextOrderbookHeadlineFreqAIResearchStrategy(ContextOrderbookOnlyFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "headline"


class ContextBybitOrderbookHeadlineFreqAIResearchStrategy(ContextOrderbookOnlyFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_headline"


class ContextBybitOrderbookRefinedParquetFreqAIResearchStrategy(ContextOrderbookOnlyFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_refined"
    orderbook_compact_feature_file = (
        r"C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_refined_latest.parquet"
    )


class ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy(ContextOrderbookOnlyFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_behaviour"
    orderbook_compact_feature_file = (
        r"C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_behaviour_latest.parquet"
    )
    return_label_horizons = (1, 3, 6, 24)
    path_label_horizons = (3, 6, 24, 72)


class ContextBybitOrderbookZoneCompressionParquetFreqAIResearchStrategy(ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_zone_compression"


class ContextBybitOrderbookRejectionStateParquetFreqAIResearchStrategy(ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_rejection_state"


class ContextBybitOrderbookObjectiveAlphaParquetFreqAIResearchStrategy(ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_objective_alpha"
    orderbook_compact_feature_file = (
        r"C:\FreqTradeStuff\user_data\orderbook_data\live\exports\orderbook_features_1h_alpha_latest.parquet"
    )


class ContextBybitOrderbookTraderStateParquetFreqAIResearchStrategy(ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_trader_state"
    orderbook_compact_feature_file = (
        r"C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet"
    )
    include_trader_event_labels = True


class ContextBybitOrderbookTraderRefinedParquetFreqAIResearchStrategy(ContextBybitOrderbookTraderStateParquetFreqAIResearchStrategy):
    orderbook_compact_feature_profile = "bybit_trader_refined"


class ContextStructureVahRejectionFreqAIResearchStrategy(ContextFreqAIResearchStrategy):
    feature_mode = "price_only"
    include_structure_context = True
    include_confluence_event_features = True
    include_confluence_event_targets = True
    include_trader_event_labels = True
    return_label_horizons = (1, 3, 6, 24)
    path_label_horizons = (3, 6, 24)
    startup_candle_count = 240


class ContextStructureOrderbookVahRejectionFreqAIResearchStrategy(ContextStructureVahRejectionFreqAIResearchStrategy):
    include_orderbook_context = True
    require_orderbook_present_for_confluence_targets = True
    orderbook_compact_feature_profile = "bybit_trader_refined"
    orderbook_compact_feature_file = (
        r"C:\FreqTradeStuff\user_data\orderbook_data\historical_bybit\features\orderbook_trader_state_1h_latest.parquet"
    )


class ContextTraderConfluenceDeltaFreqAIResearchStrategy(ContextBybitOrderbookTraderEventPriceOnlyFreqAIResearchStrategy):
    include_trader_confluence_context = True
    startup_candle_count = 240
    return_label_horizons = (1, 3, 6, 24)
    path_label_horizons = (3, 6, 24, 72)


class ContextTraderConfluenceDeltaComponentsFreqAIResearchStrategy(ContextTraderConfluenceDeltaFreqAIResearchStrategy):
    trader_confluence_feature_profile = "delta_components_only"


class ContextTraderConfluenceEpsilonFreqAIResearchStrategy(ContextTraderConfluenceDeltaFreqAIResearchStrategy):
    trader_confluence_feature_profile = "epsilon_with_flags"


class ContextTraderConfluenceEpsilonComponentsFreqAIResearchStrategy(ContextTraderConfluenceDeltaFreqAIResearchStrategy):
    trader_confluence_feature_profile = "epsilon_components_only"
