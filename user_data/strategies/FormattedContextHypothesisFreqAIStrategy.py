from __future__ import annotations

from functools import reduce
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

from user_data.Indicators.complex_volume_profile import add_volume_profile


USER_DATA_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "live_aligned_snapshots"
SNAPSHOT_PATH = SNAPSHOT_DIR / "live_context_orderbook_aligned_1h_latest.parquet"
COLUMNS_PATH = SNAPSHOT_DIR / "live_context_orderbook_aligned_1h_latest.columns.csv"


class FormattedContextHypothesisFreqAIBaseStrategy(IStrategy):
    """
    Research-only FreqAI bridge for the strict live context/orderbook aligned snapshot.

    This is not a production strategy. It exposes named, timestamp-safe feature
    families to FreqAI so we can test whether they rank trader-readable future
    path outcomes.
    """

    timeframe = "1h"
    startup_candle_count = 240
    minimal_roi = {"0": 0.0}
    stoploss = -0.99
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True

    feature_panels: tuple[str, ...] = ("fs_price_volume",)
    include_roles: tuple[str, ...] = (
        "price_volume_feature",
        "derived_transform",
        "trader_confluence_feature",
        "trader_state_feature",
    )
    require_context_available = False
    require_live_media_available = False
    require_orderbook_available = False
    require_orderbook_loose_available = False
    include_market_raw = False
    include_live_media_trader_shapes = False
    include_volume_profile_features = False
    feature_include_patterns: tuple[str, ...] = ()
    feature_exclude_patterns: tuple[str, ...] = ()
    min_feature_nonnull_ratio = 0.95
    live_media_topic_groups: dict[str, tuple[str, ...]] = {
        "risk_off": (
            "war_geopolitical",
            "war_geopolitics",
            "sanctions_trade",
            "oil_energy_shock",
            "oil_energy",
            "security_exploit",
            "dollar_risk_off",
            "pandemic_health_shock",
            "ai_bubble_risk",
        ),
        "macro_policy": (
            "rates",
            "inflation",
            "central_bank",
            "official_data_release",
            "official_macro_release",
            "inflation_rates",
            "recession_growth",
            "jobs_labor",
        ),
        "financial_stress": (
            "banking_liquidity",
            "banking_credit",
            "china_property_credit",
            "stablecoin_liquidity",
            "liquidity_stablecoin",
            "liquidation_leverage",
        ),
        "crypto_specific": (
            "etf_institutional_flow",
            "etf_institutional",
            "regulation_enforcement",
            "regulation_legal",
            "exchange_listing_delisting",
            "crypto_market_structure",
            "crypto_native",
        ),
        "relief": (
            "peace_talks_deescalation",
        ),
    }

    _feature_map_cache: dict[tuple[str, tuple[str, ...], bool, tuple[str, ...], tuple[str, ...]], dict[str, str]] = {}
    _snapshot_cache: dict[tuple[str, str, tuple[str, ...], bool, bool, bool, bool], DataFrame] = {}

    @staticmethod
    def _canonical_pair(pair: str | None) -> str:
        text = str(pair or "BTC/USDT").upper().strip()
        text = text.split(":", 1)[0]
        return text

    @classmethod
    def _columns_frame(cls) -> DataFrame:
        path = COLUMNS_PATH
        if not path.exists():
            candidates = sorted(
                SNAPSHOT_DIR.glob("live_context_orderbook_aligned_1h_*.columns.csv"),
                key=lambda candidate: candidate.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(COLUMNS_PATH)
            path = candidates[-1]
        return pd.read_csv(path)

    @classmethod
    def _feature_map(cls) -> dict[str, str]:
        cache_key = (
            cls.__name__,
            tuple(cls.feature_panels),
            bool(cls.include_market_raw),
            tuple(cls.feature_include_patterns),
            tuple(cls.feature_exclude_patterns),
        )
        cached = cls._feature_map_cache.get(cache_key)
        if cached is not None:
            return cached
        columns = cls._columns_frame()
        columns["freqai_candidate_numeric"] = columns["freqai_candidate_numeric"].fillna(False).astype(bool)
        panel_mask = columns["feature_panel"].astype(str).isin(cls.feature_panels)
        role_mask = columns["role"].astype(str).isin(cls.include_roles)
        candidate_mask = columns["freqai_candidate_numeric"]
        selected = columns.loc[panel_mask & role_mask & candidate_mask, "column"].astype(str).tolist()
        if cls.include_market_raw:
            market = columns.loc[
                columns["feature_panel"].astype(str).eq("fs_market_breadth_raw")
                & columns["column"].astype(str).str.startswith("%-"),
                "column",
            ].astype(str)
            selected.extend(market.tolist())
        selected = cls._filter_selected_features(selected)
        feature_map: dict[str, str] = {}
        used: set[str] = set()
        for source in selected:
            if source in {"date", "pair", "open", "high", "low", "close", "volume"}:
                continue
            target = cls._freqai_name(source)
            if target in used:
                continue
            feature_map[source] = target
            used.add(target)
        if not feature_map:
            raise ValueError(f"No FreqAI feature columns selected for {cls.__name__}: {cls.feature_panels}")
        cls._feature_map_cache[cache_key] = feature_map
        return feature_map

    @classmethod
    def _filter_selected_features(cls, selected: list[str]) -> list[str]:
        include_patterns = tuple(pattern for pattern in cls.feature_include_patterns if str(pattern).strip())
        exclude_patterns = tuple(pattern for pattern in cls.feature_exclude_patterns if str(pattern).strip())
        if not include_patterns and not exclude_patterns:
            return selected
        filtered: list[str] = []
        for source in selected:
            if include_patterns and not any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in include_patterns):
                continue
            if exclude_patterns and any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in exclude_patterns):
                continue
            filtered.append(source)
        return filtered

    @staticmethod
    def _freqai_name(source: str) -> str:
        if source.startswith("%-"):
            raw = source[2:]
        else:
            raw = f"la_{source}"
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
        return f"%-{safe}"

    @classmethod
    def _load_snapshot_for_pair(cls, pair: str) -> DataFrame:
        canonical_pair = cls._canonical_pair(pair)
        feature_map = cls._feature_map()
        cache_key = (
            cls.__name__,
            canonical_pair,
            tuple(feature_map.keys()),
            bool(cls.include_market_raw),
            bool(cls.include_live_media_trader_shapes),
            bool(cls.include_volume_profile_features),
            bool(cls.require_orderbook_loose_available),
        )
        cached = cls._snapshot_cache.get(cache_key)
        if cached is not None:
            return cached
        if not SNAPSHOT_PATH.exists():
            raise FileNotFoundError(SNAPSHOT_PATH)
        columns = [
            "date",
            "pair",
            "conf_external_non_market_safe_source_count",
            "context_news_web_present_24h",
            "context_global_present",
            "conf_orderbook_ready_market_count",
            "conf_orderbook_ready_market_ratio",
            "orderbook_usable_loose",
            "orderbook_tick_present",
            *feature_map.keys(),
            *cls._extra_snapshot_columns(),
        ]
        available_columns = set(cls._columns_frame()["column"].astype(str).tolist())
        available_columns.update({"date", "pair"})
        columns = [column for column in columns if column in available_columns]
        columns = list(dict.fromkeys(columns))
        frame = pd.read_parquet(SNAPSHOT_PATH, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame[frame["pair"].astype(str).str.upper().eq(canonical_pair)].copy()
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        ready_mask = pd.Series(True, index=frame.index)
        if cls.require_context_available:
            ready_mask &= (
                cls._numeric_column(frame, "conf_external_non_market_safe_source_count")
                .fillna(0.0)
                .gt(0.0)
            )
        if cls.require_live_media_available:
            ready_mask &= cls._live_media_ready(frame)
        if cls.require_orderbook_available:
            ready_mask &= (
                cls._numeric_column(frame, "conf_orderbook_ready_market_count")
                .fillna(0.0)
                .gt(0.0)
            )
        if cls.require_orderbook_loose_available:
            ready_mask &= cls._loose_orderbook_ready(frame)
        if not ready_mask.any():
            ready_mask = pd.Series(True, index=frame.index)
        feature_series: dict[str, pd.Series] = {}
        for source, target in feature_map.items():
            series = pd.to_numeric(frame[source], errors="coerce")
            nonnull_ratio = float(series.loc[ready_mask].notna().mean()) if ready_mask.any() else float(series.notna().mean())
            if nonnull_ratio >= float(cls.min_feature_nonnull_ratio):
                feature_series[target] = series
        if cls.include_live_media_trader_shapes:
            cls._append_live_media_trader_shapes(frame, feature_series, ready_mask)
        if cls.include_volume_profile_features:
            cls._append_volume_profile_features(frame, feature_series, ready_mask)
        if not feature_series:
            raise ValueError(f"No populated FreqAI features selected for {cls.__name__} / {canonical_pair}")
        output = pd.concat([frame[["date"]], DataFrame(feature_series, index=frame.index)], axis=1)
        output["fmt_context_ready"] = (
            cls._numeric_column(frame, "conf_external_non_market_safe_source_count")
            .fillna(0.0)
            .gt(0.0)
            .astype(float)
        )
        output["fmt_live_media_ready"] = cls._live_media_ready(frame).astype(float)
        output["fmt_orderbook_ready"] = (
            cls._numeric_column(frame, "conf_orderbook_ready_market_count")
            .fillna(0.0)
            .gt(0.0)
            .astype(float)
        )
        output["fmt_orderbook_ready_ratio"] = cls._numeric_column(frame, "conf_orderbook_ready_market_ratio")
        output["fmt_orderbook_loose_ready"] = cls._loose_orderbook_ready(frame).astype(float)
        cls._snapshot_cache[cache_key] = output.reset_index(drop=True)
        return cls._snapshot_cache[cache_key]

    @staticmethod
    def _numeric_column(frame: DataFrame, column: str, default: float = 0.0) -> pd.Series:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(default, index=frame.index, dtype=float)

    @classmethod
    def _extra_snapshot_columns(cls) -> tuple[str, ...]:
        columns: list[str] = []
        if cls.include_live_media_trader_shapes:
            columns.extend(
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "ctx_article_count_1h",
                    "ctx_article_count_6h",
                    "ctx_article_count_24h",
                    "ctx_news_article_count_24h",
                    "ctx_web_article_count_24h",
                    "ctx_unique_source_count_24h",
                    "ctx_source_group_count_24h",
                    "ctx_max_sources_same_topic_24h",
                    "ctx_topic_count_24h",
                    "ctx_top_topic_share_6h",
                    "ctx_news_volume_acceleration_6h",
                    "ctx_news_rows_24h",
                    "ctx_web_rows_24h",
                    "ctx_global_metrics_available",
                    "ctx_global_market_cap_change_24h",
                    "ctx_btc_eth_avg_change_24h",
                    "ctx_fear_greed_delta_24h",
                    "ctx_weighted_context_total_intensity_6h",
                    "ctx_weighted_context_total_intensity_24h",
                    "ctx_systemic_risk_stack_intensity_24h",
                    "ctx_macro_policy_stack_intensity_24h",
                    "ctx_crypto_policy_stack_intensity_24h",
                    "ctx_crypto_stress_stack_intensity_24h",
                    "ctx_cross_topic_confluence_max_24h",
                    "ctx_cross_topic_persistence_count_24h",
                    "ctx_high_severity_topic_count_24h",
                    "ctx_official_confirmed_context_intensity_24h",
                ]
            )
            for bases in cls.live_media_topic_groups.values():
                for base in bases:
                    columns.extend(
                        [
                            f"ctx_{base}_count_1h",
                            f"ctx_{base}_count_6h",
                            f"ctx_{base}_count_24h",
                            f"ctx_{base}_source_count_24h",
                            f"ctx_{base}_acceleration_6h",
                            f"ctx_{base}_z_7d",
                            f"ctx_{base}_intensity_1h",
                            f"ctx_{base}_intensity_6h",
                            f"ctx_{base}_intensity_24h",
                            f"ctx_{base}_confluence_6h",
                            f"ctx_{base}_confluence_24h",
                            f"ctx_{base}_persistence_hours_24h",
                            f"ctx_{base}_severity_max_24h",
                            f"ctx_{base}_official_confirmed_intensity_24h",
                            f"ctx_{base}_novelty_z_30d",
                            f"ctx_{base}_intensity_acceleration_6h",
                        ]
                    )
        if cls.include_volume_profile_features:
            columns.extend(["open", "high", "low", "close", "volume"])
        return tuple(dict.fromkeys(columns))

    @classmethod
    def _append_feature(
        cls,
        feature_series: dict[str, pd.Series],
        ready_mask: pd.Series,
        name: str,
        series: pd.Series,
    ) -> None:
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        sample = numeric.loc[ready_mask] if ready_mask.any() else numeric
        if sample.notna().mean() < 0.80 or sample.nunique(dropna=True) < 2:
            return
        feature_series[f"%-{name}"] = numeric.fillna(0.0)

    @classmethod
    def _topic_group_level(cls, frame: DataFrame, bases: tuple[str, ...]) -> pd.Series:
        pieces: list[pd.Series] = []
        for base in bases:
            for suffix in (
                "intensity_1h",
                "count_1h",
                "intensity_6h",
                "count_6h",
                "intensity_acceleration_6h",
                "acceleration_6h",
            ):
                column = f"ctx_{base}_{suffix}"
                if column in frame:
                    pieces.append(cls._numeric_column(frame, column).fillna(0.0).clip(lower=0.0))
        if not pieces:
            return pd.Series(0.0, index=frame.index)
        return sum(pieces)

    @classmethod
    def _append_live_media_trader_shapes(
        cls,
        frame: DataFrame,
        feature_series: dict[str, pd.Series],
        ready_mask: pd.Series,
    ) -> None:
        group_levels: dict[str, pd.Series] = {}
        group_rising: dict[tuple[str, int], pd.Series] = {}
        for group_name, bases in cls.live_media_topic_groups.items():
            raw_level = cls._topic_group_level(frame, bases)
            level = np.log1p(raw_level.clip(lower=0.0))
            group_levels[group_name] = level
            cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_level_log", level)
            prior_mean = level.shift(1).rolling(24, min_periods=8).mean()
            prior_std = level.shift(1).rolling(24, min_periods=8).std().replace(0.0, np.nan)
            cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_shock_z24", (level - prior_mean) / prior_std)
            for horizon in (1, 2, 4):
                change = level - level.shift(horizon)
                previous_change = level.shift(horizon) - level.shift(2 * horizon)
                rising = (change.gt(0.0) & level.gt(0.0)).astype(float)
                fresh_cross = (
                    level.gt(0.0)
                    & level.shift(1).rolling(horizon, min_periods=1).max().fillna(0.0).le(0.0)
                ).astype(float)
                persistence = level.gt(0.0).astype(float).rolling(horizon, min_periods=1).mean()
                group_rising[(group_name, horizon)] = rising
                cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_chg_{horizon}h", change)
                cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_accel_{horizon}h", change - previous_change)
                cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_rising_{horizon}h", rising)
                cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_fresh_cross_{horizon}h", fresh_cross)
                cls._append_feature(feature_series, ready_mask, f"lm_{group_name}_persistence_{horizon}h", persistence)

        news_24h = cls._numeric_column(frame, "ctx_news_article_count_24h").fillna(0.0).clip(lower=0.0)
        web_24h = cls._numeric_column(frame, "ctx_web_article_count_24h").fillna(0.0).clip(lower=0.0)
        global_available = cls._numeric_column(frame, "ctx_global_metrics_available").fillna(0.0).gt(0.0).astype(float)
        source_active = news_24h.gt(0.0).astype(float) + web_24h.gt(0.0).astype(float) + global_available
        cls._append_feature(feature_series, ready_mask, "lm_sources_active_count", source_active)
        same_topic = cls._numeric_column(frame, "ctx_max_sources_same_topic_24h").fillna(0.0).clip(lower=0.0)
        unique_sources = cls._numeric_column(frame, "ctx_unique_source_count_24h").fillna(0.0).clip(lower=0.0)
        source_groups = cls._numeric_column(frame, "ctx_source_group_count_24h").fillna(0.0).clip(lower=0.0)
        cls._append_feature(feature_series, ready_mask, "lm_source_confluence_log_24h", np.log1p(unique_sources + source_groups + same_topic))
        cls._append_feature(
            feature_series,
            ready_mask,
            "lm_same_topic_concentration_24h",
            same_topic / unique_sources.replace(0.0, np.nan),
        )
        article_1h = cls._numeric_column(frame, "ctx_article_count_1h").fillna(0.0).clip(lower=0.0)
        article_prior_mean = article_1h.shift(1).rolling(24, min_periods=8).mean()
        article_prior_std = article_1h.shift(1).rolling(24, min_periods=8).std().replace(0.0, np.nan)
        article_burst_z24 = (article_1h - article_prior_mean) / article_prior_std
        cls._append_feature(feature_series, ready_mask, "lm_article_burst_z24", article_burst_z24)

        risk_level = group_levels.get("risk_off", pd.Series(0.0, index=frame.index))
        relief_level = group_levels.get("relief", pd.Series(0.0, index=frame.index))
        crypto_level = group_levels.get("crypto_specific", pd.Series(0.0, index=frame.index))
        macro_level = group_levels.get("macro_policy", pd.Series(0.0, index=frame.index))
        financial_level = group_levels.get("financial_stress", pd.Series(0.0, index=frame.index))
        cls._append_feature(feature_series, ready_mask, "lm_risk_minus_relief_level", risk_level - relief_level)
        cls._append_feature(feature_series, ready_mask, "lm_stress_minus_crypto_level", (risk_level + financial_level + macro_level) - crypto_level)
        for horizon in (1, 2, 4):
            source_rising = (
                news_24h.gt(news_24h.shift(horizon)).astype(float)
                + web_24h.gt(web_24h.shift(horizon)).astype(float)
                + global_available.gt(global_available.shift(horizon).fillna(0.0)).astype(float)
            )
            topics_rising = sum(group_rising.get((name, horizon), pd.Series(0.0, index=frame.index)) for name in cls.live_media_topic_groups)
            cls._append_feature(feature_series, ready_mask, f"lm_sources_rising_count_{horizon}h", source_rising)
            cls._append_feature(feature_series, ready_mask, f"lm_topics_rising_count_{horizon}h", topics_rising)

            close = cls._numeric_column(frame, "close").replace(0.0, np.nan)
            high = cls._numeric_column(frame, "high")
            low = cls._numeric_column(frame, "low")
            volume = cls._numeric_column(frame, "volume").fillna(0.0).clip(lower=0.0)
            price_return = close / close.shift(horizon) - 1.0
            prior_high = high.shift(1).rolling(24, min_periods=8).max()
            prior_low = low.shift(1).rolling(24, min_periods=8).min()
            volume_z = (np.log1p(volume) - np.log1p(volume).shift(1).rolling(24, min_periods=8).mean()) / np.log1p(volume).shift(1).rolling(24, min_periods=8).std().replace(0.0, np.nan)
            risk_rising = group_rising.get(("risk_off", horizon), pd.Series(0.0, index=frame.index)).gt(0.0)
            relief_rising = group_rising.get(("relief", horizon), pd.Series(0.0, index=frame.index)).gt(0.0)
            crypto_rising = group_rising.get(("crypto_specific", horizon), pd.Series(0.0, index=frame.index)).gt(0.0)
            cls._append_feature(feature_series, ready_mask, f"lm_risk_rising_price_falling_{horizon}h", (risk_rising & price_return.lt(0.0)).astype(float))
            cls._append_feature(feature_series, ready_mask, f"lm_risk_rising_price_resilient_{horizon}h", (risk_rising & price_return.gt(0.0)).astype(float))
            cls._append_feature(feature_series, ready_mask, f"lm_relief_rising_price_rising_{horizon}h", (relief_rising & price_return.gt(0.0)).astype(float))
            cls._append_feature(feature_series, ready_mask, f"lm_crypto_rising_price_rising_{horizon}h", (crypto_rising & price_return.gt(0.0)).astype(float))
            cls._append_feature(feature_series, ready_mask, f"lm_source_burst_volume_expansion_{horizon}h", (source_rising.ge(2.0) & volume_z.gt(0.75)).astype(float))
            cls._append_feature(feature_series, ready_mask, f"lm_risk_rising_near_support_break_{horizon}h", (risk_rising & close.le(prior_low * 1.005)).astype(float))
            cls._append_feature(feature_series, ready_mask, f"lm_crypto_rising_near_resistance_break_{horizon}h", (crypto_rising & close.ge(prior_high * 0.995)).astype(float))

    @classmethod
    def _append_volume_profile_features(
        cls,
        frame: DataFrame,
        feature_series: dict[str, pd.Series],
        ready_mask: pd.Series,
    ) -> None:
        source = frame[["open", "high", "low", "close", "volume"]].copy()
        vp = add_volume_profile(
            source,
            window=96,
            bins=48,
            value_area_pct=0.70,
            price_source="hlc3",
            smooth_bins=3,
            pressure_delta_min=0.05,
            node_near_pct=0.01,
            volume_percentile_min=0.55,
            score_window=48,
            fast_traverse_atr_mult=1.20,
            entry_score_margin=0.02,
            prefix="vp",
        )
        key_columns = (
            "vp_distance_to_poc_pct",
            "vp_distance_to_vah_pct",
            "vp_distance_to_val_pct",
            "vp_value_area_width_pct",
            "vp_value_area_position",
            "vp_close_bin_volume_share",
            "vp_close_bin_volume_percentile",
            "vp_in_value_area",
            "vp_above_value_area",
            "vp_below_value_area",
            "vp_vah_breakout",
            "vp_val_breakdown",
            "vp_upper_rejection",
            "vp_lower_rejection",
            "vp_score_long",
            "vp_score_short",
            "vp_score_abs",
            "vp_state",
            "vp_context_score_bull",
            "vp_context_score_bear",
            "vp_context_score_balance",
            "vp_market_context",
            "vp_entry_trigger_long",
            "vp_entry_trigger_short",
            "vp_node_entry_long",
            "vp_node_entry_short",
            "vp_node_hold_long",
            "vp_node_hold_short",
            "vp_node_exit_long",
            "vp_node_exit_short",
            "vp_hvn_above_strength",
            "vp_hvn_below_strength",
            "vp_lvn_above_thinness",
            "vp_lvn_below_thinness",
            "vp_hvn_above_distance_pct",
            "vp_hvn_below_distance_pct",
            "vp_lvn_above_distance_pct",
            "vp_lvn_below_distance_pct",
            "vp_nearest_hvn_distance_pct",
            "vp_nearest_lvn_distance_pct",
        )
        for column in key_columns:
            if column in vp:
                cls._append_feature(feature_series, ready_mask, column, pd.to_numeric(vp[column], errors="coerce"))

        risk_level = feature_series.get("%-lm_risk_off_level_log", pd.Series(0.0, index=frame.index))
        stress_level = feature_series.get("%-lm_stress_minus_crypto_level", pd.Series(0.0, index=frame.index))
        crypto_level = feature_series.get("%-lm_crypto_specific_level_log", pd.Series(0.0, index=frame.index))
        relief_level = feature_series.get("%-lm_relief_level_log", pd.Series(0.0, index=frame.index))
        vp_long = cls._numeric_column(vp, "vp_score_long").fillna(0.0)
        vp_short = cls._numeric_column(vp, "vp_score_short").fillna(0.0)
        vp_above = cls._numeric_column(vp, "vp_above_value_area").fillna(0.0).gt(0.0)
        vp_below = cls._numeric_column(vp, "vp_below_value_area").fillna(0.0).gt(0.0)
        vp_val_break = cls._numeric_column(vp, "vp_val_breakdown").fillna(0.0).gt(0.0)
        vp_vah_break = cls._numeric_column(vp, "vp_vah_breakout").fillna(0.0).gt(0.0)
        cls._append_feature(feature_series, ready_mask, "lmvp_risk_x_vp_short_score", pd.to_numeric(risk_level, errors="coerce").fillna(0.0) * vp_short)
        cls._append_feature(feature_series, ready_mask, "lmvp_stress_x_vp_short_score", pd.to_numeric(stress_level, errors="coerce").fillna(0.0).clip(lower=0.0) * vp_short)
        cls._append_feature(feature_series, ready_mask, "lmvp_crypto_x_vp_long_score", pd.to_numeric(crypto_level, errors="coerce").fillna(0.0) * vp_long)
        cls._append_feature(feature_series, ready_mask, "lmvp_relief_x_vp_long_score", pd.to_numeric(relief_level, errors="coerce").fillna(0.0) * vp_long)
        cls._append_feature(feature_series, ready_mask, "lmvp_risk_below_value_area", (pd.to_numeric(risk_level, errors="coerce").fillna(0.0).gt(0.0) & vp_below).astype(float))
        cls._append_feature(feature_series, ready_mask, "lmvp_crypto_above_value_area", (pd.to_numeric(crypto_level, errors="coerce").fillna(0.0).gt(0.0) & vp_above).astype(float))
        cls._append_feature(feature_series, ready_mask, "lmvp_risk_val_breakdown", (pd.to_numeric(risk_level, errors="coerce").fillna(0.0).gt(0.0) & vp_val_break).astype(float))
        cls._append_feature(feature_series, ready_mask, "lmvp_crypto_vah_breakout", (pd.to_numeric(crypto_level, errors="coerce").fillna(0.0).gt(0.0) & vp_vah_break).astype(float))

    @classmethod
    def _live_media_ready(cls, frame: DataFrame) -> pd.Series:
        news_web = cls._numeric_column(frame, "context_news_web_present_24h").fillna(0.0).gt(0.0)
        global_context = cls._numeric_column(frame, "context_global_present").fillna(0.0).gt(0.0)
        return news_web | global_context

    @classmethod
    def _loose_orderbook_ready(cls, frame: DataFrame) -> pd.Series:
        loose = cls._numeric_column(frame, "orderbook_usable_loose").fillna(0.0).gt(0.0)
        tick = cls._numeric_column(frame, "orderbook_tick_present").fillna(0.0).gt(0.0)
        return loose | tick

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        pair = self._canonical_pair(metadata.get("pair"))
        features = self._load_snapshot_for_pair(pair)
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        merged = dataframe.merge(features, on="date", how="left", sort=False)
        return merged

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe = dataframe.copy()
        for horizon in (1, 2, 3, 4, 6, 24):
            dataframe[f"&-future_return_{horizon}h"] = dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
            upside = self._future_extreme(dataframe["high"], horizon, "max") / dataframe["close"] - 1.0
            drawdown = self._future_extreme(dataframe["low"], horizon, "min") / dataframe["close"] - 1.0
            dataframe[f"&-future_max_upside_{horizon}h"] = upside
            dataframe[f"&-future_max_drawdown_{horizon}h"] = drawdown
            reward, danger, large_move = self._path_thresholds(horizon)
            dataframe[f"&-long_reward_before_danger_{horizon}h"] = self._reward_before_danger(
                dataframe["close"],
                horizon=horizon,
                reward=reward,
                danger=-danger,
                long_side=True,
            )
            dataframe[f"&-short_reward_before_danger_{horizon}h"] = self._reward_before_danger(
                dataframe["close"],
                horizon=horizon,
                reward=reward,
                danger=-danger,
                long_side=False,
            )
            dataframe[f"&-large_upside_next_{horizon}h"] = (
                dataframe[f"&-future_max_upside_{horizon}h"].ge(large_move).astype(float).where(dataframe[f"&-future_max_upside_{horizon}h"].notna())
            )
            dataframe[f"&-large_drawdown_next_{horizon}h"] = (
                dataframe[f"&-future_max_drawdown_{horizon}h"].le(-large_move).astype(float).where(dataframe[f"&-future_max_drawdown_{horizon}h"].notna())
            )
        dataframe = self._append_structure_path_labels(dataframe)
        target_columns = [column for column in dataframe.columns if column.startswith("&-")]
        mask = pd.Series(True, index=dataframe.index)
        if self.require_context_available:
            mask &= pd.to_numeric(dataframe.get("fmt_context_ready"), errors="coerce").fillna(0.0).gt(0.0)
        if self.require_live_media_available:
            mask &= pd.to_numeric(dataframe.get("fmt_live_media_ready"), errors="coerce").fillna(0.0).gt(0.0)
        if self.require_orderbook_available:
            mask &= pd.to_numeric(dataframe.get("fmt_orderbook_ready"), errors="coerce").fillna(0.0).gt(0.0)
        if self.require_orderbook_loose_available:
            mask &= pd.to_numeric(dataframe.get("fmt_orderbook_loose_ready"), errors="coerce").fillna(0.0).gt(0.0)
        for column in target_columns:
            dataframe[column] = dataframe[column].where(mask)
        return dataframe

    @staticmethod
    def _path_thresholds(horizon: int) -> tuple[float, float, float]:
        if horizon <= 1:
            return 0.006, 0.004, 0.010
        if horizon <= 2:
            return 0.010, 0.006, 0.014
        if horizon <= 4:
            return 0.015, 0.010, 0.020
        if horizon <= 6:
            return 0.025, 0.015, 0.030
        return 0.025, 0.015, 0.040

    @staticmethod
    def _future_extreme(series: pd.Series, horizon: int, method: str) -> pd.Series:
        future = series.shift(-1).iloc[::-1]
        rolling = future.rolling(horizon, min_periods=horizon)
        result = rolling.max() if method == "max" else rolling.min()
        return result.iloc[::-1]

    @staticmethod
    def _reward_before_danger(
        close: pd.Series,
        *,
        horizon: int,
        reward: float,
        danger: float,
        long_side: bool,
    ) -> pd.Series:
        values = close.to_numpy(dtype=float)
        out = np.full(len(values), np.nan)
        for idx, current in enumerate(values):
            if not np.isfinite(current) or idx + horizon >= len(values):
                continue
            future = values[idx + 1 : idx + horizon + 1] / current - 1.0
            if long_side:
                reward_hits = np.where(future >= reward)[0]
                danger_hits = np.where(future <= danger)[0]
            else:
                reward_hits = np.where(future <= -reward)[0]
                danger_hits = np.where(future >= -danger)[0]
            first_reward = reward_hits[0] if reward_hits.size else None
            first_danger = danger_hits[0] if danger_hits.size else None
            if first_reward is None and first_danger is None:
                out[idx] = 0.0
            elif first_reward is None:
                out[idx] = 0.0
            elif first_danger is None:
                out[idx] = 1.0
            elif first_reward == first_danger:
                out[idx] = np.nan
            else:
                out[idx] = float(first_reward < first_danger)
        return pd.Series(out, index=close.index)

    @classmethod
    def _append_structure_path_labels(cls, dataframe: DataFrame) -> DataFrame:
        prior_high_24h = dataframe["high"].shift(1).rolling(24, min_periods=12).max()
        prior_low_24h = dataframe["low"].shift(1).rolling(24, min_periods=12).min()
        current_resistance_setup = (
            prior_high_24h.notna()
            & (dataframe["high"].ge(prior_high_24h * 0.995) | dataframe["close"].ge(prior_high_24h * 0.990))
        )
        current_support_setup = (
            prior_low_24h.notna()
            & (dataframe["low"].le(prior_low_24h * 1.005) | dataframe["close"].le(prior_low_24h * 1.010))
        )
        for horizon in (1, 2, 4, 6):
            future_high = cls._future_extreme(dataframe["high"], horizon, "max")
            future_low = cls._future_extreme(dataframe["low"], horizon, "min")
            future_close = dataframe["close"].shift(-horizon)
            breakout_attempt = future_high.ge(prior_high_24h * 1.001)
            breakdown_attempt = future_low.le(prior_low_24h * 0.999)
            dataframe[f"&-breakout_success_from_current_setup_{horizon}h"] = (
                current_resistance_setup & breakout_attempt & future_close.gt(prior_high_24h * 1.001)
            ).astype(float).where(prior_high_24h.notna() & future_close.notna())
            dataframe[f"&-breakout_failure_from_current_setup_{horizon}h"] = (
                current_resistance_setup & breakout_attempt & future_close.lt(prior_high_24h * 0.999)
            ).astype(float).where(prior_high_24h.notna() & future_close.notna())
            dataframe[f"&-breakdown_success_from_current_setup_{horizon}h"] = (
                current_support_setup & breakdown_attempt & future_close.lt(prior_low_24h * 0.999)
            ).astype(float).where(prior_low_24h.notna() & future_close.notna())
            dataframe[f"&-breakdown_failure_from_current_setup_{horizon}h"] = (
                current_support_setup & breakdown_attempt & future_close.gt(prior_low_24h * 1.001)
            ).astype(float).where(prior_low_24h.notna() & future_close.notna())
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe.get("do_predict", 0) == 1,
            dataframe.get("&-future_return_6h", 0) > 0,
        ]
        if conditions:
            dataframe.loc[
                reduce(lambda left, right: left & right, conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "freqai_research_probe_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe.get("do_predict", 0) == 1,
            dataframe.get("&-future_return_6h", 0) < 0,
        ]
        if conditions:
            dataframe.loc[
                reduce(lambda left, right: left & right, conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "freqai_research_probe_exit")
        return dataframe


class FormattedFreqAIPriceOnlyHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = ("fs_price_volume",)


class FormattedFreqAIContextHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    require_context_available = True


class FormattedFreqAIOrderbookHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_orderbook_transforms",
        "fs_orderbook_pressure_transforms",
        "fs_orderbook_liquidity_transforms",
        "fs_orderbook_wall_transforms",
        "fs_orderbook_pressure_walls",
        "fs_orderbook_state",
    )
    require_orderbook_available = True


class FormattedFreqAICrossConfluenceHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_escalation",
        "fs_context_state",
        "fs_orderbook_pressure_walls",
        "fs_orderbook_state",
        "fs_cross_confluence",
        "fs_market_breadth",
    )
    require_context_available = True
    require_orderbook_available = True


class FormattedFreqAIMarketBreadthHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_market_breadth",
    )
    include_market_raw = True


class FormattedFreqAIContextCompactRiskHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_escalation",
        "fs_context_state",
    )
    require_context_available = True


class FormattedFreqAIContextTopicRiskHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|risk|conflict|shock|war|geopolit|sanction|oil|energy|bank|credit|rate|inflation|central|recession|macro|liquid|stablecoin|regulat|security|exploit|dollar|systemic|weighted",
    )
    require_context_available = True


class FormattedFreqAIContextCryptoNativeHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|crypto|bitcoin|btc|ethereum|eth|etf|institution|stablecoin|liquidat|security|exploit|regulat|exchange|risk|shock",
    )
    require_context_available = True


class FormattedFreqAIContextActivityHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|article|source|doc|file|theme|word|mention|news|web|gkg|gdelt|coverage|available|safe",
    )
    require_context_available = True


class FormattedFreqAIMarketCompactHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_market_breadth",
    )


class FormattedFreqAIMarketRawBreadthHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = ("fs_price_volume",)
    include_market_raw = True
    feature_include_patterns = (
        r"price|volume|return|mkt_.*(ret|breadth|pressure|risk|dominance|alt|eth|sol)",
    )


class FormattedFreqAIMarketVolatilityHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = ("fs_price_volume",)
    include_market_raw = True
    feature_include_patterns = (
        r"price|volume|return|mkt_.*(vol|compression|expansion|range|atr|true_range)",
    )


class FormattedFreqAILiveMediaHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends",)
    require_live_media_available = True


class FormattedFreqAILiveMediaTopicRiskHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|news|web|global|source|risk|conflict|war|geopolit|sanction|oil|energy|bank|credit|rate|inflation|central|official|recession|jobs|liquid|stablecoin|regulat|etf|institution|security|exploit|exchange|crypto|shock|attention|escalation",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends",)
    require_live_media_available = True


class FormattedFreqAILiveMediaActivityHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|news|web|global|article|row|source|same_topic|unique_source|coverage|activity|acceleration|count|z_24h|rising|falling|cross",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends",)
    require_live_media_available = True


class FormattedFreqAILiveNewsActivityHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|ctx_news|news_",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends|ctx_web|ctx_global",)
    require_live_media_available = True


class FormattedFreqAILiveWebActivityHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|ctx_web|web_",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends|ctx_news|ctx_global",)
    require_live_media_available = True


class FormattedFreqAILiveGlobalMacroHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|ctx_global|global_|market_cap|fear|equity|macro",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends|ctx_news|ctx_web",)
    require_live_media_available = True


class FormattedFreqAILiveMediaTraderShapesHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|conf_context|state_context",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends",)
    require_live_media_available = True
    include_live_media_trader_shapes = True


class FormattedFreqAILiveMediaTopicRiskTraderShapesHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|news|web|global|source|risk|conflict|war|geopolit|sanction|oil|energy|bank|credit|rate|inflation|central|official|recession|jobs|liquid|stablecoin|regulat|etf|institution|security|exploit|exchange|crypto|shock|attention|escalation",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends",)
    require_live_media_available = True
    include_live_media_trader_shapes = True


class FormattedFreqAILiveMediaTraderShapesVpHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_escalation",
        "fs_context_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|conf_context|state_context",
    )
    feature_exclude_patterns = (r"gdelt|gkg|orderbook|ob1h_|google_trends",)
    require_live_media_available = True
    include_live_media_trader_shapes = True
    include_volume_profile_features = True


class FormattedFreqAILiveMediaPlusOrderbookHypothesisStrategy(FormattedContextHypothesisFreqAIBaseStrategy):
    feature_panels = (
        "fs_price_volume",
        "fs_context_transforms",
        "fs_context_escalation",
        "fs_context_state",
        "fs_orderbook_transforms",
        "fs_orderbook_pressure_transforms",
        "fs_orderbook_liquidity_transforms",
        "fs_orderbook_wall_transforms",
        "fs_orderbook_pressure_walls",
        "fs_orderbook_state",
    )
    feature_include_patterns = (
        r"price|volume|return|px_|news|web|global|source|risk|conflict|crypto|shock|attention|escalation|binance|pressure|wall|spread|microprice|imbalance|liquidity|fragility|vacuum",
    )
    feature_exclude_patterns = (r"gdelt|gkg|google_trends|bybit",)
    require_live_media_available = True
    require_orderbook_available = True


class FormattedFreqAILiveMediaPlusLooseOrderbookHypothesisStrategy(FormattedFreqAILiveMediaPlusOrderbookHypothesisStrategy):
    require_orderbook_available = False
    require_orderbook_loose_available = True
