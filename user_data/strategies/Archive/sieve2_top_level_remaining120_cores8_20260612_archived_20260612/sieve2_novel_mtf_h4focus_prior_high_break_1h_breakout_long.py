from __future__ import annotations
SIEVE_STAGE = "sieve2"
NOVEL_IDEA = True
SOURCE_STRATEGY = "novel"
SOURCE_RESULT_BATCH = "manual_20260606"
RESEARCH_PATH = "mtf_context_1h_execution_h4_focus"
UPDATE_HYPOTHESIS = "Entry-only H4 prior-high break context with 1h breakout execution; tests whether 4h continuation can be entered earlier on 1h."

import os
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from user_data.strategies.entry_sieve_tools import entry_sieve_minimal_roi, entry_sieve_stoploss

SIDE = "long"
HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"


def split_hyperopt_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")
    return {token.strip() for token in normalized.split(",") if token.strip()}


def is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def apply_explicit_hyperopt_surface(strategy_cls: type) -> None:
    selected = split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))
    if not selected:
        return
    for name in dir(strategy_cls):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = str(name) in selected


def tagged_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:entries", "mode:sieve2_novel_mtf_h4focus"))
    return param


CONTEXT_MODE = "h4_prior_high_break"
EXECUTION_MODE = "breakout"

class Sieve2NovelMtfH4focusPriorHighBreak1hBreakoutLong(IStrategy):
    timeframe = "1h"
    can_short = True
    startup_candle_count = 260
    minimal_roi = entry_sieve_minimal_roi(0.03)
    stoploss = entry_sieve_stoploss(-0.03)
    process_only_new_candles = True
    use_exit_signal = False
    INTERFACE_VERSION = 3

    context_lookback = tagged_parameter(IntParameter(8, 40, default=20, space="buy", optimize=True, load=True))
    h4_lookback = tagged_parameter(IntParameter(4, 48, default=16, space="buy", optimize=True, load=True))
    exec_lookback = tagged_parameter(IntParameter(3, 30, default=8, space="buy", optimize=True, load=True))
    body_ratio_min = tagged_parameter(DecimalParameter(0.10, 0.85, default=0.35, decimals=2, space="buy", optimize=True, load=True))
    range_ratio_min = tagged_parameter(DecimalParameter(0.55, 2.80, default=1.10, decimals=2, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(DecimalParameter(0.50, 2.80, default=1.05, decimals=2, space="buy", optimize=True, load=True))
    retest_tolerance = tagged_parameter(DecimalParameter(0.001, 0.030, default=0.008, decimals=3, space="buy", optimize=True, load=True))
    close_follow_min = tagged_parameter(DecimalParameter(0.00, 0.65, default=0.15, decimals=2, space="buy", optimize=True, load=True))
    require_h4_confirm = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))
    require_volume_confirm = tagged_parameter(CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True))

    def informative_pairs(self):
        dp = getattr(self, "dp", None)
        if dp is None:
            return []
        try:
            pairs = self.dp.current_whitelist()
        except Exception:
            pairs = []
        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = dataframe.copy()
        dp = getattr(self, "dp", None)
        if dp is not None and metadata and metadata.get("pair"):
            for tf in ("4h", "1d"):
                informative = dp.get_pair_dataframe(pair=metadata["pair"], timeframe=tf)
                if informative is not None and not informative.empty:
                    dataframe = merge_informative_pair(dataframe, self._features(informative.copy(), tf), self.timeframe, tf, ffill=True)
        return self._features(dataframe, "1h")

    def _features(self, frame: DataFrame, tf: str) -> DataFrame:
        close = pd.to_numeric(frame["close"], errors="coerce")
        open_ = pd.to_numeric(frame["open"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        look = int(self.context_lookback.value if tf == "1d" else self.h4_lookback.value if tf == "4h" else self.exec_lookback.value)
        rng = (high - low).replace(0, np.nan)
        body = close - open_
        frame[f"body_ratio_{tf}"] = (body.abs() / rng).replace([np.inf, -np.inf], np.nan)
        frame[f"prior_high_{tf}"] = high.shift(1).rolling(look, min_periods=max(2, look // 3)).max()
        frame[f"prior_low_{tf}"] = low.shift(1).rolling(look, min_periods=max(2, look // 3)).min()
        frame[f"range_mid_{tf}"] = (frame[f"prior_high_{tf}"] + frame[f"prior_low_{tf}"]) / 2.0
        frame[f"range_ratio_{tf}"] = rng / rng.rolling(look, min_periods=max(2, look // 3)).mean().replace(0, np.nan)
        frame[f"volume_ratio_{tf}"] = volume / volume.rolling(look, min_periods=max(2, look // 3)).mean().replace(0, np.nan)
        frame[f"hh_{tf}"] = high.gt(high.shift(1).rolling(look, min_periods=max(2, look // 3)).max())
        frame[f"ll_{tf}"] = low.lt(low.shift(1).rolling(look, min_periods=max(2, look // 3)).min())
        frame[f"hl_{tf}"] = low.gt(low.shift(1).rolling(max(2, look // 2), min_periods=2).min())
        frame[f"lh_{tf}"] = high.lt(high.shift(1).rolling(max(2, look // 2), min_periods=2).max())
        return frame

    def _context(self, dataframe: DataFrame) -> Series:
        c1 = self._num(dataframe, "close_1d"); o1 = self._num(dataframe, "open_1d"); h1 = self._num(dataframe, "high_1d"); l1 = self._num(dataframe, "low_1d")
        c4 = self._num(dataframe, "close_4h"); o4 = self._num(dataframe, "open_4h"); h4 = self._num(dataframe, "high_4h"); l4 = self._num(dataframe, "low_4h")
        bull1 = c1.gt(o1) & self._num(dataframe, "body_ratio_1d").ge(float(self.body_ratio_min.value))
        bear1 = c1.lt(o1) & self._num(dataframe, "body_ratio_1d").ge(float(self.body_ratio_min.value))
        bull4 = c4.gt(o4); bear4 = c4.lt(o4)
        vol_ok = self._num(dataframe, "volume_ratio_1d").ge(float(self.volume_ratio_min.value)) | ~self._bool_param(self.require_volume_confirm)
        h4_long_ok = bull4 | ~self._bool_param(self.require_h4_confirm); h4_short_ok = bear4 | ~self._bool_param(self.require_h4_confirm)
        mode = CONTEXT_MODE
        if mode == "prior_high_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & bull1
        elif mode == "failed_low_reclaim": ctx = l1.lt(self._num(dataframe,"prior_low_1d")) & c1.gt(self._num(dataframe,"prior_low_1d")) & bull1
        elif mode == "trend_hhhl": ctx = self._bool(dataframe,"hh_1d") & self._bool(dataframe,"hl_1d") & c1.gt(self._num(dataframe,"range_mid_1d"))
        elif mode == "range_expansion": ctx = bull1 & self._num(dataframe,"range_ratio_1d").ge(float(self.range_ratio_min.value))
        elif mode == "midline_reclaim": ctx = c1.gt(self._num(dataframe,"range_mid_1d")) & c1.shift(1).le(self._num(dataframe,"range_mid_1d"))
        elif mode == "compression_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & self._num(dataframe,"range_ratio_1d").le(float(self.range_ratio_min.value))
        elif mode == "volume_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & self._num(dataframe,"volume_ratio_1d").ge(float(self.volume_ratio_min.value))
        elif mode == "h4_prior_high_break": ctx = c4.gt(self._num(dataframe,"prior_high_4h")) & bull4
        elif mode == "h4_hl_reclaim": ctx = self._bool(dataframe,"hl_4h") & c4.gt(self._num(dataframe,"range_mid_4h"))
        elif mode == "h4_failed_low_reclaim": ctx = l4.lt(self._num(dataframe,"prior_low_4h")) & c4.gt(self._num(dataframe,"prior_low_4h"))
        elif mode == "h4_range_expansion": ctx = bull4 & self._num(dataframe,"range_ratio_4h").ge(float(self.range_ratio_min.value))
        elif mode == "h4_compression_break": ctx = c4.gt(self._num(dataframe,"prior_high_4h")) & self._num(dataframe,"range_ratio_4h").le(float(self.range_ratio_min.value))
        elif mode == "dual_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & c4.gt(self._num(dataframe,"prior_high_4h"))
        elif mode == "support_h4_break": ctx = c1.gt(self._num(dataframe,"range_mid_1d")) & c4.gt(self._num(dataframe,"prior_high_4h"))
        elif mode == "pullback_h4_reclaim": ctx = c1.gt(self._num(dataframe,"range_mid_1d")) & self._bool(dataframe,"hl_4h")
        elif mode == "prior_low_break": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & bear1
        elif mode == "failed_high_reject": ctx = h1.gt(self._num(dataframe,"prior_high_1d")) & c1.lt(self._num(dataframe,"prior_high_1d")) & bear1
        elif mode == "trend_lhll": ctx = self._bool(dataframe,"ll_1d") & self._bool(dataframe,"lh_1d") & c1.lt(self._num(dataframe,"range_mid_1d"))
        elif mode == "bear_range_expansion": ctx = bear1 & self._num(dataframe,"range_ratio_1d").ge(float(self.range_ratio_min.value))
        elif mode == "midline_reject": ctx = c1.lt(self._num(dataframe,"range_mid_1d")) & c1.shift(1).ge(self._num(dataframe,"range_mid_1d"))
        elif mode == "compression_breakdown": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & self._num(dataframe,"range_ratio_1d").le(float(self.range_ratio_min.value))
        elif mode == "volume_breakdown": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & self._num(dataframe,"volume_ratio_1d").ge(float(self.volume_ratio_min.value))
        elif mode == "h4_prior_low_break": ctx = c4.lt(self._num(dataframe,"prior_low_4h")) & bear4
        elif mode == "h4_lh_reject": ctx = self._bool(dataframe,"lh_4h") & c4.lt(self._num(dataframe,"range_mid_4h"))
        elif mode == "h4_failed_high_reject": ctx = h4.gt(self._num(dataframe,"prior_high_4h")) & c4.lt(self._num(dataframe,"prior_high_4h"))
        elif mode == "h4_bear_range_expansion": ctx = bear4 & self._num(dataframe,"range_ratio_4h").ge(float(self.range_ratio_min.value))
        elif mode == "h4_compression_breakdown": ctx = c4.lt(self._num(dataframe,"prior_low_4h")) & self._num(dataframe,"range_ratio_4h").le(float(self.range_ratio_min.value))
        elif mode == "dual_breakdown": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & c4.lt(self._num(dataframe,"prior_low_4h"))
        elif mode == "resistance_h4_break": ctx = c1.lt(self._num(dataframe,"range_mid_1d")) & c4.lt(self._num(dataframe,"prior_low_4h"))
        elif mode == "bounce_h4_reject": ctx = c1.lt(self._num(dataframe,"range_mid_1d")) & self._bool(dataframe,"lh_4h")
        else: ctx = pd.Series(False, index=dataframe.index)
        return (ctx & (h4_long_ok if SIDE == "long" else h4_short_ok) & vol_ok).fillna(False)

    def _execution(self, dataframe: DataFrame) -> Series:
        close = self._num(dataframe,"close"); open_ = self._num(dataframe,"open"); high = self._num(dataframe,"high"); low = self._num(dataframe,"low")
        ph = self._num(dataframe,"prior_high_1h"); pl = self._num(dataframe,"prior_low_1h"); mid = self._num(dataframe,"range_mid_1h")
        tol = float(self.retest_tolerance.value); follow = float(self.close_follow_min.value)
        bull = close.gt(open_) & self._num(dataframe,"body_ratio_1h").ge(follow)
        bear = close.lt(open_) & self._num(dataframe,"body_ratio_1h").ge(follow)
        if EXECUTION_MODE == "breakout": exe = close.gt(ph) & bull
        elif EXECUTION_MODE == "breakdown": exe = close.lt(pl) & bear
        elif EXECUTION_MODE == "retest" and SIDE == "long": exe = low.le(ph*(1+tol)) & close.gt(ph) & bull
        elif EXECUTION_MODE == "retest" and SIDE == "short": exe = high.ge(pl*(1-tol)) & close.lt(pl) & bear
        elif EXECUTION_MODE == "bos" and SIDE == "long": exe = close.gt(ph) & self._bool(dataframe,"hl_1h")
        elif EXECUTION_MODE == "bos" and SIDE == "short": exe = close.lt(pl) & self._bool(dataframe,"lh_1h")
        elif EXECUTION_MODE == "higher_low_break": exe = self._bool(dataframe,"hl_1h") & close.gt(mid) & bull
        elif EXECUTION_MODE == "lower_high_break": exe = self._bool(dataframe,"lh_1h") & close.lt(mid) & bear
        elif EXECUTION_MODE == "reclaim": exe = low.lt(mid) & close.gt(mid) & bull
        elif EXECUTION_MODE == "reject": exe = high.gt(mid) & close.lt(mid) & bear
        else: exe = pd.Series(False, index=dataframe.index)
        return exe.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        condition = self._context(dataframe) & self._execution(dataframe)
        if SIDE == "long": dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, f"{CONTEXT_MODE}_1h_{EXECUTION_MODE}_long")
        else: dataframe.loc[condition, ["enter_short", "enter_tag"]] = (1, f"{CONTEXT_MODE}_1h_{EXECUTION_MODE}_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        return dataframe

    @staticmethod
    def _column_alias(frame: DataFrame, column: str) -> str | None:
        if column in frame.columns:
            return column
        for suffix in ("_1d", "_4h"):
            if column.endswith(suffix):
                merged = f"{column}{suffix}"
                if merged in frame.columns:
                    return merged
        return None

    @staticmethod
    def _num(frame: DataFrame, column: str, default: float = 0.0) -> Series:
        resolved = Sieve2NovelMtfBaseMixin._column_alias(frame, column) if "Sieve2NovelMtfBaseMixin" in globals() else None
        if resolved is None:
            if column in frame.columns:
                resolved = column
            else:
                for suffix in ("_1d", "_4h"):
                    if column.endswith(suffix) and f"{column}{suffix}" in frame.columns:
                        resolved = f"{column}{suffix}"
                        break
        if resolved is None:
            return pd.Series(default, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[resolved], errors="coerce").replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _bool(frame: DataFrame, column: str) -> Series:
        resolved = column if column in frame.columns else None
        if resolved is None:
            for suffix in ("_1d", "_4h"):
                if column.endswith(suffix) and f"{column}{suffix}" in frame.columns:
                    resolved = f"{column}{suffix}"
                    break
        if resolved is None:
            return pd.Series(False, index=frame.index, dtype="bool")
        return pd.Series(frame[resolved], index=frame.index).astype("boolean").fillna(False).astype(bool)

    @staticmethod
    def _bool_param(param: Any) -> bool:
        value = getattr(param, "value", param)
        if isinstance(value, str): return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

apply_explicit_hyperopt_surface(Sieve2NovelMtfH4focusPriorHighBreak1hBreakoutLong)
