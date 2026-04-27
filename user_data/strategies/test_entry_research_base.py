from __future__ import annotations

from datetime import datetime
import re
from typing import Any

import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import CategoricalParameter

from entry_sieve_tools import apply_explicit_hyperopt_surface, entry_sieve_minimal_roi, entry_sieve_stoploss


def _is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def _mode_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    return token.strip("_") or "entry_test_mode"


def _tag_parameter(param: Any, family: str, mode: str) -> None:
    setattr(param, "batch_tags", (f"family:{family}", f"mode:{mode}"))


def _ensure_minimum_mode_surface(cls: type, family: str, mode: str) -> None:
    tagged_locals = 0
    for _, value in vars(cls).items():
        if not _is_parameter_object(value):
            continue
        tags = getattr(value, "batch_tags", ()) or ()
        has_family = any(str(t) == f"family:{family}" for t in tags)
        has_mode = any(str(t) == f"mode:{mode}" for t in tags)
        if has_family and has_mode:
            tagged_locals += 1
    if tagged_locals >= 2:
        return
    companion = CategoricalParameter(
        ["base", "with_exits"],
        default="base",
        space="buy",
        optimize=False,
        load=False,
    )
    _tag_parameter(companion, family, mode)
    setattr(cls, "_mode_pairing_profile", companion)


class FixedExitResearchMixin:
    """
    Shared first-pass research baseline.

    Keep trade management deliberately plain so backtests compare entry quality:
    fixed ROI, fixed stop, no signal exits, no custom stop, no leverage, no adds.
    """

    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_time, current_rate, min_stake, leverage, entry_tag, side, kwargs
        return float(min(float(proposed_stake or 0.0), float(max_stake or proposed_stake or 0.0)))

    def adjust_trade_position(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        return None

    def custom_exit(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        return None


class LadderSingleEntryMixin(FixedExitResearchMixin):
    entry_mask_method = ""
    entry_side = "long"
    entry_tag = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mode = f"entry_{_mode_token(getattr(cls, 'entry_tag', ''))}"
        for _, value in vars(cls).items():
            if _is_parameter_object(value):
                _tag_parameter(value, "entries", mode)
        _ensure_minimum_mode_surface(cls, "entries", mode)
        apply_explicit_hyperopt_surface(cls)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        mask_method = getattr(self, self.entry_mask_method)
        entry_mask = pd.Series(mask_method(dataframe), index=dataframe.index).fillna(False).astype(bool)
        plot_col = f"plot_{self.entry_tag}"
        dataframe[plot_col] = entry_mask.astype(float)

        if self.entry_side == "short":
            dataframe.loc[entry_mask, "enter_short"] = 1
        else:
            dataframe.loc[entry_mask, "enter_long"] = 1
        dataframe.loc[entry_mask, "enter_tag"] = self.entry_tag
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


class PivotSingleEntryMixin(FixedExitResearchMixin):
    allowed_entry_tag = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mode = f"entry_{_mode_token(getattr(cls, 'allowed_entry_tag', ''))}"
        for _, value in vars(cls).items():
            if _is_parameter_object(value):
                _tag_parameter(value, "entries", mode)
        _ensure_minimum_mode_surface(cls, "entries", mode)
        apply_explicit_hyperopt_surface(cls)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        tag = str(self.allowed_entry_tag)
        keep = dataframe["enter_tag"].fillna("").astype(str).eq(tag)
        dataframe.loc[~keep, ["enter_long", "enter_short"]] = 0
        dataframe.loc[~keep, "enter_tag"] = ""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        return dataframe
