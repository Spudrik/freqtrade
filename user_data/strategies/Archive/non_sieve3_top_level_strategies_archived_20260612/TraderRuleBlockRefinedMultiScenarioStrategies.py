from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class _RefinedMultiScenarioBase(TraderRuleBlockResearchStrategy):
    can_short = True
    use_custom_stoploss = True
    stoploss = -0.025

    rule_exit_settings: dict[str, dict[str, float]] = {
        "lead__validated__support_break_support_cleared_short": {
            "take_profit": 0.050,
            "hold_hours": 18.0,
            "stop_loss": 0.025,
        },
        "lead__validated__val_vacuum_no_support_reclaim_warning_short": {
            "take_profit": 0.035,
            "hold_hours": 8.0,
            "stop_loss": 0.010,
        },
        "sieve_prior_day_low_break_vp_short": {
            "take_profit": 0.020,
            "hold_hours": 48.0,
            "stop_loss": 0.020,
        },
        "rect_break_orderbook_accept_long": {
            "take_profit": 0.055,
            "hold_hours": 24.0,
            "stop_loss": 0.025,
        },
        "rect_squeeze_upper_release_long": {
            "take_profit": 0.055,
            "hold_hours": 24.0,
            "stop_loss": 0.025,
        },
    }

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        _ = pair, current_time, current_rate, after_fill, kwargs
        raw_tag = getattr(trade, "enter_tag", None) or getattr(trade, "entry_tag", None) or ""
        rule_id = str(raw_tag).split()[0]
        settings = self.rule_exit_settings.get(rule_id)
        if settings is None:
            return None
        stop_loss = float(settings.get("stop_loss", abs(self.stoploss)))
        return stoploss_from_open(-stop_loss, current_profit, is_short=trade.is_short, leverage=trade.leverage)


class TraderRuleBlockRefinedNoVahMultiScenarioStrategy(_RefinedMultiScenarioBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_refined_multi_scenario_no_vah.parquet"
    )


class TraderRuleBlockRefinedSqueezeOnlyMultiScenarioStrategy(_RefinedMultiScenarioBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_refined_multi_scenario_squeeze_only.parquet"
    )


class TraderRuleBlockRefinedSqueezePriorityMultiScenarioStrategy(_RefinedMultiScenarioBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah.parquet"
    )


class TraderRuleBlockRefinedAcceptOnlyMultiScenarioStrategy(_RefinedMultiScenarioBase):
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_refined_multi_scenario_accept_only.parquet"
    )


class TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy(TraderRuleBlockRefinedSqueezePriorityMultiScenarioStrategy):
    """
    Research-only validation of the refined squeeze-priority block with the
    best exit settings selected by the latest exit sweep.

    Trader question:
    If the entry block is unchanged, do per-lead exits improve quality versus
    the current fixed exits?
    """

    rule_exit_settings: dict[str, dict[str, float]] = {
        "lead__validated__support_break_support_cleared_short": {
            "take_profit": 0.050,
            "hold_hours": 18.0,
            "stop_loss": 0.025,
        },
        "lead__validated__val_vacuum_no_support_reclaim_warning_short": {
            "take_profit": 0.035,
            "hold_hours": 8.0,
            "stop_loss": 0.010,
        },
        "sieve_prior_day_low_break_vp_short": {
            "take_profit": 0.050,
            "hold_hours": 48.0,
            "stop_loss": 0.035,
        },
        "rect_break_orderbook_accept_long": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.035,
        },
        "rect_squeeze_upper_release_long": {
            "take_profit": 0.020,
            "hold_hours": 72.0,
            "stop_loss": 0.025,
        },
    }


class TraderRuleBlockRefinedSqueezePriorityStorySvRiskStrategy(TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy):
    """
    Research-only validation of the refined tailored block with story-specific
    structure/volume position sizing.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_refined_squeeze_priority_story_sv_risk.parquet"
    )


class TraderRuleBlockRefinedSqueezePriorityStorySvExtremeObRiskStrategy(TraderRuleBlockRefinedSqueezePriorityTailoredExitStrategy):
    """
    Research-only validation of the refined tailored block with story-specific
    structure/volume sizing plus an extreme orderbook invalidation brake.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_refined_squeeze_priority_story_sv_extreme_ob_risk.parquet"
    )
