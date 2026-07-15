from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class _MtfIndicatorMiningBase(TraderRuleBlockResearchStrategy):
    can_short = True
    use_custom_stoploss = True
    stoploss = -0.020

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


class TraderRuleBlockMtf4hTlv2ResBreakLongStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of one mined MTF indicator lead.

    Trader question:
    When price breaks 4h TLV2 resistance with bullish pressure, does the
    breakout continue?
    """

    can_short = False
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_4h_tlv2_res_break_long_thr_045.parquet"
    )
    rule_exit_settings = {
        "mtf_4h_tlv2_res_break_long": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020}
    }


class TraderRuleBlockMtf1dTlv2ResRetestLongStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of a mined 1d TLV2 retest lead.

    Trader question:
    After breaking daily TLV2 resistance, does a retest that holds continue?
    """

    can_short = False
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_1d_tlv2_res_retest_long_thr_065.parquet"
    )
    rule_exit_settings = {
        "mtf_1d_tlv2_res_retest_long": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.020}
    }


class TraderRuleBlockMtf4hTriangleUpperBreakLongStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of a mined 4h triangle breakout lead.

    Trader question:
    When a 4h triangle breaks upward with volume, does it follow through?
    """

    can_short = False
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_4h_triangle_upper_break_long_thr_045.parquet"
    )
    rule_exit_settings = {
        "mtf_4h_triangle_upper_break_long": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020}
    }


class TraderRuleBlockMtf4hVpValAcceptShortStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of a mined 4h VP value-loss lead.

    Trader question:
    When price accepts below 4h VAL with bearish pressure, does it continue
    toward lower value?
    """

    can_short = True
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_4h_vp_val_accept_short_thr_055.parquet"
    )
    rule_exit_settings = {
        "mtf_4h_vp_val_accept_short": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020}
    }


class TraderRuleBlockMtf1dRectangleLowerBreakShortStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of a mined daily rectangle breakdown lead.

    Trader question:
    When a daily rectangle breaks downward with volume, does it follow through?
    """

    can_short = True
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_1d_rectangle_lower_break_short_thr_035.parquet"
    )
    rule_exit_settings = {
        "mtf_1d_rectangle_lower_break_short": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020}
    }


class TraderRuleBlockMtf1dTlv2SupRetestShortStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of a mined daily TLV2 support-retest short lead.

    Trader question:
    After breaking daily TLV2 support, does a retest rejection continue lower?
    """

    can_short = True
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_1d_tlv2_sup_retest_short_thr_045.parquet"
    )
    rule_exit_settings = {
        "mtf_1d_tlv2_sup_retest_short": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.020}
    }


class TraderRuleBlockMtfValidatedConfluenceStrategy(_MtfIndicatorMiningBase):
    """
    Research-only validation of the MTF leads after per-rule confluence filters.

    Trader question:
    If the mined MTF entries are only taken when the matching structure,
    volume, or orderbook filter agrees, does trade quality improve?
    """

    can_short = True
    stoploss = -0.020
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_validated_confluence_dedup_fix_validated_exit_selected_confluence.parquet"
    )
    rule_exit_settings = {
        "mtf_4h_tlv2_res_break_long": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020},
        "mtf_1d_tlv2_res_retest_long": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.020},
        "mtf_4h_triangle_upper_break_long": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020},
        "mtf_4h_vp_val_accept_short": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020},
        "mtf_1d_rectangle_lower_break_short": {"take_profit": 0.050, "hold_hours": 12.0, "stop_loss": 0.020},
        "mtf_1d_tlv2_sup_retest_short": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.020},
    }


class TraderRuleBlockMtfBestPerRuleConfluenceStrategy(TraderRuleBlockMtfValidatedConfluenceStrategy):
    """
    Research-only validation of the single best confluence filter per MTF rule.

    Trader question:
    If each MTF idea only uses its own strongest confluence filter, does the
    smaller cleaner block beat the broader filtered block?
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_validated_confluence_dedup_fix_best_per_rule_selected_confluence.parquet"
    )


class TraderRuleBlockMtfBestPerRuleTailoredExitStrategy(TraderRuleBlockMtfBestPerRuleConfluenceStrategy):
    """
    Research-only validation of the MTF best-per-rule confluence block with
    exit settings selected by the exit-sweep tool.

    Trader question:
    If entry logic is unchanged, do exits tailored to each MTF market story
    improve trade quality versus crude fixed exits?
    """

    rule_exit_settings = {
        "mtf_4h_tlv2_res_break_long": {"take_profit": 0.050, "hold_hours": 36.0, "stop_loss": 0.014},
        "mtf_4h_triangle_upper_break_long": {"take_profit": 0.020, "hold_hours": 12.0, "stop_loss": 0.020},
        "mtf_4h_vp_val_accept_short": {"take_profit": 0.050, "hold_hours": 36.0, "stop_loss": 0.010},
        "mtf_1d_rectangle_lower_break_short": {"take_profit": 0.075, "hold_hours": 36.0, "stop_loss": 0.010},
        "mtf_1d_tlv2_sup_retest_short": {"take_profit": 0.050, "hold_hours": 72.0, "stop_loss": 0.035},
    }


class TraderRuleBlockMtfBestPerRuleStorySvRiskStrategy(TraderRuleBlockMtfBestPerRuleTailoredExitStrategy):
    """
    Research-only validation of the MTF tailored block with story-specific
    structure/volume position sizing.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_best_story_sv_risk.parquet"
    )


class TraderRuleBlockMtfBestPerRuleStorySvExtremeObRiskStrategy(TraderRuleBlockMtfBestPerRuleTailoredExitStrategy):
    """
    Research-only validation of the MTF tailored block with story-specific
    structure/volume sizing plus an extreme orderbook invalidation brake.
    """

    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_mtf_best_story_sv_extreme_ob_risk.parquet"
    )
