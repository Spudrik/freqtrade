from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import stoploss_from_open

from TraderRuleBlockResearchStrategy import TraderRuleBlockResearchStrategy


class TraderRuleBlockBtcSieveCompressionRangeStrategy(TraderRuleBlockResearchStrategy):
    """
    Research-only validation of BTC sieve leads after the strongest broad
    confluence filter found so far.

    Trader question:
    If proven BTC sieve entries are only taken when compression/range-break
    context agrees, and each lead family gets its own stop/target/hold exit,
    does the block remain useful in a normal Freqtrade backtest?
    """

    can_short = True
    use_custom_stoploss = True
    stoploss = -0.035
    signal_file = (
        r"C:\FreqTradeStuff\user_data\research_news_data\context_features\reports"
        r"\trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_selected_exit_rules.parquet"
    )

    rule_exit_settings: dict[str, dict[str, float]] = {
        "sieve2_overtrade_tlv2_resistance_breakout_long_1h_vp_market_guard": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.018,
        },
        "sieve2_bos_bull_continuation_long_1h": {"take_profit": 0.050, "hold_hours": 72.0, "stop_loss": 0.035},
        "sieve1_bos_bull_continuation_long_1h": {"take_profit": 0.050, "hold_hours": 72.0, "stop_loss": 0.012},
        "sieve2_multi2_vp_prior_month_high_break_vp_val_long": {
            "take_profit": 0.035,
            "hold_hours": 48.0,
            "stop_loss": 0.025,
        },
        "sieve2_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard": {
            "take_profit": 0.050,
            "hold_hours": 48.0,
            "stop_loss": 0.018,
        },
        "sieve2_multi2_vp_prior_month_high_break_vp_node_long": {
            "take_profit": 0.035,
            "hold_hours": 48.0,
            "stop_loss": 0.025,
        },
        "sieve1_vp_lvn_fast_traverse_long_1h": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.035},
        "sieve2_multi2_tlv2_vp_res_break_vp_val_long_4h": {
            "take_profit": 0.050,
            "hold_hours": 24.0,
            "stop_loss": 0.035,
        },
        "sieve1_tlv2_resistance_breakout_long_4h": {"take_profit": 0.050, "hold_hours": 36.0, "stop_loss": 0.035},
        "sieve2_overtrade_multi2_tlv2_vp_res_break_vp_val_long_1h_vp_market_guard": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.012,
        },
        "sieve1_multi2_tlv2_vp_res_break_vp_val_long_8h": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.025,
        },
        "sieve2_reframed_vp_poc_reject_short_1h": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.012},
        "sieve1_bos_bull_continuation_long_8h": {"take_profit": 0.050, "hold_hours": 48.0, "stop_loss": 0.025},
        "sieve2_multi2_tlv2_vp_res_break_vp_bullctx_long_1h": {
            "take_profit": 0.050,
            "hold_hours": 36.0,
            "stop_loss": 0.012,
        },
        "sieve2_multi2_vp_prior_month_high_break_vp_bullctx_long": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.018,
        },
        "sieve2_reversal_double_top_present_short_1h": {
            "take_profit": 0.050,
            "hold_hours": 36.0,
            "stop_loss": 0.018,
        },
        "sieve2_overtrade_multi2_tlv2_vp_sup_reclaim_vp_node_long_1h_vp_market_guard": {
            "take_profit": 0.050,
            "hold_hours": 48.0,
            "stop_loss": 0.035,
        },
        "sieve2_reframed_mtf_4h_triangle_long_1h_breakout": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.012,
        },
        "sieve1_geometry_triangle_squeeze_breakout_long_4h": {
            "take_profit": 0.050,
            "hold_hours": 72.0,
            "stop_loss": 0.012,
        },
        "sieve1_multi2_tlv2_vp_res_break_vp_val_long_1d": {
            "take_profit": 0.020,
            "hold_hours": 72.0,
            "stop_loss": 0.025,
        },
        "sieve1_bos_bull_continuation_long_1d": {"take_profit": 0.050, "hold_hours": 24.0, "stop_loss": 0.018},
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
