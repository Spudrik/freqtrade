"""Separate PAPER tests of two losing signal families with opposite entries.

These are prospective counter-hypotheses, not repairs to or replacements for the
frozen original accounts.  The leader test reverses both impulse directions.
The fast test reverses only calculated-level bounces, not break/retests.
"""

from __future__ import annotations

from freqtrade.enums import RunMode

from user_data.strategies.paper_fast_reaction import PaperFastAuto
from user_data.strategies.paper_leader_impulse import PaperLeaderImpulse


class PaperLeaderInverse(PaperLeaderImpulse):
    """Enter against a completed BTC impulse plus same-side local break."""

    def bot_start(self, **kwargs) -> None:
        _ = kwargs
        exchange = self.config.get("exchange", {})
        if (self.config.get("dry_run") is not True
                or self.config.get("runmode") != RunMode.DRY_RUN
                or self.config.get("trading_mode") != "futures"
                or self.config.get("margin_mode") != "isolated"
                or exchange.get("name") != "binance"
                or any(exchange.get(key) for key in ("key", "secret", "password", "privateKey"))
                or self.config.get("bot_name") != "paper_leader_inverse"
                or self.config.get("db_url") != (
                    "sqlite:///user_data/research_news_data/context_features/"
                    "integrated_paper_20260926/leader_inverse_trades.sqlite"
                )
                or self.config.get("force_entry_enable")
                or self.config.get("api_server", {}).get("enabled")):
            raise RuntimeError("Inverse leader strategy requires its isolated PAPER account")

    def populate_entry_trend(self, dataframe, metadata):
        frame = super().populate_entry_trend(dataframe, metadata)
        original_long = frame["enter_long"].eq(1)
        original_short = frame["enter_short"].eq(1)
        frame["enter_long"] = original_short.astype(int)
        frame["enter_short"] = original_long.astype(int)
        frame.loc[original_short, "enter_tag"] = "inverse_btc_impulse_local_break_long"
        frame.loc[original_long, "enter_tag"] = "inverse_btc_impulse_local_break_short"
        return frame


class PaperFastLevelInverse(PaperFastAuto):
    """Reverse only level bounces; invalidate beyond the signal candle extreme."""

    account_key = "fast_level_inverse"

    def populate_entry_trend(self, dataframe, metadata):
        frame = super().populate_entry_trend(dataframe, metadata)
        original_long = frame["enter_tag"].eq("fast_level_long")
        original_short = frame["enter_tag"].eq("fast_level_short")
        frame["enter_long"] = original_short.astype(int)
        frame["enter_short"] = original_long.astype(int)
        frame.loc[original_short, "enter_tag"] = "inverse_fast_level_long"
        frame.loc[original_long, "enter_tag"] = "inverse_fast_level_short"
        # The original calculated level selects the opportunity.  For the
        # opposite trade, a move beyond the signal candle's adverse extreme
        # invalidates the reversal thesis.  The inherited sizing and exits then
        # apply the same ATR risk/target/time rules to the actual trade side.
        frame.loc[original_short, "fast_trigger_level"] = frame.loc[original_short, "low"]
        frame.loc[original_long, "fast_trigger_level"] = frame.loc[original_long, "high"]
        return frame
