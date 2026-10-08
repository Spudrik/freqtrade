"""Focused checks for the explicit paper-retirement exit control."""

from datetime import datetime, timedelta, timezone

import pytest

from freqtrade.enums import RunMode
from user_data.strategies.integrated_paper import IntegratedPaper
from user_data.strategies.integrated_paper_variants import PaperVariant01, PaperVariant10
from user_data.strategies.paper_inverse_signals import PaperFastLevelInverse
from user_data.strategies.paper_trial_common import paper_user_force_close_reason


ROOT_DB = "sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/"
IDENTITIES = (
    (IntegratedPaper, "integrated_paper_auto", "IntegratedPaper", ROOT_DB + "auto_trades.sqlite"),
    (IntegratedPaper, "integrated_paper_manual", "IntegratedPaper", ROOT_DB + "manual_trades.sqlite"),
    (PaperVariant01, "integrated_paper_v01", "PaperVariant01", ROOT_DB + "v01_trades.sqlite"),
    (PaperVariant10, "integrated_paper_v10", "PaperVariant10", ROOT_DB + "v10_trades.sqlite"),
    (PaperFastLevelInverse, "paper_fast_level_inverse", "PaperFastLevelInverse",
     ROOT_DB + "fast_level_inverse_trades.sqlite"),
)
NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


def _config(bot_name: str, strategy_name: str, db_url: str, **extra) -> dict:
    return {
        "dry_run": True,
        "runmode": RunMode.DRY_RUN,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "exchange": {"name": "binance", "key": "", "secret": ""},
        "bot_name": bot_name,
        "strategy": strategy_name,
        "db_url": db_url,
        **extra,
    }


@pytest.mark.parametrize("strategy_cls,bot_name,strategy_name,db_url", IDENTITIES)
def test_explicit_force_close_returns_early_for_each_retirement_identity(
    strategy_cls, bot_name, strategy_name, db_url,
) -> None:
    strategy = strategy_cls.__new__(strategy_cls)
    strategy.config = _config(bot_name, strategy_name, db_url, paper_force_close=True)

    assert strategy.custom_exit("BTC/USDT:USDT", object(), NOW, 100.0, 0.0) == "paper_user_force_close"


def test_unflagged_force_close_does_not_change_normal_fast_exit() -> None:
    strategy = PaperFastLevelInverse.__new__(PaperFastLevelInverse)
    strategy.config = _config(*IDENTITIES[-1][1:])

    class Trade:
        is_short = False
        open_date_utc = NOW - timedelta(minutes=5)
        open_rate = 100.0

        @staticmethod
        def get_custom_data(key):
            assert key == "paper_fast_plan"
            return {"target_price": 105.0}

    assert strategy.custom_exit("SOL/USDT:USDT", Trade(), NOW, 105.0, 0.05) == "fast_atr_target"
    assert paper_user_force_close_reason({"paper_force_close": False}, "OtherStrategy") is None
    assert paper_user_force_close_reason({}, "OtherStrategy") is None


@pytest.mark.parametrize("config,strategy_name", [
    (_config("paper_fast_auto", "PaperFastAuto", ROOT_DB + "fast_auto_trades.sqlite",
             paper_force_close=True), "PaperFastAuto"),
    (_config("integrated_paper_auto", "IntegratedPaper", ROOT_DB + "auto_trades.sqlite",
             paper_force_close=True, dry_run=False, runmode=RunMode.LIVE), "IntegratedPaper"),
])
def test_force_close_refuses_unrelated_or_live_identity(config: dict, strategy_name: str) -> None:
    with pytest.raises(RuntimeError, match="paper_force_close"):
        paper_user_force_close_reason(config, strategy_name)


def test_force_close_flag_must_be_an_explicit_boolean() -> None:
    config = _config(*IDENTITIES[0][1:], paper_force_close=1)
    with pytest.raises(RuntimeError, match="boolean"):
        paper_user_force_close_reason(config, "IntegratedPaper")
