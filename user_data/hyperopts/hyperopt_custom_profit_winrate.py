"""
Custom hyperopt loss focused on:
- Profit (primary)
- Winrate (secondary, but important)
- Reasonable trade activity (avoid near buy-and-hold / too-few trades)
- Risk control via drawdown (moderate influence, not dominant)

Design goals for this strategy family:
- Do not reward high trade frequency by itself.
- Only penalize *too low* trade activity.
- Keep drawdown as a guardrail, not the main objective.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_underwater
from freqtrade.optimize.hyperopt import IHyperOptLoss


# Profit / winrate emphasis
PROFIT_WEIGHT = 420.0
WINRATE_WEIGHT = 160.0
MIN_WINRATE = 0.52
WINRATE_BELOW_FLOOR_PENALTY = 90.0

# Activity guardrail: only low-trade penalty, no high-trade bonus/penalty.
MIN_TRADES_PER_DAY = 0.35
LOW_ACTIVITY_PENALTY_WEIGHT = 45.0

# Drawdown guardrails: moderate pressure, escalates only when DD gets high.
DD_SOFT = 0.22
DD_HARD = 0.38
DD_SOFT_PENALTY_WEIGHT = 140.0
DD_HARD_STEP_PENALTY = 55.0

# Safety fallback for degenerate epochs.
MAX_LOSS = 90000.0


class ProfitWinrateModerateRiskHyperOptLoss(IHyperOptLoss):
    """
    Profit + winrate centric objective with moderate risk shaping.

    Lower loss is better.
    """

    @staticmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        backtest_stats: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        _ = kwargs

        if results is None or len(results) == 0:
            return MAX_LOSS

        starting_balance = float(config.get("dry_run_wallet", 0.0) or 0.0)
        if starting_balance <= 0.0:
            return MAX_LOSS

        day_count = max(1, (max_date - min_date).days + 1)
        total_profit_abs = float(results["profit_abs"].sum())
        total_profit_ratio = total_profit_abs / starting_balance
        avg_trades_per_day = float(trade_count) / float(day_count)

        # Use backtest_stats winrate when available, fallback to direct results calc.
        winrate = float(backtest_stats.get("winrate", 0.0) or 0.0)
        if winrate <= 0.0 and len(results) > 0:
            winrate = float((results["profit_abs"] > 0).mean())

        # Relative drawdown (0..1). Keep this as a moderate penalty source.
        try:
            dd_df = calculate_underwater(
                results,
                value_col="profit_abs",
                starting_balance=starting_balance,
            )
            relative_drawdown = float(dd_df["drawdown_relative"].max())
        except (Exception, ValueError):
            relative_drawdown = 0.0

        # Core objective: maximize profit and winrate.
        # Negative terms reward higher values because hyperopt minimizes loss.
        profit_term = -PROFIT_WEIGHT * total_profit_ratio
        winrate_term = -WINRATE_WEIGHT * winrate

        # Penalty when winrate drops under desired floor.
        winrate_floor_gap = max(0.0, MIN_WINRATE - winrate)
        winrate_floor_penalty = WINRATE_BELOW_FLOOR_PENALTY * (winrate_floor_gap**2)

        # Penalize only low trading activity (prevents single-trade buy-hold solutions).
        low_activity_gap = max(0.0, MIN_TRADES_PER_DAY - avg_trades_per_day)
        low_activity_penalty = LOW_ACTIVITY_PENALTY_WEIGHT * (low_activity_gap**2)

        # Moderate drawdown penalties:
        # - soft quadratic penalty above DD_SOFT
        # - extra step penalty above DD_HARD
        dd_soft_gap = max(0.0, relative_drawdown - DD_SOFT)
        dd_penalty = DD_SOFT_PENALTY_WEIGHT * (dd_soft_gap**2)
        if relative_drawdown > DD_HARD:
            dd_penalty += DD_HARD_STEP_PENALTY

        # Strongly penalize non-profitable epochs to stabilize optimization.
        non_profit_penalty = 0.0
        if total_profit_abs <= 0.0:
            non_profit_penalty = 120.0 + (abs(total_profit_ratio) * 240.0)

        loss = (
            profit_term
            + winrate_term
            + winrate_floor_penalty
            + low_activity_penalty
            + dd_penalty
            + non_profit_penalty
        )

        if loss > MAX_LOSS:
            return MAX_LOSS
        return float(loss)

