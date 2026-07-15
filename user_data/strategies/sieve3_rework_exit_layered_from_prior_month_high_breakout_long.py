from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import LayeredTargetExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_prior_month_high_breakout_long import Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong


class Sieve3ReworkExitLayeredFromPriorMonthHighBreakoutLong(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitLayeredFromPriorMonthHighBreakoutLong,
    Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong,
    LayeredTargetExitMixin,
    source_profile="prior_month_breakout_long",
    source_target_columns=("vp1d_vah", "vp4h_vah", "vp_vah", "vp_hvn_above", "prior_week_high"),
)

