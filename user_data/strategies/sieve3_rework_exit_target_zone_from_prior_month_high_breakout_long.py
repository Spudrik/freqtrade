from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import TargetZoneExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_prior_month_high_breakout_long import Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong


class Sieve3ReworkExitTargetZoneFromPriorMonthHighBreakoutLong(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitTargetZoneFromPriorMonthHighBreakoutLong,
    Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong,
    TargetZoneExitMixin,
    source_profile="prior_month_breakout_long",
    source_target_columns=("vp1d_vah", "vp4h_vah", "vp_vah", "vp_hvn_above", "prior_week_high"),
)

