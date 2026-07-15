from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import LayeredTargetExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_mtf_std_daily_macd_volume_breakout_long_1h import Sieve3ExitBreakevenFromMtfStdDailyMacdVolumeBreakoutLong1H


class Sieve3ReworkExitLayeredFromMtfStdDailyMacdVolumeBreakoutLong1H(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitLayeredFromMtfStdDailyMacdVolumeBreakoutLong1H,
    Sieve3ExitBreakevenFromMtfStdDailyMacdVolumeBreakoutLong1H,
    LayeredTargetExitMixin,
    source_profile="mtf_macd_breakout_long",
    source_target_columns=("prior_high_1d",),
)

