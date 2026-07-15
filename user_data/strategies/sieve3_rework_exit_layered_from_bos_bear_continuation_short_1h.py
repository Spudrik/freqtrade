from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import LayeredTargetExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_bos_bear_continuation_short_1h import Sieve3ExitBreakevenFromBosBearContinuationShort1H


class Sieve3ReworkExitLayeredFromBosBearContinuationShort1H(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitLayeredFromBosBearContinuationShort1H,
    Sieve3ExitBreakevenFromBosBearContinuationShort1H,
    LayeredTargetExitMixin,
    source_profile="bos_bear_short",
    source_target_columns=("ms_pivot_low", "ms_bearish_break_level"),
)

