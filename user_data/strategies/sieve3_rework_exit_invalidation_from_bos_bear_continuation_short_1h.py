from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import SourceInvalidationExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_bos_bear_continuation_short_1h import Sieve3ExitBreakevenFromBosBearContinuationShort1H


class Sieve3ReworkExitInvalidationFromBosBearContinuationShort1H(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitInvalidationFromBosBearContinuationShort1H,
    Sieve3ExitBreakevenFromBosBearContinuationShort1H,
    SourceInvalidationExitMixin,
    source_profile="bos_bear_short",
    source_target_columns=("ms_pivot_low", "ms_bearish_break_level"),
)

