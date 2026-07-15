from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import LayeredTargetExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_mtfx_h4_tlv2_sup_break_short_1h_choch import Sieve3ExitBreakevenFromMtfxH4Tlv2SupBreakShort1HChoch


class Sieve3ReworkExitLayeredFromMtfxH4Tlv2SupBreakShort1HChoch(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitLayeredFromMtfxH4Tlv2SupBreakShort1HChoch,
    Sieve3ExitBreakevenFromMtfxH4Tlv2SupBreakShort1HChoch,
    LayeredTargetExitMixin,
    source_profile="mtfx_tlv2_break_short",
    source_target_columns=("ltfms_pivot_low", "htfvp_prior_val", "htfvp_lvn_below", "mtfx_prior_low"),
)

