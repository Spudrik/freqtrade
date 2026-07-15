from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import SourceInvalidationExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_mtfx_h4_tlv2_sup_break_short_1h_choch import Sieve3ExitBreakevenFromMtfxH4Tlv2SupBreakShort1HChoch


class Sieve3ReworkExitInvalidationFromMtfxH4Tlv2SupBreakShort1HChoch(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitInvalidationFromMtfxH4Tlv2SupBreakShort1HChoch,
    Sieve3ExitBreakevenFromMtfxH4Tlv2SupBreakShort1HChoch,
    SourceInvalidationExitMixin,
    source_profile="mtfx_tlv2_break_short",
    source_target_columns=("ltfms_pivot_low", "htfvp_prior_val", "htfvp_lvn_below", "mtfx_prior_low"),
)

