from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import TargetZoneExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_mtfx_h4_tlv2_sup_break_short_1h_choch import Sieve3ExitBreakevenFromMtfxH4Tlv2SupBreakShort1HChoch


class Sieve3ReworkExitTargetZoneFromMtfxH4Tlv2SupBreakShort1HChoch(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitTargetZoneFromMtfxH4Tlv2SupBreakShort1HChoch,
    Sieve3ExitBreakevenFromMtfxH4Tlv2SupBreakShort1HChoch,
    TargetZoneExitMixin,
    source_profile="mtfx_tlv2_break_short",
    source_target_columns=("ltfms_pivot_low", "htfvp_prior_val", "htfvp_lvn_below", "mtfx_prior_low"),
)

