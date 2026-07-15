from freqtrade.strategy import IStrategy

from _sieve3_exit_rework_core import SourceInvalidationExitMixin, materialize_rework_strategy
from sieve3_exit_breakeven_from_multi2_tlv2_vp_res_break_vp_bullctx_long_1h import Sieve3ExitBreakevenFromMulti2Tlv2VpResBreakVpBullctxLong1H


class Sieve3ReworkExitInvalidationFromMulti2Tlv2VpResBreakVpBullctxLong1H(IStrategy):
    pass


materialize_rework_strategy(
    Sieve3ReworkExitInvalidationFromMulti2Tlv2VpResBreakVpBullctxLong1H,
    Sieve3ExitBreakevenFromMulti2Tlv2VpResBreakVpBullctxLong1H,
    SourceInvalidationExitMixin,
    source_profile="tlv2_vp_bull_long",
    source_target_columns=("tlv2_resistance_line_rank0", "vp_vah", "vp_hvn_above", "vp_poc"),
)

