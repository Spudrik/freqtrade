from freqtrade.strategy import BooleanParameter

from DailyStructureLadderStrategyTOP10 import DailyStructureLadderStrategyTOP10
from test_entry_research_base import LadderSingleEntryMixin


class TestLadderLongResRetestHold(LadderSingleEntryMixin, DailyStructureLadderStrategyTOP10):
    can_short = False
    entry_mask_method = "_long_res_retest_hold_mask"
    entry_side = "long"
    entry_tag = "long_res_retest_hold"

    enable_long_res_retest_hold = BooleanParameter(default=True, space="buy", optimize=False, load=False)
