from freqtrade.strategy import BooleanParameter

from DailyStructureLadderStrategyTOP10 import DailyStructureLadderStrategyTOP10
from test_entry_research_base import LadderSingleEntryMixin


class TestLadderLongSupHold(LadderSingleEntryMixin, DailyStructureLadderStrategyTOP10):
    can_short = False
    entry_mask_method = "_long_support_mask"
    entry_side = "long"
    entry_tag = "long_sup_hold"

    enable_long_sup_hold = BooleanParameter(default=True, space="buy", optimize=False, load=False)
