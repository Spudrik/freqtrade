from freqtrade.strategy import BooleanParameter

from DailyStructureLadderStrategyTOP10 import DailyStructureLadderStrategyTOP10
from test_entry_research_base import LadderSingleEntryMixin


class TestLadderShortSupBreak(LadderSingleEntryMixin, DailyStructureLadderStrategyTOP10):
    can_short = True
    entry_mask_method = "_short_support_break_mask"
    entry_side = "short"
    entry_tag = "short_sup_break"

    enable_short_sup_break = BooleanParameter(default=True, space="buy", optimize=False, load=False)
