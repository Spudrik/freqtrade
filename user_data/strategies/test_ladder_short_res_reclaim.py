from freqtrade.strategy import BooleanParameter

from DailyStructureLadderStrategyTOP10 import DailyStructureLadderStrategyTOP10
from test_entry_research_base import LadderSingleEntryMixin


class TestLadderShortResReclaim(LadderSingleEntryMixin, DailyStructureLadderStrategyTOP10):
    can_short = True
    entry_mask_method = "_short_res_reclaim_mask"
    entry_side = "short"
    entry_tag = "short_res_reclaim"

    enable_short_res_reclaim = BooleanParameter(default=True, space="buy", optimize=False, load=False)
