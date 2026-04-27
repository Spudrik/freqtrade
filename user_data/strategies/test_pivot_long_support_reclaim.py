from freqtrade.strategy import CategoricalParameter

from PivotTrendlineMTFResearchStrategy import PivotTrendlineMTFResearchStrategy
from test_entry_research_base import PivotSingleEntryMixin


class TestPivotLongSupportReclaim(PivotSingleEntryMixin, PivotTrendlineMTFResearchStrategy):
    can_short = False
    allowed_entry_tag = "long_support_reclaim"

    entry_mode = CategoricalParameter(["support_reclaim", "all"], default="support_reclaim", space="buy", optimize=False, load=False)
