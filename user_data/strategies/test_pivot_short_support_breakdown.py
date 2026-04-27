from freqtrade.strategy import CategoricalParameter

from PivotTrendlineMTFResearchStrategy import PivotTrendlineMTFResearchStrategy
from test_entry_research_base import PivotSingleEntryMixin


class TestPivotShortSupportBreakdown(PivotSingleEntryMixin, PivotTrendlineMTFResearchStrategy):
    can_short = True
    allowed_entry_tag = "short_support_breakdown"

    entry_mode = CategoricalParameter(["support_breakdown", "all"], default="support_breakdown", space="buy", optimize=False, load=False)
