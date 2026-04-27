from freqtrade.strategy import CategoricalParameter

from PivotTrendlineMTFResearchStrategy import PivotTrendlineMTFResearchStrategy
from test_entry_research_base import PivotSingleEntryMixin


class TestPivotShortTrendPullback(PivotSingleEntryMixin, PivotTrendlineMTFResearchStrategy):
    can_short = True
    allowed_entry_tag = "short_trend_pullback"

    entry_mode = CategoricalParameter(["trend_pullback", "all"], default="trend_pullback", space="buy", optimize=False, load=False)
