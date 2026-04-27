from freqtrade.strategy import CategoricalParameter

from PivotTrendlineMTFResearchStrategy import PivotTrendlineMTFResearchStrategy
from test_entry_research_base import PivotSingleEntryMixin


class TestPivotLongTrendPullback(PivotSingleEntryMixin, PivotTrendlineMTFResearchStrategy):
    can_short = False
    allowed_entry_tag = "long_trend_pullback"

    entry_mode = CategoricalParameter(["trend_pullback", "all"], default="trend_pullback", space="buy", optimize=False, load=False)
