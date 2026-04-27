from freqtrade.strategy import CategoricalParameter

from PivotTrendlineMTFResearchStrategy import PivotTrendlineMTFResearchStrategy
from test_entry_research_base import PivotSingleEntryMixin


class TestPivotLongResistanceBreakout(PivotSingleEntryMixin, PivotTrendlineMTFResearchStrategy):
    can_short = False
    allowed_entry_tag = "long_resistance_breakout"

    entry_mode = CategoricalParameter(["resistance_breakout", "all"], default="resistance_breakout", space="buy", optimize=False, load=False)
