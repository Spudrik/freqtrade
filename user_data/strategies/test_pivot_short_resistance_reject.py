from freqtrade.strategy import CategoricalParameter

from PivotTrendlineMTFResearchStrategy import PivotTrendlineMTFResearchStrategy
from test_entry_research_base import PivotSingleEntryMixin


class TestPivotShortResistanceReject(PivotSingleEntryMixin, PivotTrendlineMTFResearchStrategy):
    can_short = True
    allowed_entry_tag = "short_resistance_reject"

    entry_mode = CategoricalParameter(["resistance_reject", "all"], default="resistance_reject", space="buy", optimize=False, load=False)
