from __future__ import annotations

from .context_source_catalog_view import ContextSourceCatalogView


class HistoricalDataSourcesTab(ContextSourceCatalogView):
    tab_key = "historical_data_sources"
    tab_title = "Historical Sources"
    catalog_group = "historical_archives"
    intro_text = (
        "Historical Sources tracks existing timestamped data blocks that could backfill context research. "
        "Use the buttons to open the source/download page and the planned local storage folder. "
        "Actual importers should be added later only after timestamp and license checks."
    )
