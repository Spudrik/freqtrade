from __future__ import annotations

from .context_source_catalog_view import ContextSourceCatalogView


class NewsBackfillSourcesTab(ContextSourceCatalogView):
    tab_key = "news_backfill_sources"
    tab_title = "News Backfill"
    catalog_group = "news_backfill"
    intro_text = (
        "News Backfill tracks sources that should merge into the existing News/Web collector pattern "
        "when they are implemented. Do not create a parallel news ingestion framework for these sources."
    )
