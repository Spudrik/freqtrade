from __future__ import annotations

from .context_source_catalog_view import ContextSourceCatalogView


class MarketContextSourcesTab(ContextSourceCatalogView):
    tab_key = "market_context_sources"
    tab_title = "Market Context Sources"
    catalog_group = "market_context_sources"
    intro_text = (
        "Market Context Sources tracks future numeric macro, energy, and crypto network inputs. "
        "Where possible, these should extend the existing Global Context collector and use its storage behavior."
    )
