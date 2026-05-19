from __future__ import annotations

from .context_source_catalog_view import ContextSourceCatalogView


class SocialContextSourcesTab(ContextSourceCatalogView):
    tab_key = "social_context_sources"
    tab_title = "Social Context Sources"
    catalog_group = "social_context_sources"
    intro_text = (
        "Social Context Sources tracks Reddit/Twitter-style historical sentiment or attention datasets. "
        "Use aggregate timestamped features only, and verify license, privacy, and platform terms before import."
    )
