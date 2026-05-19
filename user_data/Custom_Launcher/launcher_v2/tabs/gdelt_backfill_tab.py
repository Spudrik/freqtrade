from __future__ import annotations

from .context_source_catalog_view import ContextSourceCatalogView


class GdeltBackfillTab(ContextSourceCatalogView):
    tab_key = "gdelt_backfill"
    tab_title = "GDELT Backfill"
    catalog_group = "gdelt_backfill"
    intro_text = (
        "GDELT Backfill is the planned structured-event path for historical geopolitics, conflict, oil, "
        "banking, sanctions, and macro theme counts. Keep it as a source-family import feeding the same "
        "normalised context events and hourly feature store."
    )
