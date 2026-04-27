from __future__ import annotations

from ..services.collector_service import WEB_PROFILE
from .research_collector_tab import ResearchCollectorTab


class WebTab(ResearchCollectorTab):
    tab_key = "web"
    tab_title = "Web Lab"
    profile = WEB_PROFILE
