from __future__ import annotations

from ..services.collector_service import NEWS_PROFILE
from .research_collector_tab import ResearchCollectorTab


class NewsTab(ResearchCollectorTab):
    tab_key = "news"
    tab_title = "News Lab"
    profile = NEWS_PROFILE
