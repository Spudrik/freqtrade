from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import webbrowser

from .collector_service import open_path


DATA_MANAGEMENT_CONTEXT_SOURCE_TODO = """
Context data backlog for future research trials.

Rules before implementing any downloader/importer:
1. Confirm the source has a usable availability timestamp or capture timestamp.
2. Confirm license/provenance; avoid random file-share dumps without clear terms.
3. Reuse existing News/Web/Global Context collectors when the shape fits.
4. Store raw imports under user_data/research_news_data/<source_family>/.
5. Store derived numeric features only in context_features.
6. Keep strategies free of scraping, parsing, API calls, and slow NLP.
7. Do not add orderbook sources here; orderbook is a separate task.

High-value candidates:
- GDELT Events/GKG for structured geopolitics, conflict, oil, banking, macro themes.
- Common Crawl CC-NEWS for WARC capture-time news text sampling.
- Guardian Open Platform for clean historical article API access.
- NYT Archive metadata/headlines if API terms fit the research use.
- Media Cloud for media-volume/source-diversity research.
- Internet Archive TV News captions for broadcast narrative intensity.
- Coin Metrics Community for crypto network/context metrics.
- FRED/ALFRED release/vintage data for macro no-lookahead correctness.
- EIA or similar for oil/energy history.
- Reddit/crypto social archives only with clear license and privacy handling.
"""


@dataclass(frozen=True)
class ContextSourceItem:
    key: str
    name: str
    priority: str
    status: str
    contents: str
    timestamp: str
    merge_target: str
    storage: str
    source_url: str
    docs_url: str
    license_note: str
    next_step: str


class ContextSourceCatalogService:
    def __init__(self, app_dir: Path) -> None:
        self.app_dir = Path(app_dir)

    def data_root(self) -> Path:
        return (self.app_dir / "../research_news_data").resolve()

    def items(self, group: str) -> list[ContextSourceItem]:
        return list(CATALOG_GROUPS.get(group, ()))

    def open_url(self, url: str) -> None:
        text = str(url or "").strip()
        if not text:
            raise ValueError("No URL configured for this source.")
        webbrowser.open(text)

    def open_storage_folder(self, item: ContextSourceItem) -> None:
        storage = str(item.storage or "").strip()
        if not storage:
            raise ValueError("No storage folder configured for this source.")
        folder = self.data_root() / storage
        folder.mkdir(parents=True, exist_ok=True)
        open_path(folder)

    @staticmethod
    def item_rows(items: list[ContextSourceItem]) -> list[tuple[Any, ...]]:
        return [
            (
                item.priority,
                item.name,
                item.status,
                item.merge_target,
                item.timestamp,
                item.storage,
            )
            for item in items
        ]


CATALOG_GROUPS: dict[str, tuple[ContextSourceItem, ...]] = {
    "historical_archives": (
        ContextSourceItem(
            key="gdelt_events_gkg",
            name="GDELT Events and GKG",
            priority="1",
            status="Candidate backfill",
            contents="Global event records, themes, entities, tone, locations, conflict/geopolitics/macro themes.",
            timestamp="Event date plus update/capture cadence; use downloaded file time as availability where needed.",
            merge_target="New GDELT backfill feeding normalised_context_events",
            storage="historical_sources/gdelt",
            source_url="https://www.gdeltproject.org/",
            docs_url="https://data.gdeltproject.org/documentation/",
            license_note="Free/open GDELT data; confirm exact field semantics before using as no-lookahead features.",
            next_step="Build a small 2020-2026 filtered downloader for GDELT event/theme counts.",
        ),
        ContextSourceItem(
            key="common_crawl_news",
            name="Common Crawl CC-NEWS",
            priority="2",
            status="Candidate sampled import",
            contents="News WARC files from many domains; useful for broad timestamped article capture samples.",
            timestamp="WARC capture timestamp; treat as available_at.",
            merge_target="News/Web backfill adapter after filtering and parsing",
            storage="historical_sources/common_crawl_news",
            source_url="https://commoncrawl.org/blog/news-dataset-available",
            docs_url="https://data.commoncrawl.org/crawl-data/CC-NEWS/",
            license_note="Open crawl data, but source articles may be copyrighted; store derived metadata/features carefully.",
            next_step="Sample selected months and keyword filters before any large download.",
        ),
        ContextSourceItem(
            key="guardian_open_platform",
            name="Guardian Open Platform",
            priority="3",
            status="Candidate API import",
            contents="Guardian archive articles, tags, sections, publication dates, and article metadata.",
            timestamp="Publication date from API; API fetch time is available_at for local import.",
            merge_target="News Lab source adapter",
            storage="historical_sources/guardian",
            source_url="https://open-platform.theguardian.com/",
            docs_url="https://open-platform.theguardian.com/documentation/",
            license_note="Free developer access for allowed uses; requires key and terms review.",
            next_step="Add API-key setting and one-month dry-run import if terms fit.",
        ),
        ContextSourceItem(
            key="nyt_archive",
            name="NYT Archive API",
            priority="4",
            status="Candidate metadata import",
            contents="Monthly NYT article metadata/headlines/abstracts going far back historically.",
            timestamp="Publication date from API; API fetch time is available_at for local import.",
            merge_target="News Lab metadata source adapter",
            storage="historical_sources/nyt_archive",
            source_url="https://developer.nytimes.com/docs/archive-product/1/overview",
            docs_url="https://developer.nytimes.com/",
            license_note="Requires API key and terms review; likely metadata/headline use only.",
            next_step="Prototype metadata-only monthly import if key is available.",
        ),
        ContextSourceItem(
            key="reddit_academic_torrents",
            name="Reddit Academic Torrents / Arctic Shift",
            priority="5",
            status="Candidate social archive",
            contents="Historical Reddit submissions/comments for crypto and market communities.",
            timestamp="created_utc from Reddit objects.",
            merge_target="Social Context source family, not News/Web",
            storage="historical_sources/reddit",
            source_url="https://academictorrents.com/collection/datasetreddit?sort_dir=DESC&sort_field=added",
            docs_url="https://github.com/arthurheitmann/arctic_shift",
            license_note="Very large data and platform terms/privacy need review before use.",
            next_step="Use subreddit/month subsets only; aggregate counts and descriptors, avoid user-level modeling.",
        ),
    ),
    "gdelt_backfill": (
        ContextSourceItem(
            key="gdelt_event_counts",
            name="GDELT structured event counts",
            priority="1",
            status="Best first backfill",
            contents="Counts by event code/theme for conflict, protests, sanctions, banking stress, energy, macro themes.",
            timestamp="Use GDELT event/date fields plus local import time; no future file backfill into earlier candles.",
            merge_target="normalised_context_events and context_features_1h",
            storage="gdelt",
            source_url="https://www.gdeltproject.org/",
            docs_url="https://data.gdeltproject.org/documentation/",
            license_note="Free/open; field-level interpretation still needs test notes.",
            next_step="Create a small downloader/importer for a bounded historical timerange.",
        ),
        ContextSourceItem(
            key="gdelt_gkg_themes",
            name="GDELT GKG theme/tone aggregates",
            priority="2",
            status="Planned after event counts",
            contents="Themes, organizations, countries, tone, and topical co-occurrence from global news.",
            timestamp="GKG record time / file time; use conservative available_at.",
            merge_target="GDELT source family feature groups",
            storage="gdelt",
            source_url="https://data.gdeltproject.org/gkg/index.html",
            docs_url="https://data.gdeltproject.org/documentation/",
            license_note="Large files; start with theme-level hourly aggregates, not raw article storage.",
            next_step="Aggregate selected theme prefixes into hourly counts.",
        ),
    ),
    "news_backfill": (
        ContextSourceItem(
            key="guardian_news_adapter",
            name="Guardian API to News Lab",
            priority="1",
            status="Merge into existing News Lab",
            contents="Article metadata/body/tags for macro, world, business, markets, technology, crypto-adjacent terms.",
            timestamp="Published date plus fetched/imported time; features align by fetched/imported time unless availability is proven.",
            merge_target="Existing News Lab schema",
            storage="news_backfill/guardian",
            source_url="https://open-platform.theguardian.com/",
            docs_url="https://open-platform.theguardian.com/documentation/",
            license_note="Requires key and terms review.",
            next_step="Add as a News Lab source adapter only if API key is configured.",
        ),
        ContextSourceItem(
            key="media_cloud_adapter",
            name="Media Cloud search/API",
            priority="2",
            status="Potential News Lab adapter",
            contents="News search/archive source diversity and volume metrics across media ecosystems.",
            timestamp="Story publication time plus API fetch/import time.",
            merge_target="News Lab or derived source-volume store",
            storage="news_backfill/media_cloud",
            source_url="https://www.mediacloud.org/documentation",
            docs_url="https://www.mediacloud.org/documentation",
            license_note="Confirm API access and data reuse terms.",
            next_step="Test a small query for BTC/macroeconomic topics and inspect returned metadata.",
        ),
        ContextSourceItem(
            key="common_crawl_filtered_news",
            name="Common Crawl filtered news sample",
            priority="3",
            status="Potential Web/News import",
            contents="Filtered article captures from CC-NEWS for selected historical windows.",
            timestamp="WARC capture timestamp.",
            merge_target="Web Lab if parsed from archived HTML, News Lab if normalized as article records",
            storage="news_backfill/common_crawl_news",
            source_url="https://commoncrawl.org/blog/news-dataset-available",
            docs_url="https://data.commoncrawl.org/crawl-data/CC-NEWS/",
            license_note="Store minimal derived data; avoid broad redistribution of copyrighted article text.",
            next_step="Prototype one month, one keyword family, one output file.",
        ),
    ),
    "market_context_sources": (
        ContextSourceItem(
            key="coin_metrics_community",
            name="Coin Metrics Community",
            priority="1",
            status="Candidate Global Context source",
            contents="BTC/ETH network and market metrics with UTC timestamps.",
            timestamp="ISO 8601 UTC time in API responses.",
            merge_target="Global Context collector",
            storage="global_context/coin_metrics",
            source_url="https://docs.coinmetrics.io/access-our-data/api",
            docs_url="https://gitbook-docs.coinmetrics.io/packages/coin-metrics-community-data",
            license_note="Community data is free for non-commercial use under stated terms; rate limited.",
            next_step="Add selected daily BTC/ETH metrics to Global Context after terms review.",
        ),
        ContextSourceItem(
            key="fred_alfred_vintages",
            name="FRED / ALFRED vintages",
            priority="2",
            status="Improve existing Global Context",
            contents="Macro series with vintage/release-date handling for no-lookahead research.",
            timestamp="Release/vintage dates should drive available_at.",
            merge_target="Existing Global Context collector",
            storage="global_context/fred_alfred",
            source_url="https://fred.stlouisfed.org/docs/api/fred/",
            docs_url="https://alfred.stlouisfed.org/",
            license_note="Requires FRED key for some API use; release timing must be modeled carefully.",
            next_step="Extend current FRED sources to use release/vintage dates where practical.",
        ),
        ContextSourceItem(
            key="oil_energy_prices",
            name="Oil / energy prices",
            priority="3",
            status="Candidate Global Context source",
            contents="WTI, Brent, natural gas, energy shock proxy returns.",
            timestamp="Market data timestamp or release date depending source.",
            merge_target="Global Context collector",
            storage="global_context/energy",
            source_url="https://www.eia.gov/opendata/",
            docs_url="https://www.eia.gov/opendata/",
            license_note="Confirm source license and latency; daily data may be enough for first pass.",
            next_step="Add WTI/Brent daily returns if a stable free endpoint is confirmed.",
        ),
    ),
    "social_context_sources": (
        ContextSourceItem(
            key="reddit_crypto_subsets",
            name="Reddit crypto/market subsets",
            priority="1",
            status="Research candidate",
            contents="Subreddit post/comment volumes, keyword families, source/community activity.",
            timestamp="created_utc.",
            merge_target="Social Context source family",
            storage="social_context/reddit",
            source_url="https://academictorrents.com/collection/datasetreddit?sort_dir=DESC&sort_field=added",
            docs_url="https://github.com/arthurheitmann/arctic_shift",
            license_note="Use aggregate features; avoid user-level profiles and unclear redistribution.",
            next_step="Import a small subreddit/month subset and aggregate hourly counts.",
        ),
        ContextSourceItem(
            key="bitcoin_tweet_kaggle",
            name="Bitcoin tweet datasets",
            priority="2",
            status="Trial-only candidate",
            contents="Historical Bitcoin-related tweets with timestamps from public dataset pages.",
            timestamp="Dataset-provided tweet timestamp.",
            merge_target="Social Context source family",
            storage="social_context/twitter_kaggle",
            source_url="https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329",
            docs_url="https://www.kaggle.com/datasets/andreapenasmartinez/bitcoin-twitter-sentiment-dataset-20132023/versions/1",
            license_note="Kaggle dataset licenses and platform terms vary; verify before importing.",
            next_step="Only use if license is explicit and file fields include reliable timestamps.",
        ),
    ),
}
