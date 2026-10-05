"""Build the frozen 2020-2026 Federal Reserve liquidity-action catalogue.

Candidate pages are fixed from the official annual press-release indexes without
reading crypto prices.  The catalogue preserves individual announcements, exact
official release clocks, simultaneous-release clusters, and conservative broad
episodes so many facility updates from one crisis cannot be counted as independent
market confirmations.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_fed_liquidity_catalogue_pilot as pilot,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "fed_liquidity_catalogue_20260905a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "fed_liquidity_event_catalogue.csv"
MANIFEST_PATH = OUTPUT_ROOT / "fed_liquidity_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "fed_liquidity_catalogue_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "fed_liquidity_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "fed_liquidity_catalogue_result.json"

ARCHIVE_URLS = {
    year: f"https://www.federalreserve.gov/newsevents/pressreleases/{year}-press.htm"
    for year in range(2020, 2027)
}
SELECTION_VOCABULARY = (
    "CPFF",
    "PDCF",
    "MMLF",
    "PPPLF",
    "Main Street Lending",
    "Municipal Liquidity",
    "PMCCF",
    "SMCCF",
    "TALF",
    "FIMA Repo",
    "dollar liquidity swap lines",
    "BTFP",
    "intraday credit",
    "credit flow",
    "money-market liquidity",
)
ACTION_TERMS = (
    "establish",
    "expand",
    "activate",
    "fully operational",
    "additional funding",
    "extend",
    "wind down",
    "cease",
    "temporary change",
    "revised pricing",
)


def _spec(
    page: str,
    *,
    action_class: str = "facility_or_liquidity_action",
    episode: str,
) -> dict[str, str]:
    year_match = re.search(r"20\d{2}", page)
    if year_match is None:
        raise ValueError(f"Page identifier has no year: {page}")
    year = int(year_match.group())
    inclusion_basis = {
        "facility_or_liquidity_action": (
            "facility creation activation material change extension wind-down or termination"
        ),
        "emergency_regulatory_capacity_action": (
            "temporary emergency balance-sheet or credit-capacity change"
        ),
        "settlement_credit_action": (
            "temporary settlement or intraday-credit availability change"
        ),
    }[action_class]
    return {
        "event_id": f"fed_liquidity_{page.removesuffix('.htm')}",
        "year": str(year),
        "source_url": (
            "https://www.federalreserve.gov/newsevents/pressreleases/" + page
        ),
        "action_class": action_class,
        "inclusion_basis": inclusion_basis,
        "programme_episode_id": episode,
    }


# This complete URL surface was selected from official 2020-2026 annual indexes
# using SELECTION_VOCABULARY and ACTION_TERMS before any market outcomes were read.
EVENT_SPECS: tuple[dict[str, str], ...] = (
    _spec("monetary20200315b.htm", episode="covid_2020_support"),
    _spec("monetary20200315c.htm", episode="covid_2020_support"),
    _spec("monetary20200317a.htm", episode="covid_2020_support"),
    _spec("monetary20200317b.htm", episode="covid_2020_support"),
    _spec("monetary20200318a.htm", episode="covid_2020_support"),
    _spec("monetary20200319b.htm", episode="covid_2020_support"),
    _spec("monetary20200320a.htm", episode="covid_2020_support"),
    _spec("monetary20200320b.htm", episode="covid_2020_support"),
    _spec("monetary20200323b.htm", episode="covid_2020_support"),
    _spec("monetary20200331a.htm", episode="covid_2020_support"),
    _spec("monetary20200406a.htm", episode="covid_2020_support"),
    _spec("monetary20200409a.htm", episode="covid_2020_support"),
    _spec("monetary20200416a.htm", episode="covid_2020_support"),
    _spec("monetary20200427a.htm", episode="covid_2020_support"),
    _spec("monetary20200430a.htm", episode="covid_2020_support"),
    _spec("monetary20200430b.htm", episode="covid_2020_support"),
    _spec("monetary20200603a.htm", episode="covid_2020_support"),
    _spec("monetary20200608a.htm", episode="covid_2020_support"),
    _spec("monetary20200615a.htm", episode="covid_2020_support"),
    _spec("monetary20200717a.htm", episode="covid_2020_support"),
    _spec("monetary20200723a.htm", episode="covid_2020_support"),
    _spec("monetary20200728a.htm", episode="covid_2020_support"),
    _spec("monetary20200729b.htm", episode="covid_2020_support"),
    _spec("monetary20200811a.htm", episode="covid_2020_support"),
    _spec(
        "other20201001a.htm",
        action_class="settlement_credit_action",
        episode="covid_2020_support",
    ),
    _spec("monetary20201030a.htm", episode="covid_2020_support"),
    _spec("monetary20201130a.htm", episode="covid_2020_support"),
    _spec("monetary20201216c.htm", episode="covid_2020_support"),
    _spec("monetary20210308a.htm", episode="covid_2021_lifecycle"),
    _spec("monetary20210602a.htm", episode="covid_2021_lifecycle"),
    _spec("monetary20210616c.htm", episode="covid_2021_lifecycle"),
    _spec("monetary20210625a.htm", episode="covid_2021_lifecycle"),
    _spec("monetary20230312a.htm", episode="banking_stress_2023"),
    _spec("monetary20230312b.htm", episode="banking_stress_2023"),
    _spec("monetary20230319a.htm", episode="banking_stress_2023"),
    _spec("monetary20240124a.htm", episode="btfp_lifecycle_2024"),
    _spec(
        "bcreg20200323a.htm",
        action_class="emergency_regulatory_capacity_action",
        episode="covid_2020_support",
    ),
    _spec(
        "bcreg20200401a.htm",
        action_class="emergency_regulatory_capacity_action",
        episode="covid_2020_support",
    ),
    _spec(
        "other20200423a.htm",
        action_class="settlement_credit_action",
        episode="covid_2020_support",
    ),
    _spec(
        "bcreg20200515a.htm",
        action_class="emergency_regulatory_capacity_action",
        episode="covid_2020_support",
    ),
)

OUTPUT_COLUMNS = (
    "event_id",
    "available_at_utc",
    "official_release_local_text",
    "release_date_text",
    "release_time_text",
    "official_title",
    "action_class",
    "inclusion_basis",
    "programme_episode_id",
    "announcement_cluster_id",
    "announcement_schedule_known",
    "expectation_consensus",
    "source_url",
    "annual_archive_url",
    "source_retrieval_timestamp",
    "source_content_sha256",
)


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", None)
    if content is not None:
        return content.encode("utf-8") if isinstance(content, str) else bytes(content)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    raise TypeError("Official response contains neither content nor text")


def _fetch(
    url: str,
    *,
    requester: Any = requests.get,
    timeout: int = 45,
) -> tuple[str, bytes]:
    response = requester(
        url,
        headers={"User-Agent": "Objective02b-fed-liquidity-catalogue/1.0"},
        timeout=timeout,
    )
    if int(response.status_code) != 200:
        raise RuntimeError(
            f"Official Federal Reserve source returned {response.status_code}: {url}"
        )
    content = _response_bytes(response)
    encoding = getattr(response, "encoding", None) or "utf-8"
    return content.decode(encoding, errors="replace"), content


def archive_links(html: str, archive_url: str) -> set[str]:
    links = re.findall(r"href=[\"'](?P<href>[^\"']+)[\"']", html, flags=re.I)
    return {urljoin(archive_url, link).split("#", 1)[0] for link in links}


def build_catalogue(
    *,
    archive_html: Mapping[int, str],
    page_html: Mapping[str, str],
    retrieved_at_utc: str,
    page_content: Mapping[str, bytes] | None = None,
) -> DataFrame:
    """Build and validate the frozen catalogue from already fetched official pages."""
    records: list[dict[str, Any]] = []
    archive_link_sets = {
        year: archive_links(html, ARCHIVE_URLS[year])
        for year, html in archive_html.items()
    }
    for spec in EVENT_SPECS:
        source_url = spec["source_url"]
        pilot._validate_source_url(source_url)
        year = int(spec["year"])
        if year not in archive_link_sets:
            raise ValueError(f"Missing annual archive page for {year}")
        if source_url not in archive_link_sets[year]:
            raise ValueError(f"Frozen page is not linked from {year} archive: {source_url}")
        if source_url not in page_html:
            raise ValueError(f"Missing official page content: {source_url}")
        parsed = pilot.FederalReservePageParser.parse(page_html[source_url])
        available = pd.Timestamp(parsed["available_at_utc"])
        cluster = "fed_release_" + available.strftime("%Y%m%dT%H%M%SZ")
        raw = (
            page_content[source_url]
            if page_content is not None
            else page_html[source_url].encode("utf-8")
        )
        records.append(
            {
                "event_id": spec["event_id"],
                **parsed,
                "action_class": spec["action_class"],
                "inclusion_basis": spec["inclusion_basis"],
                "programme_episode_id": spec["programme_episode_id"],
                "announcement_cluster_id": cluster,
                "announcement_schedule_known": False,
                "expectation_consensus": None,
                "source_url": source_url,
                "annual_archive_url": ARCHIVE_URLS[year],
                "source_retrieval_timestamp": retrieved_at_utc,
                "source_content_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    catalogue = DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    validate_catalogue(catalogue)
    return catalogue.sort_values(["available_at_utc", "event_id"]).reset_index(drop=True)


def validate_catalogue(catalogue: DataFrame) -> None:
    missing = [column for column in OUTPUT_COLUMNS if column not in catalogue]
    if missing:
        raise ValueError(f"Catalogue missing columns: {missing}")
    if len(catalogue) != len(EVENT_SPECS):
        raise ValueError(f"Expected {len(EVENT_SPECS)} records, got {len(catalogue)}")
    for column in ("event_id", "source_url"):
        if catalogue[column].duplicated().any():
            raise ValueError(f"Duplicate {column} in catalogue")
    if catalogue["available_at_utc"].isna().any():
        raise ValueError("Every record requires an exact official release time")
    timestamps = pd.to_datetime(catalogue["available_at_utc"], utc=True, errors="raise")
    if timestamps.dt.tz is None:
        raise ValueError("Release timestamps must be timezone-aware")
    if catalogue["expectation_consensus"].notna().any():
        raise ValueError("Historical consensus must remain explicitly unavailable")
    if catalogue["announcement_schedule_known"].astype(bool).any():
        raise ValueError("No press-release clock is known to have been preannounced")
    expected_episodes = {
        "covid_2020_support",
        "covid_2021_lifecycle",
        "banking_stress_2023",
        "btfp_lifecycle_2024",
    }
    if set(catalogue["programme_episode_id"]) != expected_episodes:
        raise ValueError("Catalogue does not preserve the four frozen broad episodes")
    for url in catalogue["source_url"]:
        pilot._validate_source_url(str(url))


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def report_text(catalogue: DataFrame) -> str:
    year_counts = catalogue.groupby(pd.to_datetime(catalogue["available_at_utc"]).dt.year).size()
    episode_counts = catalogue.groupby("programme_episode_id").size()
    lines = [
        "# Federal Reserve liquidity-action catalogue",
        "",
        f"- Official announcement pages: `{len(catalogue)}`",
        f"- Distinct release clocks: `{catalogue['announcement_cluster_id'].nunique()}`",
        f"- Conservative broad episodes: `{catalogue['programme_episode_id'].nunique()}`",
        "- Market outcomes read: **No**",
        "- Profit used: **No**",
        "- Historical expectations: **unavailable**",
        "",
        "## Coverage",
        "",
        f"- By year: `{year_counts.to_dict()}`",
        f"- By broad episode: `{episode_counts.to_dict()}`",
        "",
        "## Interpretation boundary",
        "",
        "The pages provide exact announcement times, but they are not independent market "
        "replications. Most records are facility launches or changes within the single "
        "2020 pandemic response. Any outcome test must collapse simultaneous pages, keep "
        "whole broad episodes together, and describe the current evidence as concentrated.",
        "",
    ]
    return "\n".join(lines)


def execute(*, overwrite: bool = False, requester: Any = requests.get) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_fed_liquidity_catalogue":
            raise ValueError("Existing Fed liquidity catalogue result is not terminal")
        return result
    retrieved_at = g0.utc_now()
    archive_html: dict[int, str] = {}
    source_manifest: list[dict[str, Any]] = []
    for year, url in ARCHIVE_URLS.items():
        html, content = _fetch(url, requester=requester)
        archive_html[year] = html
        source_manifest.append(
            {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "kind": "annual_archive"}
        )
    page_html: dict[str, str] = {}
    page_content: dict[str, bytes] = {}
    for spec in EVENT_SPECS:
        url = spec["source_url"]
        html, content = _fetch(url, requester=requester)
        page_html[url] = html
        page_content[url] = content
        source_manifest.append(
            {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "kind": "event_page"}
        )
    catalogue = build_catalogue(
        archive_html=archive_html,
        page_html=page_html,
        page_content=page_content,
        retrieved_at_utc=retrieved_at,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_json(source_manifest, MANIFEST_PATH)
    freeze = {
        "schema_version": 1,
        "status": "completed_fed_liquidity_catalogue",
        "created_at_utc": retrieved_at,
        "outcomes_read": False,
        "profit_used": False,
        "selection_vocabulary": list(SELECTION_VOCABULARY),
        "action_terms": list(ACTION_TERMS),
        "event_pages": len(catalogue),
        "release_clocks": int(catalogue["announcement_cluster_id"].nunique()),
        "programme_episodes": int(catalogue["programme_episode_id"].nunique()),
        "year_counts": {
            str(year): int(count)
            for year, count in catalogue.groupby(
                pd.to_datetime(catalogue["available_at_utc"]).dt.year
            ).size().items()
        },
        "episode_counts": {
            str(name): int(count)
            for name, count in catalogue.groupby("programme_episode_id").size().items()
        },
        "expectation_policy": "Unavailable; no direction is assigned from these pages.",
        "independence_policy": (
            "Collapse identical release clocks and keep each programme episode whole."
        ),
    }
    g0.atomic_write_json(freeze, FREEZE_PATH)
    REPORT_PATH.write_text(report_text(catalogue), encoding="utf-8", newline="\n")
    result = {
        **freeze,
        "status": "completed_fed_liquidity_catalogue",
        "analysis_script": _artifact(ANALYSIS_PATH),
        "artifacts": {
            "catalogue": _artifact(CATALOGUE_PATH),
            "manifest": _artifact(MANIFEST_PATH),
            "freeze": _artifact(FREEZE_PATH),
            "report": _artifact(REPORT_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {"status": "ready_not_executed", "event_pages": len(EVENT_SPECS)},
                indent=2,
            )
        )
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
