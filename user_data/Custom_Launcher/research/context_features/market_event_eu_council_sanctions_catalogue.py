"""Build an outcome-blind Council of the EU sanctions publication catalogue."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy/"
    "eu_council_sanctions_catalogue_20260913a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "eu_council_sanctions_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "eu_council_sanctions_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "eu_council_sanctions_source_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "eu_council_sanctions_catalogue_result.json"

MIRROR_HOST = "skribi.consilium.europa.eu"
PUBLIC_HOST = "www.consilium.europa.eu"
ARCHIVE_PATH = "/en/press/press-releases/"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
SANCTIONS_TOPIC_ID = "130644"
REQUEST_TIMEOUT = 45
USER_AGENT = "FreqTradeStuff Objective02b EU-sanctions-source-audit/1.0"
SENSITIVITY_OFFSETS_HOURS = (-2, -1, 0, 1, 2)

ALIGNMENT_RE = re.compile(r"alignment of (?:certain )?(?:third )?countr", re.I)
STATEMENT_RE = re.compile(
    r"(?:^|:\s)(?:statement|declaration|remarks|press statement|"
    r"european council conclusions)",
    re.I,
)
RENEWAL_RE = re.compile(
    r"\b(?:renew(?:s|ed|al)?|prolong(?:s|ed)?|extend(?:s|ed)?|roll(?:s|ed)? over)\b",
    re.I,
)
EASING_RE = re.compile(
    r"\b(?:lift(?:s|ed|ing)?|suspend(?:s|ed|ing)?|remove(?:s|d|ing)?|"
    r"delist(?:s|ed|ing)?|ease(?:s|d|ing)?|relax(?:es|ed|ing)?)\b",
    re.I,
)
EXEMPTION_ACTION_RE = re.compile(
    r"\b(?:introduc(?:e|es|ed|ing|tion)|creat(?:e|es|ed|ing|ion)|"
    r"adopt(?:s|ed|ing|ion))\b.{0,80}\b(?:new\s+)?(?:humanitarian\s+)?"
    r"(?:exemptions?|exceptions?)\b",
    re.I,
)
NEW_EXEMPTION_ACTION_RE = re.compile(
    r"\b(?:introduc(?:e|es|ed|ing|tion)|creat(?:e|es|ed|ing|ion)|"
    r"adopt(?:s|ed|ing|ion))\b.{0,80}\bnew\b.{0,40}\b"
    r"(?:humanitarian\s+)?(?:exemptions?|exceptions?)\b",
    re.I,
)
FRAMEWORK_RE = re.compile(
    r"(?:\b(?:establish(?:es|ed|ing)?|creat(?:e|es|ed|ing)|set(?:s)?\s+up|"
    r"introduc(?:e|es|ed|ing))\b.{0,80}\b(?:framework|regime)\b|"
    r"\badopt(?:s|ed|ing)?\b.{0,80}\bnew\b.{0,40}\b(?:framework|regime)\b)",
    re.I,
)
TIGHTENING_ACTION_RE = re.compile(
    r"\b(?:add(?:s|ed|ing)?|list(?:s|ed|ing)|designat(?:e|es|ed|ing)|"
    r"(?:re)?impos(?:e|es|ed|ing)|introduc(?:e|es|ed|ing)|tighten(?:s|ed|ing)?|"
    r"strengthen(?:s|ed|ing)?|expand(?:s|ed|ing)?|broaden(?:s|ed|ing)?|"
    r"widen(?:s|ed|ing)?)\b|\bto\s+list\b",
    re.I,
)
SANCTION_VERB_RE = re.compile(
    r"\b(?:the\s+)?(?:EU|Council)\s+(?:sanctioned|sanctioning)\b|"
    r"\b(?:the\s+)?(?:EU|Council)\s+sanctions\s+(?!against\b|on\b|"
    r"regime\b|framework\b|list\b|package\b|measures?\b|policy\b|rules?\b|"
    r"renew\w*\b|prolong\w*\b|extend\w*\b)",
    re.I,
)
PACKAGE_ACTION_RE = re.compile(
    r"\b(?:adopt(?:s|ed|ing)?|agree(?:s|d|ing)?)\b.{0,80}"
    r"\b(?:\d+(?:st|nd|rd|th)\s+)?package\b",
    re.I,
)
SANCTIONS_CONTEXT_RE = re.compile(
    r"\b(?:sanction|restrictive measure|asset freeze|travel ban|listing)\w*\b",
    re.I,
)


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def archive_url(page: int) -> str:
    return (
        f"https://{MIRROR_HOST}{ARCHIVE_PATH}?dateFrom={START_DATE}"
        f"&dateTo={END_DATE}&topic={SANCTIONS_TOPIC_ID}&page={page}"
    )


def valid_archive_response_url(url: str, *, page: int) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    query = parse_qs(parsed.query)
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() == MIRROR_HOST
        and parsed.path == ARCHIVE_PATH
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.fragment
        and query
        == {
            "dateFrom": [START_DATE],
            "dateTo": [END_DATE],
            "topic": [SANCTIONS_TOPIC_ID],
            "page": [str(page)],
        }
    )


def fetch_page(
    page: int,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[bytes, dict[str, Any]]:
    url = archive_url(page)
    response = requester(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=timeout,
        allow_redirects=False,
    )
    content = bytes(response.content)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Council archive page {page} returned {response.status_code}")
    if not valid_archive_response_url(str(response.url), page=page):
        raise ValueError("Council response did not come from the frozen official mirror URL")
    if b"Press releases and statements" not in content:
        raise ValueError("Council response does not contain the expected archive heading")
    return content, {
        "page": page,
        "url": str(response.url),
        "http_status": int(response.status_code),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def total_pages(content: bytes) -> int:
    soup = BeautifulSoup(content, "html.parser")
    pages = {1}
    for link in soup.select('nav[aria-label="Select a page"] a[href]'):
        values = parse_qs(urlparse(str(link["href"])).query).get("page", [])
        if len(values) == 1 and values[0].isdigit():
            pages.add(int(values[0]))
    maximum = max(pages)
    if maximum < 1 or maximum > 100:
        raise ValueError(f"Council archive page count is implausible: {maximum}")
    return maximum


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def has_direct_easing_action(text: str, combined: str) -> bool:
    return bool(EASING_RE.search(text) and SANCTIONS_CONTEXT_RE.search(combined))


def has_framework_action(text: str, combined: str) -> bool:
    return bool(FRAMEWORK_RE.search(text) and SANCTIONS_CONTEXT_RE.search(combined))


def has_tightening_action(text: str, combined: str) -> bool:
    return bool(
        (
            TIGHTENING_ACTION_RE.search(text)
            or SANCTION_VERB_RE.search(text)
            or PACKAGE_ACTION_RE.search(text)
        )
        and SANCTIONS_CONTEXT_RE.search(combined)
    )


def classify_renewal(combined: str) -> tuple[str, str]:
    if has_direct_easing_action(combined, combined) or NEW_EXEMPTION_ACTION_RE.search(
        combined
    ):
        return "include", "substantive_easing_or_removal"
    return "exclude", "routine_renewal_or_extension"


def classify_release(title: str, summary: str) -> tuple[str, str]:
    combined = clean_text(f"{title} {summary}")
    if ALIGNMENT_RE.search(combined):
        return "exclude", "third_country_alignment_notice"
    renewal = bool(RENEWAL_RE.search(combined))
    title_easing = bool(
        has_direct_easing_action(title, combined)
        or EXEMPTION_ACTION_RE.search(title)
    )
    title_framework = has_framework_action(title, combined)
    title_tightening = has_tightening_action(title, combined)
    if title_easing:
        return "include", "substantive_easing_or_removal"
    if title_framework:
        return "include", "new_sanctions_framework_or_regime"
    if title_tightening:
        return "include", "substantive_tightening_or_new_listing"
    if renewal:
        return classify_renewal(combined)
    if has_direct_easing_action(combined, combined) or EXEMPTION_ACTION_RE.search(
        combined
    ):
        return "include", "substantive_easing_or_removal"
    if has_framework_action(combined, combined):
        return "include", "new_sanctions_framework_or_regime"
    tightening_action = has_tightening_action(combined, combined)
    if STATEMENT_RE.search(title) and not tightening_action:
        return "exclude", "statement_or_conclusions_without_new_action"
    if tightening_action:
        return "include", "substantive_tightening_or_new_listing"
    if SANCTIONS_CONTEXT_RE.search(combined):
        return "manual_review", "sanctions_context_but_action_unclear"
    return "exclude", "topic_match_without_discrete_sanctions_action"


def parse_archive_page(content: bytes, *, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(content, "html.parser")
    rows: list[dict[str, Any]] = []
    for date_group in soup.select("li.gsc-excerpt-list__item"):
        heading = date_group.select_one("h2.gsc-excerpt-list__item-date")
        if heading is None:
            continue
        display_date = pd.to_datetime(
            clean_text(heading.get_text(" ")), format="%d %B %Y", errors="raise"
        )
        for item in date_group.select("li.gsc-excerpt-item"):
            link = item.select_one("a.gsc-excerpt-item__link[href]")
            title_node = item.select_one(".gsc-excerpt-item__title")
            time_node = item.select_one("time.gsc-time-badge")
            if link is None or title_node is None or time_node is None:
                raise ValueError("Council archive entry lacks link, title, or minute clock")
            path = str(link["href"])
            if not path.startswith("/en/press/press-releases/"):
                raise ValueError(f"Unexpected Council release path: {path}")
            title = clean_text(title_node.get_text(" "))
            summary_node = item.select_one("#excerpt-text")
            summary = clean_text(summary_node.get_text(" ")) if summary_node else ""
            entity_node = item.select_one("#excerpt-tag")
            entity = clean_text(entity_node.get_text(" ")) if entity_node else ""
            display_time = clean_text(time_node.get_text(" "))
            if re.fullmatch(r"\d{2}:\d{2}", display_time) is None:
                raise ValueError(f"Unexpected Council display time: {display_time}")
            displayed = pd.Timestamp(
                f"{display_date.date().isoformat()}T{display_time}:00Z"
            )
            attribute = str(time_node.get("datetime", ""))
            if attribute:
                attribute_time = pd.to_datetime(attribute, errors="raise")
                if (
                    attribute_time.date() != display_date.date()
                    or attribute_time.strftime("%H:%M") != display_time
                ):
                    raise ValueError("Council visible and datetime clocks disagree")
            selection_status, action_class = classify_release(title, summary)
            digest = hashlib.sha256(path.encode()).hexdigest()[:12]
            rows.append(
                {
                    "event_id": f"eu_sanctions_{displayed:%Y%m%d}_{digest}",
                    "displayed_publication_clock": displayed.isoformat(),
                    "displayed_clock_timezone_status": (
                        "utc_inferred_from_current_official_rss_same_clock;"
                        "historical_offset_not_embedded"
                    ),
                    "strict_minute_attribution_eligible": False,
                    "required_timing_sensitivity_hours": "-2|-1|0|1|2",
                    "title": title,
                    "summary": summary,
                    "publishing_entity": entity,
                    "selection_status": selection_status,
                    "action_class": action_class,
                    "archive_page": page,
                    "source_path": path,
                    "source_url": f"https://{PUBLIC_HOST}{path}",
                    "archive_source_url": archive_url(page),
                    "market_expectation_status": "unavailable_not_imputed",
                    "selection_method": "outcome_blind_fixed_text_rules_v1",
                }
            )
    if not rows:
        raise ValueError(f"Council archive page {page} contained no releases")
    return rows


def build_catalogue(page_contents: Sequence[bytes]) -> DataFrame:
    records: list[dict[str, Any]] = []
    for page, content in enumerate(page_contents, start=1):
        records.extend(parse_archive_page(content, page=page))
    catalogue = DataFrame.from_records(records)
    if catalogue.empty or catalogue["event_id"].duplicated().any():
        raise ValueError("Council catalogue is empty or has duplicate release paths")
    timestamps = pd.to_datetime(catalogue["displayed_publication_clock"], utc=True)
    if not timestamps.between(
        pd.Timestamp(f"{START_DATE}T00:00:00Z"),
        pd.Timestamp(f"{END_DATE}T23:59:59Z"),
        inclusive="both",
    ).all():
        raise ValueError("Council release falls outside the frozen date range")
    catalogue["whole_event_partition"] = timestamps.dt.year.map(
        {
            2020: "context_only_2020",
            2021: "development_2021_2023",
            2022: "development_2021_2023",
            2023: "development_2021_2023",
            2024: "internal_validation_2024",
            2025: "holdout_2025",
        }
    )
    catalogue["episode_id"] = ""
    included = catalogue["selection_status"].eq("include")
    dates = timestamps.dt.strftime("%Y%m%d")
    catalogue.loc[included, "episode_id"] = "eu_sanctions_episode_" + dates[included]
    catalogue.loc[~included, "episode_id"] = (
        "not_selected_" + catalogue.loc[~included, "event_id"].astype(str)
    )
    catalogue["episode_members"] = catalogue.groupby("episode_id")["event_id"].transform(
        "size"
    )
    forbidden = ("profit", "future_return", "price", "volume", "direction")
    if any(
        token in column.casefold()
        for column in catalogue.columns
        for token in forbidden
    ):
        raise ValueError("Council source catalogue contains a market outcome column")
    return catalogue.sort_values(
        ["displayed_publication_clock", "event_id"], kind="stable"
    ).reset_index(drop=True)


def count_catalogue(catalogue: DataFrame) -> DataFrame:
    timestamps = pd.to_datetime(catalogue["displayed_publication_clock"], utc=True)
    counted = catalogue.assign(year=timestamps.dt.year)
    return (
        counted.groupby(["year", "selection_status", "action_class"], sort=True)
        .agg(releases=("event_id", "nunique"), episodes=("episode_id", "nunique"))
        .reset_index()
    )


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    selected = catalogue[catalogue["selection_status"].eq("include")]
    return {
        "schema_version": 1,
        "status": "frozen_outcome_blind_eu_sanctions_source_pilot",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "plain_question": (
            "Can a complete, reproducible Council sanctions catalogue be defined before "
            "opening any crypto outcome?"
        ),
        "archive_period": {"start": START_DATE, "end": END_DATE},
        "archive_topic_id": SANCTIONS_TOPIC_ID,
        "archive_pages": int(catalogue["archive_page"].max()),
        "all_release_rows": len(catalogue),
        "selected_substantive_rows": len(selected),
        "selected_whole_date_episodes": int(selected["episode_id"].nunique()),
        "manual_review_rows": int(catalogue["selection_status"].eq("manual_review").sum()),
        "counts": counts.to_dict(orient="records"),
        "episode_rule": (
            "All selected substantive Council releases on the same displayed UTC date "
            "are one conservative episode, even when they concern different targets."
        ),
        "timing_rule": {
            "status": (
                "The page clock matches the current official Council RSS UTC clock, but "
                "historical pages do not embed an offset."
            ),
            "strict_minute_attribution_eligible": False,
            "required_offsets_hours": list(SENSITIVITY_OFFSETS_HOURS),
            "allowed_initial_horizons": ["1h", "4h", "24h"],
            "forbidden_initial_horizons": ["5m", "15m", "30m"],
        },
        "selection_rules": {
            "include": [
                "new or additional listings, sanctions, restrictive measures, or packages",
                "new sanctions frameworks or regimes",
                "substantive easing, suspension, lifting, or removal",
            ],
            "exclude": [
                "third-country alignment notices",
                "routine renewals or extensions without a new action",
                "statements or conclusions without a new action",
                "topic matches without a discrete sanctions action",
            ],
        },
        "boundaries": [
            "Selection is based only on official title and excerpt text.",
            (
                "The fixed official topic archive returned only 3 total rows in "
                "2020-2022. This catalogue is source-complete for the rows returned by "
                "that filtered archive, not a complete historical universe of all EU "
                "sanctions before 2023."
            ),
            "No market outcome, memorable price move, or profit was used to select events.",
            "The catalogue does not encode whether an action was expected or its likely sign.",
            "Any later market test must preserve whole-date episodes and timing sensitivity.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_eu_council_sanctions_source_pilot":
        raise ValueError("Existing Council sanctions result is invalid")
    for value in result.get("artifacts", {}).values():
        path = Path(value["path"])
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing Council sanctions artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    first_content, first_manifest = fetch_page(1)
    page_count = total_pages(first_content)
    contents = [first_content]
    manifest_rows = [first_manifest]
    for page in range(2, page_count + 1):
        content, manifest = fetch_page(page)
        contents.append(content)
        manifest_rows.append(manifest)
    catalogue = build_catalogue(contents)
    counts = count_catalogue(catalogue)
    freeze = freeze_document(catalogue, counts)
    source_manifest = {
        "schema_version": 1,
        "retrieved_at_utc": g0.utc_now(),
        "official_mirror_host": MIRROR_HOST,
        "public_host": PUBLIC_HOST,
        "archive_filter": {
            "dateFrom": START_DATE,
            "dateTo": END_DATE,
            "topic": SANCTIONS_TOPIC_ID,
        },
        "pages": manifest_rows,
        "browser_equivalence_check": (
            "Main and mirror archive result URLs and a 2024 release DOM snapshot were "
            "identical in the bounded 2026-09-10 manual source check."
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(source_manifest, SOURCE_MANIFEST_PATH)
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_eu_council_sanctions_source_pilot",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "profit_used": False,
        "all_release_rows": len(catalogue),
        "selected_substantive_rows": int(
            catalogue["selection_status"].eq("include").sum()
        ),
        "selected_whole_date_episodes": int(
            catalogue.loc[catalogue["selection_status"].eq("include"), "episode_id"].nunique()
        ),
        "manual_review_rows": int(
            catalogue["selection_status"].eq("manual_review").sum()
        ),
        "artifacts": {
            "catalogue": artifact(CATALOGUE_PATH),
            "counts": artifact(COUNTS_PATH),
            "source_manifest": artifact(SOURCE_MANIFEST_PATH),
            "freeze": artifact(FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "outcomes_will_be_read": False,
                    "profit_will_be_used": False,
                    "timing_sensitivity_hours": list(SENSITIVITY_OFFSETS_HOURS),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
