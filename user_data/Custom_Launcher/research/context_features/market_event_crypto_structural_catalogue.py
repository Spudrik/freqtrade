"""Freeze an outcome-blind catalogue of BTC, ETH, and SEC structural events."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# The frozen declarative source rows are kept one-record-per-line for auditability.
# ruff: noqa: E402, E501
import argparse
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from pandas import DataFrame
from pypdf import PdfReader


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
    "crypto_structural_events_catalogue_20260913a"
)
CATALOGUE_PATH = OUTPUT_ROOT / "crypto_structural_events_catalogue.csv"
COUNTS_PATH = OUTPUT_ROOT / "crypto_structural_events_counts.csv"
SOURCE_MANIFEST_PATH = OUTPUT_ROOT / "official_source_manifest.json"
FREEZE_PATH = OUTPUT_ROOT / "crypto_structural_events_source_freeze.json"
REPORT_PATH = OUTPUT_ROOT / "crypto_structural_events_catalogue_report.md"
RESULT_PATH = OUTPUT_ROOT / "crypto_structural_events_catalogue_result.json"
EXPECTED_OUTPUTS = (
    CATALOGUE_PATH,
    COUNTS_PATH,
    SOURCE_MANIFEST_PATH,
    FREEZE_PATH,
    REPORT_PATH,
    RESULT_PATH,
)

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
REQUEST_TIMEOUT = 45
MAX_RETRIEVAL_WORKERS = 4
USER_AGENT = "FreqTradeStuff Objective02b structural-source-freeze/1.0"

BITCOIN_RELEASES_URL = "https://bitcoincore.org/en/releases/"
BITCOIN_TAPROOT_URL = "https://bitcoincore.org/en/releases/0.21.1/"
BITCOIN_CHAINPARAMS_URL = (
    "https://raw.githubusercontent.com/bitcoin/bitcoin/master/src/kernel/chainparams.cpp"
)
ETHEREUM_HISTORY_URL = "https://ethereum.org/ethereum-forks/"

APPROVED_HOSTS = {
    "bitcoincore.org",
    "raw.githubusercontent.com",
    "ethereum.org",
    "blog.ethereum.org",
    "www.sec.gov",
    "blockstream.info",
    "etherscan.io",
}
ETHEREUM_GENESIS_TIME = "2020-12-01T12:00:23Z"
ETHEREUM_EPOCH_SECONDS = 384


@dataclass(frozen=True)
class SourceSpec:
    url: str
    source_class: str
    media_type: str
    expected_tokens: tuple[str, ...]


def _sec_pdf(exchange: str, year: int, release: str, suffix: str = "") -> str:
    return (
        f"https://www.sec.gov/files/rules/sro/{exchange}/{year}/"
        f"{release}{suffix}.pdf"
    )


SEC_ORDERS: tuple[dict[str, Any], ...] = (
    {"release": "34-88284", "date": "2020-02-26", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2019-39", "United States Bitcoin and Treasury Investment Trust"),)},
    {"release": "34-93559", "date": "2021-11-12", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2021-019", "VanEck Bitcoin Trust"),)},
    {"release": "34-93700", "date": "2021-12-01", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2021-024", "WisdomTree Bitcoin Trust"),)},
    {"release": "34-93859", "date": "2021-12-22", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2021-31", "Valkyrie Bitcoin Fund"),)},
    {"release": "34-93860", "date": "2021-12-22", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2021-029", "Kryptoin Bitcoin ETF Trust"),)},
    {"release": "34-94006", "date": "2022-01-20", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2021-37", "First Trust SkyBridge Bitcoin ETF Trust"),)},
    {"release": "34-94080", "date": "2022-01-27", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2021-039", "Wise Origin Bitcoin Trust"),)},
    {"release": "34-94395", "date": "2022-03-10", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2021-57", "NYDIG Bitcoin ETF"),)},
    {"release": "34-94396", "date": "2022-03-10", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2021-052", "Global X Bitcoin Trust"),)},
    {"release": "34-94571", "date": "2022-03-31", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2021-051", "ARK 21Shares Bitcoin ETF"),)},
    {"release": "34-94999", "date": "2022-05-27", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2021-67", "One River Carbon Neutral Bitcoin Trust"),)},
    {"release": "34-95179", "date": "2022-06-29", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2021-89", "Bitwise Bitcoin ETP Trust"),)},
    {"release": "34-95180", "date": "2022-06-29", "exchange": "nysearca", "action": "denied", "members": (("SR-NYSEArca-2021-90", "Grayscale Bitcoin Trust"),)},
    {"release": "34-96011", "date": "2022-10-11", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2022-006", "WisdomTree Bitcoin Trust"),)},
    {"release": "34-96751", "date": "2023-01-26", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2022-031", "ARK 21Shares Bitcoin ETF"),)},
    {"release": "34-97102", "date": "2023-03-10", "exchange": "cboebzx", "action": "denied", "members": (("SR-CboeBZX-2022-035", "VanEck Bitcoin Trust"),)},
    {"release": "34-99306", "date": "2024-01-10", "exchange": "nysearca", "action": "approved", "members": (
        ("SR-NYSEArca-2021-90", "Grayscale Bitcoin Trust"), ("SR-NYSEArca-2023-44", "Bitwise Bitcoin ETF"),
        ("SR-NYSEArca-2023-58", "Hashdex Bitcoin ETF"), ("SR-NASDAQ-2023-016", "iShares Bitcoin Trust"),
        ("SR-NASDAQ-2023-019", "Valkyrie Bitcoin Fund"), ("SR-CboeBZX-2023-028", "ARK 21Shares Bitcoin ETF"),
        ("SR-CboeBZX-2023-038", "Invesco Galaxy Bitcoin ETF"), ("SR-CboeBZX-2023-040", "VanEck Bitcoin Trust"),
        ("SR-CboeBZX-2023-042", "WisdomTree Bitcoin Fund"), ("SR-CboeBZX-2023-044", "Fidelity Wise Origin Bitcoin Fund"),
        ("SR-CboeBZX-2023-072", "Franklin Bitcoin ETF"),
    )},
    {"release": "34-100224", "date": "2024-05-23", "exchange": "nysearca", "action": "approved", "asset": "ETH", "members": (
        ("SR-NYSEArca-2023-70", "Grayscale Ethereum Trust"), ("SR-NYSEArca-2024-31", "Bitwise Ethereum ETF"),
        ("SR-NASDAQ-2023-045", "iShares Ethereum Trust"), ("SR-CboeBZX-2023-069", "VanEck Ethereum Trust"),
        ("SR-CboeBZX-2023-070", "ARK 21Shares Ethereum ETF"), ("SR-CboeBZX-2023-087", "Invesco Galaxy Ethereum ETF"),
        ("SR-CboeBZX-2023-095", "Fidelity Ethereum Fund"), ("SR-CboeBZX-2024-018", "Franklin Ethereum ETF"),
    )},
    {"release": "34-100541", "date": "2024-07-17", "exchange": "nysearca", "action": "approved", "asset": "ETH", "members": (
        ("SR-NYSEArca-2024-44", "Grayscale Ethereum Mini Trust"), ("SR-NYSEArca-2024-53", "ProShares Ethereum ETF"),
    )},
    {"release": "34-100610", "date": "2024-07-26", "exchange": "nysearca", "action": "approved", "members": (
        ("SR-NYSEArca-2024-45", "Grayscale Bitcoin Mini Trust"), ("SR-CboeBZX-2023-101", "Pando Asset Spot Bitcoin Trust"),
    )},
    {"release": "34-101998", "date": "2024-12-19", "exchange": "nasdaq", "action": "approved", "asset": "BTC+ETH", "members": (
        ("SR-NASDAQ-2024-028", "Hashdex Nasdaq Crypto Index US ETF"), ("SR-CboeBZX-2024-091", "Franklin Crypto Index ETF"),
    )},
    {"release": "34-103570", "date": "2025-07-29", "exchange": "nysearca", "action": "approved", "asset": "BTC+ETH", "suffix": "qewre", "members": (
        ("SR-NYSEArca-2025-15", "Bitwise Bitcoin and Ethereum ETF"),
    )},
)


ETH_CLOCKS: tuple[dict[str, Any], ...] = (
    {"clock": "eth_2020_muir", "time": "2020-01-02T08:30:49Z", "components": ("Muir Glacier",), "block": "9200000", "schedule": "2019-12-23", "schedule_url": "https://blog.ethereum.org/2019/12/23/ethereum-muir-glacier-upgrade-announcement", "status": "anticipated_block_height_variable_wall_clock", "bundle": "ethereum_difficulty_bomb_delay_program"},
    {"clock": "eth_2020_beacon_genesis", "time": "2020-12-01T12:00:23Z", "components": ("Beacon Chain genesis",), "epoch": "0", "schedule": "2020-11-04", "final": "2020-11-27", "schedule_url": "https://blog.ethereum.org/2020/11/04/eth2-quick-update-no-19", "final_url": "https://blog.ethereum.org/2020/11/27/eth2-quick-update-no-21", "status": "conditional_deposit_threshold_then_confirmed", "bundle": "ethereum_beacon_chain_launch_program"},
    {"clock": "eth_2021_berlin", "time": "2021-04-15T10:07:03Z", "components": ("Berlin",), "block": "12244000", "schedule": "2021-03-08", "schedule_url": "https://blog.ethereum.org/2021/03/08/ethereum-berlin-upgrade-announcement", "status": "anticipated_block_height_variable_wall_clock", "bundle": "ethereum_execution_upgrade_program"},
    {"clock": "eth_2021_london", "time": "2021-08-05T12:33:42Z", "components": ("London",), "block": "12965000", "schedule": "2021-07-15", "schedule_url": "https://blog.ethereum.org/2021/07/15/london-mainnet-announcement", "status": "anticipated_block_height_variable_wall_clock", "bundle": "ethereum_execution_upgrade_program"},
    {"clock": "eth_2021_altair", "time": "2021-10-27T10:56:23Z", "components": ("Altair",), "epoch": "74240", "schedule": "2021-10-05", "schedule_url": "https://blog.ethereum.org/2021/10/05/altair-announcement", "status": "anticipated_exact_epoch", "bundle": "ethereum_beacon_upgrade_program"},
    {"clock": "eth_2021_arrow", "time": "2021-12-09T19:55:23Z", "components": ("Arrow Glacier",), "block": "13773000", "schedule": "2021-11-10", "schedule_url": "https://blog.ethereum.org/2021/11/10/arrow-glacier-announcement", "status": "anticipated_block_height_variable_wall_clock", "bundle": "ethereum_difficulty_bomb_delay_program"},
    {"clock": "eth_2022_gray", "time": "2022-06-30T10:54:04Z", "components": ("Gray Glacier",), "block": "15050000", "schedule": "2022-06-16", "schedule_url": "https://blog.ethereum.org/2022/06/16/gray-glacier-announcement", "status": "anticipated_block_height_variable_wall_clock", "bundle": "ethereum_difficulty_bomb_delay_program"},
    {"clock": "eth_2022_bellatrix", "time": "2022-09-06T11:34:47Z", "components": ("Bellatrix",), "epoch": "144896", "schedule": "2022-08-24", "schedule_url": "https://blog.ethereum.org/2022/08/24/mainnet-merge-announcement", "status": "anticipated_exact_epoch", "bundle": "ethereum_merge_program"},
    {"clock": "eth_2022_paris", "time": "2022-09-15T06:42:59Z", "components": ("Paris",), "block": "15537394", "ttd": "58750000000000000000000", "schedule": "2022-08-24", "schedule_url": "https://blog.ethereum.org/2022/08/24/mainnet-merge-announcement", "status": "conditional_terminal_total_difficulty", "bundle": "ethereum_merge_program"},
    {"clock": "eth_2023_shapella", "time": "2023-04-12T22:27:35Z", "components": ("Shanghai", "Capella"), "block": "17034870", "epoch": "194048", "schedule": "2023-03-28", "schedule_url": "https://blog.ethereum.org/2023/03/28/shapella-mainnet-announcement", "status": "anticipated_exact_epoch", "bundle": "ethereum_shapella_program"},
    {"clock": "eth_2024_dencun", "time": "2024-03-13T13:55:35Z", "components": ("Cancun", "Deneb"), "block": "19426587", "epoch": "269568", "schedule": "2024-02-27", "schedule_url": "https://blog.ethereum.org/2024/02/27/dencun-mainnet-announcement", "status": "anticipated_exact_epoch", "bundle": "ethereum_dencun_program"},
    {"clock": "eth_2025_pectra", "time": "2025-05-07T10:05:11Z", "components": ("Prague", "Electra"), "block": "22431084", "epoch": "364032", "schedule": "2025-04-23", "schedule_url": "https://blog.ethereum.org/2025/04/23/pectra-mainnet", "status": "anticipated_exact_epoch", "bundle": "ethereum_pectra_program"},
    {"clock": "eth_2025_fusaka", "time": "2025-12-03T21:49:11Z", "components": ("Osaka", "Fulu"), "block": "23935694", "epoch": "411392", "schedule": "2025-11-06", "schedule_url": "https://blog.ethereum.org/2025/11/06/fusaka-mainnet-announcement", "status": "anticipated_exact_epoch", "bundle": "ethereum_fusaka_program"},
    {"clock": "eth_2025_bpo1", "time": "2025-12-09T14:21:11Z", "components": ("BPO1",), "epoch": "412672", "schedule": "2025-11-06", "schedule_url": "https://blog.ethereum.org/2025/11/06/fusaka-mainnet-announcement", "status": "anticipated_exact_epoch", "bundle": "ethereum_fusaka_program"},
)


BTC_CLOCKS: tuple[dict[str, str], ...] = (
    {"clock": "btc_2020_halving", "name": "2020 subsidy halving", "time": "2020-05-11T19:23:43Z", "block": "630000", "hash": "000000000000000000024bead8df69990852c202db0e0097c1a12ea637d7e96d", "status": "anticipated_height_variable_wall_clock", "bundle": "bitcoin_subsidy_schedule"},
    {"clock": "btc_2021_taproot", "name": "Taproot", "time": "2021-11-14T05:15:27Z", "block": "709632", "hash": "0000000000000000000687bca986194dc2c1f949318629b44bb54ec0a94d8244", "status": "conditional_signalling_then_locked_height", "bundle": "bitcoin_taproot_deployment", "schedule": "2021-05-01"},
    {"clock": "btc_2024_halving", "name": "2024 subsidy halving", "time": "2024-04-20T00:09:27Z", "block": "840000", "hash": "0000000000000000000320283a032748cef8227873ff4872689bf23f1cda83a5", "status": "anticipated_height_variable_wall_clock", "bundle": "bitcoin_subsidy_schedule"},
)


def sec_process_bundle(member_id: str, product: str) -> str:
    text = product.casefold()
    if "ark 21shares" in text:
        return "sec_ark_21shares_bitcoin_process"
    if "vaneck bitcoin" in text:
        return "sec_vaneck_bitcoin_process"
    if "wisdomtree bitcoin" in text:
        return "sec_wisdomtree_bitcoin_process"
    if "grayscale bitcoin trust" in text:
        return "sec_grayscale_bitcoin_trust_process"
    return "sec_" + re.sub(r"[^a-z0-9]+", "_", member_id.casefold()).strip("_")


def _protocol_row(
    *, family: str, subfamily: str, member_id: str, clock_id: str,
    process_bundle_id: str, affected_asset: str, event_name: str,
    actual: str, official_url: str, schedule_url: str = "",
    schedule: str = "", final: str = "", block: str = "", epoch: str = "",
    ttd: str = "", anticipated_status: str, provenance: str,
    timestamp_provenance: str, actual_time_derivation: str = "",
) -> dict[str, Any]:
    if anticipated_status == "anticipated_exact_epoch":
        anticipation = "eligible_after_frozen_final_epoch_schedule"
    elif anticipated_status == "conditional_signalling_then_locked_height":
        anticipation = (
            "requires_documented_lock_in_plus_contemporaneous_block_arrival_estimate"
        )
    else:
        anticipation = "requires_contemporaneous_arrival_or_condition_estimate"
    conditional_pre_lock_in = (
        anticipated_status == "conditional_signalling_then_locked_height"
    )
    final_schedule = "" if conditional_pre_lock_in else (final or schedule)
    return {
        "member_id": member_id,
        "family": family,
        "subfamily": subfamily,
        "activation_clock_id": clock_id,
        "date_episode_id": f"{family}_{actual[:10]}",
        "process_bundle_id": process_bundle_id,
        "affected_asset": affected_asset,
        "event_name": event_name,
        "order_document_id": "",
        "decision_action": "",
        "official_source_url": official_url,
        "schedule_source_url": schedule_url,
        "actual_time_source_url": provenance,
        "actual_time_or_date": actual,
        "actual_time_precision": "exact_second",
        "activation_block": block,
        "activation_epoch": epoch,
        "activation_ttd": ttd,
        "first_official_schedule_date_or_time": schedule,
        "first_schedule_precision": "date_only" if schedule else "unavailable",
        "final_schedule_knowable_date_or_time": final_schedule,
        "final_schedule_precision": "date_only" if final_schedule else "unavailable",
        "anticipated_or_conditional_status": anticipated_status,
        "reschedule_revision_lineage": (
            "The 2021-05-01 Taproot release preceded miner lock-in; a documented "
            "lock-in source is required before final schedule knowability."
            if conditional_pre_lock_in
            else "Retain dated schedule revisions; final activation time must not be "
            "backfilled into earlier anticipation windows."
        ),
        "expectation_status": "unavailable",
        "exact_minute_eligible": True,
        "immediate_1h_4h_eligible": True,
        "anticipation_eligibility": anticipation,
        "statistical_inference_eligible": False,
        "statistical_status": (
            "case_study_or_prospective"
            if family == "bitcoin_protocol"
            else "small_heterogeneous_programme_linked_not_model_ready"
        ),
        "timestamp_provenance": timestamp_provenance,
        "actual_time_derivation": actual_time_derivation,
    }


def build_catalogue() -> DataFrame:
    rows: list[dict[str, Any]] = []
    for item in BTC_CLOCKS:
        source = BITCOIN_TAPROOT_URL if item["name"] == "Taproot" else BITCOIN_CHAINPARAMS_URL
        rows.append(
            _protocol_row(
                family="bitcoin_protocol", subfamily="consensus_activation",
                member_id=item["clock"], clock_id=item["clock"],
                process_bundle_id=item["bundle"], affected_asset="BTC",
                event_name=item["name"], actual=item["time"], official_url=source,
                schedule_url=BITCOIN_TAPROOT_URL if item.get("schedule") else "",
                schedule=item.get("schedule", ""), block=item["block"],
                anticipated_status=item["status"],
                provenance=f"https://blockstream.info/api/block/{item['hash']}",
                timestamp_provenance=(
                    "canonical_chain_derived_secondary_not_official_publication_time"
                ),
            )
        )
    for item in ETH_CLOCKS:
        for component in item["components"]:
            epoch_anchored = bool(item.get("epoch"))
            actual_source = (
                item["schedule_url"]
                if component == "BPO1"
                else (
                    ETHEREUM_HISTORY_URL
                    if epoch_anchored
                    else f"https://etherscan.io/block/{item['block']}"
                )
            )
            rows.append(
                _protocol_row(
                    family="ethereum_protocol", subfamily="mainnet_network_upgrade",
                    member_id=f"{item['clock']}_{component.casefold().replace(' ', '_')}",
                    clock_id=item["clock"], process_bundle_id=item["bundle"],
                    affected_asset="ETH", event_name=component, actual=item["time"],
                    official_url=(
                        item["schedule_url"]
                        if component == "BPO1"
                        else ETHEREUM_HISTORY_URL
                    ),
                    schedule_url=item["schedule_url"], schedule=item["schedule"],
                    final=item.get("final", ""), block=item.get("block", ""),
                    epoch=item.get("epoch", ""), ttd=item.get("ttd", ""),
                    anticipated_status=item["status"],
                    provenance=actual_source,
                    timestamp_provenance=(
                        "official_epoch_anchor_plus_deterministic_genesis_384_second_"
                        "derivation_not_official_publication_time"
                        if epoch_anchored
                        else "canonical_chain_explorer_secondary_not_official_"
                        "publication_time"
                    ),
                    actual_time_derivation=(
                        f"{ETHEREUM_GENESIS_TIME} + {item['epoch']} * "
                        f"{ETHEREUM_EPOCH_SECONDS} seconds"
                        if epoch_anchored
                        else ""
                    ),
                )
            )
    for order in SEC_ORDERS:
        url = _sec_pdf(
            order["exchange"], int(order["date"][:4]), order["release"],
            order.get("suffix", ""),
        )
        for file_number, product in order["members"]:
            rows.append(
                {
                    "member_id": f"{order['release']}:{file_number}",
                    "family": "sec_spot_etp_decision",
                    "subfamily": "spot_btc_eth_exchange_listing_order",
                    "activation_clock_id": "",
                    "date_episode_id": f"sec_{order['date']}",
                    "process_bundle_id": sec_process_bundle(file_number, product),
                    "affected_asset": order.get("asset", "BTC"),
                    "event_name": product,
                    "order_document_id": order["release"],
                    "decision_action": order["action"],
                    "official_source_url": url,
                    "schedule_source_url": "",
                    "actual_time_source_url": url,
                    "actual_time_or_date": order["date"],
                    "actual_time_precision": "date_only_not_first_public_minute",
                    "activation_block": "",
                    "activation_epoch": "",
                    "activation_ttd": "",
                    "first_official_schedule_date_or_time": "",
                    "first_schedule_precision": "unavailable",
                    "final_schedule_knowable_date_or_time": "",
                    "final_schedule_precision": "unavailable",
                    "anticipated_or_conditional_status": "decision_date_not_exact_publication_clock",
                    "reschedule_revision_lineage": (
                        "Underlying docket deadlines and procedural revisions require a "
                        "separate source audit before anticipation testing."
                    ),
                    "expectation_status": "unavailable",
                    "exact_minute_eligible": False,
                    "immediate_1h_4h_eligible": False,
                    "anticipation_eligibility": "only_if_prior_official_deadline_is_separately_frozen",
                    "statistical_inference_eligible": False,
                    "statistical_status": "date_scale_source_ready_minute_scale_blocked",
                    "timestamp_provenance": "sec_issue_date_not_first_public_minute",
                    "actual_time_derivation": "",
                }
            )
    catalogue = DataFrame(rows).sort_values(
        ["actual_time_or_date", "family", "activation_clock_id", "member_id"],
        kind="stable",
    ).reset_index(drop=True)
    for column in (
        "official_source_sha256",
        "official_source_retrieved_at_utc",
        "schedule_source_sha256",
        "schedule_source_retrieved_at_utc",
        "actual_time_source_sha256",
        "actual_time_source_retrieved_at_utc",
    ):
        catalogue[column] = ""
    validate_catalogue(catalogue)
    return catalogue


def validate_catalogue(frame: DataFrame) -> None:
    if len(frame) != 63 or frame["member_id"].duplicated().any():
        raise ValueError("Structural member surface must contain 63 unique rows")
    expected_members = {"bitcoin_protocol": 3, "ethereum_protocol": 18, "sec_spot_etp_decision": 42}
    if frame.groupby("family").size().to_dict() != expected_members:
        raise ValueError("Structural member family counts changed")
    episode_counts = {
        "bitcoin_protocol": 3,
        "ethereum_protocol": 14,
        "sec_spot_etp_decision": 19,
    }
    actual_episodes = {
        family: int(group["date_episode_id"].nunique())
        for family, group in frame.groupby("family", sort=True)
    }
    if actual_episodes != episode_counts:
        raise ValueError("Structural episode counts changed")
    eth = frame[frame["family"].eq("ethereum_protocol")]
    if eth["activation_clock_id"].nunique() != 14:
        raise ValueError("Ethereum activation-clock count changed")
    validate_ethereum_timestamp_evidence(eth)
    sec = frame[frame["family"].eq("sec_spot_etp_decision")]
    if sec["order_document_id"].nunique() != 22:
        raise ValueError("SEC final-order document count changed")
    if sec["exact_minute_eligible"].any() or sec["immediate_1h_4h_eligible"].any():
        raise ValueError("SEC date-only records cannot be minute-scale eligible")
    if not sec["actual_time_or_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("SEC dates must remain date-only and must not become midnight clocks")
    if not frame["expectation_status"].eq("unavailable").all():
        raise ValueError("No historical expectation source is frozen in this catalogue")
    forbidden = ("profit", "return", "price", "volume", "direction", "surprise_value")
    if any(token in column.casefold() for column in frame.columns for token in forbidden):
        raise ValueError("Outcome field found in source-only catalogue")
    for url in pd.concat(
        [frame["official_source_url"], frame["schedule_source_url"], frame["actual_time_source_url"]]
    ):
        if url and not approved_source_url(str(url)):
            raise ValueError(f"Unapproved source URL: {url}")


def validate_ethereum_timestamp_evidence(eth: DataFrame) -> None:
    epoch_rows = eth[eth["activation_epoch"].ne("")]
    for row in epoch_rows.itertuples(index=False):
        expected_time = ethereum_epoch_time(int(row.activation_epoch))
        if row.actual_time_or_date != expected_time:
            raise ValueError(
                f"Ethereum epoch-derived time changed for {row.member_id}: "
                f"{row.actual_time_or_date} != {expected_time}"
            )
        expected_derivation = (
            f"{ETHEREUM_GENESIS_TIME} + {row.activation_epoch} * "
            f"{ETHEREUM_EPOCH_SECONDS} seconds"
        )
        if row.actual_time_derivation != expected_derivation:
            raise ValueError(f"Ethereum epoch derivation missing for {row.member_id}")
        if not row.timestamp_provenance.startswith("official_epoch_anchor_plus_"):
            raise ValueError(f"Ethereum epoch provenance changed for {row.member_id}")
    block_rows = eth[eth["activation_epoch"].eq("")]
    for row in block_rows.itertuples(index=False):
        expected_url = f"https://etherscan.io/block/{row.activation_block}"
        if row.actual_time_source_url != expected_url:
            raise ValueError(f"Ethereum block timestamp source changed for {row.member_id}")
        if row.timestamp_provenance != (
            "canonical_chain_explorer_secondary_not_official_publication_time"
        ):
            raise ValueError(f"Ethereum block timestamp provenance changed for {row.member_id}")


def ethereum_epoch_time(epoch: int) -> str:
    if epoch < 0:
        raise ValueError("Ethereum epoch cannot be negative")
    value = pd.Timestamp(ETHEREUM_GENESIS_TIME) + pd.Timedelta(
        seconds=epoch * ETHEREUM_EPOCH_SECONDS
    )
    return value.isoformat().replace("+00:00", "Z")


def count_catalogue(frame: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for family, group in frame.groupby("family", sort=True):
        for year in range(2020, 2026):
            yearly = group[group["actual_time_or_date"].str[:4].eq(str(year))]
            records.append(
                {
                    "family": family,
                    "period": str(year),
                    "member_count": len(yearly),
                    "activation_clock_count": int(yearly["activation_clock_id"].replace("", pd.NA).nunique()),
                    "date_episode_count": int(yearly["date_episode_id"].nunique()),
                    "order_document_count": int(yearly["order_document_id"].replace("", pd.NA).nunique()),
                    "process_bundle_count": int(yearly["process_bundle_id"].nunique()),
                }
            )
        records.append(
            {
                "family": family,
                "period": "total",
                "member_count": len(group),
                "activation_clock_count": int(group["activation_clock_id"].replace("", pd.NA).nunique()),
                "date_episode_count": int(group["date_episode_id"].nunique()),
                "order_document_count": int(group["order_document_id"].replace("", pd.NA).nunique()),
                "process_bundle_count": int(group["process_bundle_id"].nunique()),
            }
        )
    return DataFrame(records)


def approved_source_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() == "https"
        and (parsed.hostname or "").casefold() in APPROVED_HOSTS
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.fragment
    )


def _normalise_text(value: str) -> str:
    return " ".join(value.replace("\u2013", "-").replace("\u2014", "-").split()).casefold()


def _pdf_text(content: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)


def _ethereum_history_tokens() -> tuple[str, ...]:
    tokens: set[str] = set()
    for item in ETH_CLOCKS:
        if "BPO1" not in item["components"]:
            tokens.update(item["components"])
        for field in ("block", "epoch"):
            if item.get(field) and "BPO1" not in item["components"]:
                tokens.add(f"{int(item[field]):,}")
        if item.get("ttd"):
            tokens.add(item["ttd"])
    return tuple(sorted(tokens))


def _ethereum_schedule_specs() -> list[SourceSpec]:
    by_url: dict[str, set[str]] = {}
    for item in ETH_CLOCKS:
        tokens = {
            token
            for component in item["components"]
            for token in (
                ("Beacon Chain", "genesis")
                if component == "Beacon Chain genesis"
                else (component,)
            )
        }
        by_url.setdefault(item["schedule_url"], set()).update(tokens)
        if "BPO1" in item["components"]:
            by_url[item["schedule_url"]].add(item["epoch"])
        if item.get("final_url"):
            by_url.setdefault(item["final_url"], set()).add("Beacon Chain")
    return [
        SourceSpec(url, "official_project_publication", "html", tuple(sorted(tokens)))
        for url, tokens in sorted(by_url.items())
    ]


def _chain_timestamp_specs() -> list[SourceSpec]:
    specs = [
        SourceSpec(
            f"https://blockstream.info/api/block/{item['hash']}",
            "canonical_chain_derived_secondary",
            "json",
            (item["hash"], item["block"], str(int(pd.Timestamp(item["time"]).timestamp()))),
        )
        for item in BTC_CLOCKS
    ]
    for item in ETH_CLOCKS:
        if item.get("epoch"):
            continue
        timestamp = pd.Timestamp(item["time"]).strftime("%b-%d-%Y %I:%M:%S %p +UTC")
        specs.append(
            SourceSpec(
                f"https://etherscan.io/block/{item['block']}",
                "canonical_chain_explorer_secondary",
                "html",
                (item["block"], timestamp),
            )
        )
    return specs


def _sec_order_specs() -> list[SourceSpec]:
    specs = []
    for order in SEC_ORDERS:
        issue_date = pd.Timestamp(order["date"]).strftime("%B %d, %Y").replace(" 0", " ")
        tokens = (order["release"], issue_date, *(member[0] for member in order["members"]))
        specs.append(
            SourceSpec(
                _sec_pdf(order["exchange"], int(order["date"][:4]), order["release"], order.get("suffix", "")),
                "official_regulator_order",
                "pdf",
                tuple(tokens),
            )
        )
    return specs


def source_specs() -> tuple[SourceSpec, ...]:
    specs: list[SourceSpec] = [
        SourceSpec(BITCOIN_RELEASES_URL, "official_project_publication", "html", ("Bitcoin Core :: Releases",)),
        SourceSpec(BITCOIN_CHAINPARAMS_URL, "official_project_repository", "text", ("nSubsidyHalvingInterval = 210000",)),
        SourceSpec(BITCOIN_TAPROOT_URL, "official_project_publication", "html", ("Taproot", "709632")),
        SourceSpec(
            ETHEREUM_HISTORY_URL,
            "official_project_publication",
            "html",
            _ethereum_history_tokens(),
        ),
    ]
    specs.extend(_ethereum_schedule_specs())
    specs.extend(_chain_timestamp_specs())
    specs.extend(_sec_order_specs())
    unique = {spec.url: spec for spec in specs}
    if len(unique) != len(specs):
        raise ValueError("Duplicate source specification URL")
    return tuple(unique[url] for url in sorted(unique))


def fetch_source(
    spec: SourceSpec,
    *,
    requester: Callable[..., Any] = requests.get,
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    if not approved_source_url(spec.url):
        raise ValueError(f"Unapproved frozen source URL: {spec.url}")
    response = requester(
        spec.url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        allow_redirects=False,
    )
    content = bytes(response.content)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Frozen source returned HTTP {response.status_code}: {spec.url}")
    if str(response.url) != spec.url:
        raise ValueError(f"Frozen source response URL changed: {response.url}")
    if spec.media_type == "pdf":
        if not content.startswith(b"%PDF"):
            raise ValueError(f"SEC order is not a PDF: {spec.url}")
        searchable = _pdf_text(content)
    elif spec.media_type == "json":
        document = json.loads(content.decode("utf-8"))
        searchable = json.dumps(document, sort_keys=True)
    else:
        searchable = content.decode("utf-8", errors="replace")
    normalised = _normalise_text(searchable)
    missing = [token for token in spec.expected_tokens if _normalise_text(token) not in normalised]
    if missing:
        raise ValueError(f"Frozen source verification tokens missing from {spec.url}: {missing}")
    return {
        "url": spec.url,
        "source_class": spec.source_class,
        "media_type": spec.media_type,
        "http_status": int(response.status_code),
        "content_bytes": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "retrieved_at_utc": g0.utc_now(),
        "verified_tokens": list(spec.expected_tokens),
    }


def retrieve_sources() -> list[dict[str, Any]]:
    specs = source_specs()
    with ThreadPoolExecutor(max_workers=MAX_RETRIEVAL_WORKERS) as executor:
        rows = list(executor.map(fetch_source, specs))
    return sorted(rows, key=lambda row: row["url"])


def bind_source_evidence(
    catalogue: DataFrame, sources: Sequence[Mapping[str, Any]]
) -> DataFrame:
    evidence = {str(source["url"]): source for source in sources}
    result = catalogue.copy()
    for prefix, url_column in (
        ("official_source", "official_source_url"),
        ("schedule_source", "schedule_source_url"),
        ("actual_time_source", "actual_time_source_url"),
    ):
        for index, url in result[url_column].items():
            if not url:
                continue
            source = evidence.get(str(url))
            if source is None:
                raise ValueError(f"Catalogue source is absent from manifest: {url}")
            result.at[index, f"{prefix}_sha256"] = source["content_sha256"]
            result.at[index, f"{prefix}_retrieved_at_utc"] = source["retrieved_at_utc"]
    required = result["official_source_url"].ne("")
    if not result.loc[required, "official_source_sha256"].ne("").all():
        raise ValueError("Not every catalogue row is bound to verified official source evidence")
    return result


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def atomic_write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def freeze_document(catalogue: DataFrame, counts: DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_outcome_blind_crypto_structural_source_catalogue",
        "period": {"start": START_DATE, "end": END_DATE},
        "market_outcomes_opened": False,
        "profit_used": False,
        "direction_tested": False,
        "inferred_surprise_used": False,
        "counts": counts.to_dict(orient="records"),
        "assertions": {
            "member_rows": len(catalogue),
            "bitcoin_activation_clocks": 3,
            "ethereum_named_components": 18,
            "ethereum_activation_clocks": 14,
            "sec_final_order_documents": 22,
            "sec_underlying_proposal_members": 42,
            "sec_date_episodes": 19,
            "ethereum_partition_clocks_2020_2022": 9,
            "ethereum_partition_clocks_2023_2025": 5,
            "sec_partition_episodes_2020_2022": 11,
            "sec_partition_episodes_2023_2025": 8,
        },
        "boundaries": [
            "Bitcoin is case-study/prospective because three clocks cannot support inference.",
            "Ethereum is small, heterogeneous, and programme-linked; its clocks are not independent evidence or model-ready.",
            "SEC is date-scale source-ready but minute-scale blocked; issue dates are not first-public minutes.",
            "Related stages and repeated regulatory processes remain linked by process_bundle_id.",
            "Actual Bitcoin block times are canonical-chain-derived secondary observations, not official publication times.",
            "Pre-merge Ethereum block times use Etherscan as a labelled secondary chain explorer.",
            "Ethereum epoch-clock seconds are deterministically derived from the official epoch anchor and genesis cadence, not treated as publication timestamps.",
            "Eventual activation timestamps must never be backfilled into pre-event knowledge.",
            "No market outcome, price/volume, direction, profit, or inferred surprise was used.",
        ],
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def report_text() -> str:
    return "\n".join(
        [
            "# Crypto Structural Event Source Catalogue",
            "",
            "- Bitcoin: `3` activation clocks; case-study/prospective only.",
            "- Ethereum: `18` named components at `14` linked clocks; not model-ready.",
            "- SEC: `22` final orders, `42` proposal members, `19` date episodes.",
            "- Ethereum clock evidence: official anchors plus explicit epoch derivation or labelled Etherscan block timestamps.",
            "- SEC exact-minute eligibility: **No**; issue date is not first-public minute.",
            "- Market outcomes opened: **No**",
            "- Direction, profit, or inferred surprise used: **No**",
            "",
        ]
    )


def _expected_artifact_keys() -> set[str]:
    return {"catalogue", "counts", "source_manifest", "source_freeze", "report", "analysis_script"}


def _expected_artifact_paths() -> dict[str, Path]:
    return {
        "catalogue": EXPECTED_OUTPUTS[0],
        "counts": EXPECTED_OUTPUTS[1],
        "source_manifest": EXPECTED_OUTPUTS[2],
        "source_freeze": EXPECTED_OUTPUTS[3],
        "report": EXPECTED_OUTPUTS[4],
        "analysis_script": ANALYSIS_PATH,
    }


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_crypto_structural_source_freeze":
        raise ValueError("Existing structural catalogue result is invalid")
    if result.get("member_rows") != 63 or result.get("source_documents") != 48:
        raise ValueError("Existing structural result counts changed")
    for flag in ("market_outcomes_opened", "profit_used", "direction_tested", "inferred_surprise_used"):
        if result.get(flag) is not False:
            raise ValueError(f"Existing source-only gate changed: {flag}")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _expected_artifact_keys():
        raise ValueError("Existing structural artifact key set changed")
    expected_paths = _expected_artifact_paths()
    for key, value in artifacts.items():
        path = Path(value["path"])
        if path.resolve() != expected_paths[key].resolve():
            raise ValueError(f"Existing structural artifact path changed: {key}")
        if not path.is_file() or g0.sha256_file(path) != value["sha256"]:
            raise ValueError(f"Existing structural artifact changed: {path}")
    existing = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
    expected = {path.name for path in EXPECTED_OUTPUTS}
    if existing != expected:
        raise ValueError("Structural output folder does not contain the exact six artifacts")


def _check_output_state(*, overwrite: bool) -> dict[str, Any] | None:
    existing = (
        [path for path in OUTPUT_ROOT.iterdir() if path.is_file()]
        if OUTPUT_ROOT.is_dir()
        else []
    )
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    if existing and not overwrite:
        raise FileExistsError(f"Partial structural output exists before retrieval: {existing}")
    return None


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    existing = _check_output_state(overwrite=overwrite)
    if existing is not None:
        return existing
    catalogue = build_catalogue()
    sources = retrieve_sources()
    catalogue = bind_source_evidence(catalogue, sources)
    counts = count_catalogue(catalogue)
    source_counts = Counter(row["source_class"] for row in sources)
    manifest = {
        "schema_version": 1,
        "retrieved_at_utc": g0.utc_now(),
        "max_retrieval_workers": MAX_RETRIEVAL_WORKERS,
        "source_counts": dict(sorted(source_counts.items())),
        "sources": sources,
    }
    freeze = freeze_document(catalogue, counts)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(catalogue, CATALOGUE_PATH)
    g0.atomic_write_csv(counts, COUNTS_PATH)
    g0.atomic_write_json(manifest, SOURCE_MANIFEST_PATH)
    freeze["source_manifest"] = artifact(SOURCE_MANIFEST_PATH)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    atomic_write_text(report_text(), REPORT_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_crypto_structural_source_freeze",
        "created_at_utc": g0.utc_now(),
        "market_outcomes_opened": False,
        "profit_used": False,
        "direction_tested": False,
        "inferred_surprise_used": False,
        "member_rows": len(catalogue),
        "source_documents": len(sources),
        "artifacts": {
            "catalogue": artifact(CATALOGUE_PATH),
            "counts": artifact(COUNTS_PATH),
            "source_manifest": artifact(SOURCE_MANIFEST_PATH),
            "source_freeze": artifact(FREEZE_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    validate_existing_result(result)
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
                    "market_outcomes_will_be_opened": False,
                    "profit_will_be_used": False,
                    "direction_will_be_tested": False,
                    "inferred_surprise_will_be_used": False,
                    "frozen_member_rows": 63,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())
