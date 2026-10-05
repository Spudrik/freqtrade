# The long assertions keep each frozen count leg visible in one statement.
# ruff: noqa: E501, S101

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_crypto_structural_catalogue"
)
cs = importlib.import_module(MODULE)


def test_frozen_member_and_episode_counts() -> None:
    frame = cs.build_catalogue()
    assert len(frame) == 63
    assert frame.groupby("family").size().to_dict() == {
        "bitcoin_protocol": 3,
        "ethereum_protocol": 18,
        "sec_spot_etp_decision": 42,
    }
    assert frame.groupby("family")["date_episode_id"].nunique().to_dict() == {
        "bitcoin_protocol": 3,
        "ethereum_protocol": 14,
        "sec_spot_etp_decision": 19,
    }
    assert frame.loc[frame["family"].eq("ethereum_protocol"), "activation_clock_id"].nunique() == 14
    assert frame.loc[frame["family"].eq("sec_spot_etp_decision"), "order_document_id"].nunique() == 22


def test_partitions_and_simultaneous_members_are_frozen() -> None:
    frame = cs.build_catalogue()
    eth = frame[frame["family"].eq("ethereum_protocol")]
    sec = frame[frame["family"].eq("sec_spot_etp_decision")]
    assert eth.loc[eth["actual_time_or_date"].str[:4].astype(int).le(2022), "activation_clock_id"].nunique() == 9
    assert eth.loc[eth["actual_time_or_date"].str[:4].astype(int).ge(2023), "activation_clock_id"].nunique() == 5
    assert sec.loc[sec["actual_time_or_date"].str[:4].astype(int).le(2022), "date_episode_id"].nunique() == 11
    assert sec.loc[sec["actual_time_or_date"].str[:4].astype(int).ge(2023), "date_episode_id"].nunique() == 8
    assert eth.loc[eth["activation_clock_id"].eq("eth_2023_shapella"), "event_name"].tolist() == ["Capella", "Shanghai"]
    assert sec.loc[sec["order_document_id"].eq("34-99306"), "date_episode_id"].nunique() == 1
    assert len(sec.loc[sec["order_document_id"].eq("34-99306")]) == 11


def test_programme_and_regulatory_process_links_remain_explicit() -> None:
    frame = cs.build_catalogue()
    eth = frame[frame["family"].eq("ethereum_protocol")]
    merge = eth[eth["event_name"].isin(["Bellatrix", "Paris"])]
    assert merge["activation_clock_id"].nunique() == 2
    assert merge["process_bundle_id"].nunique() == 1
    sec = frame[frame["family"].eq("sec_spot_etp_decision")]
    grayscale = sec[sec["event_name"].eq("Grayscale Bitcoin Trust")]
    assert set(grayscale["decision_action"]) == {"approved", "denied"}
    assert grayscale["process_bundle_id"].nunique() == 1


def test_source_only_and_timestamp_semantics() -> None:
    frame = cs.build_catalogue()
    sec = frame[frame["family"].eq("sec_spot_etp_decision")]
    assert not sec["exact_minute_eligible"].any()
    assert not sec["immediate_1h_4h_eligible"].any()
    assert sec["actual_time_or_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert sec["timestamp_provenance"].eq("sec_issue_date_not_first_public_minute").all()
    assert frame["expectation_status"].eq("unavailable").all()
    forbidden = ("profit", "return", "price", "volume", "direction", "surprise_value")
    assert not any(token in column.casefold() for column in frame.columns for token in forbidden)


def test_protocol_timing_boundaries_do_not_backfill_future_knowledge() -> None:
    frame = cs.build_catalogue()
    btc = frame[frame["family"].eq("bitcoin_protocol")]
    assert btc["timestamp_provenance"].eq(
        "canonical_chain_derived_secondary_not_official_publication_time"
    ).all()
    assert btc.loc[btc["event_name"].str.contains("halving"), "first_official_schedule_date_or_time"].eq("").all()
    variable = frame[
        frame["anticipated_or_conditional_status"].str.contains("variable|terminal")
    ]
    assert variable["anticipation_eligibility"].eq(
        "requires_contemporaneous_arrival_or_condition_estimate"
    ).all()
    taproot = btc[btc["event_name"].eq("Taproot")].iloc[0]
    assert taproot["final_schedule_knowable_date_or_time"] == ""
    assert taproot["final_schedule_precision"] == "unavailable"
    assert taproot["anticipation_eligibility"] == (
        "requires_documented_lock_in_plus_contemporaneous_block_arrival_estimate"
    )


def test_ethereum_epoch_times_and_provenance_are_derived() -> None:
    frame = cs.build_catalogue()
    eth = frame[frame["family"].eq("ethereum_protocol")]
    epoch = eth[eth["activation_epoch"].ne("")]
    assert len(epoch) == 12
    assert epoch["timestamp_provenance"].str.startswith(
        "official_epoch_anchor_plus_"
    ).all()
    assert epoch["actual_time_derivation"].str.contains(
        cs.ETHEREUM_GENESIS_TIME, regex=False
    ).all()
    assert cs.ethereum_epoch_time(194048) == "2023-04-12T22:27:35Z"
    damaged = frame.copy()
    damaged.loc[
        damaged["activation_clock_id"].eq("eth_2021_altair"),
        "actual_time_or_date",
    ] = "2021-10-27T10:56:24Z"
    with pytest.raises(ValueError, match="epoch-derived time changed"):
        cs.validate_catalogue(damaged)


def test_ethereum_block_timestamp_and_paris_regression() -> None:
    frame = cs.build_catalogue()
    paris = frame[frame["event_name"].eq("Paris")].iloc[0]
    assert paris["actual_time_or_date"] == "2022-09-15T06:42:59Z"
    assert paris["actual_time_source_url"] == "https://etherscan.io/block/15537394"
    assert paris["timestamp_provenance"] == (
        "canonical_chain_explorer_secondary_not_official_publication_time"
    )

    spec = next(
        item
        for item in cs.source_specs()
        if item.url == "https://etherscan.io/block/15537394"
    )

    class Response:
        status_code = 200
        url = spec.url
        content = b"Block 15537394 Sep-15-2022 06:42:42 AM +UTC"

    with pytest.raises(ValueError, match="tokens missing"):
        cs.fetch_source(spec, requester=lambda *args, **kwargs: Response())


def test_counts_table_matches_year_surface() -> None:
    counts = cs.count_catalogue(cs.build_catalogue())
    sec = counts[counts["family"].eq("sec_spot_etp_decision")].set_index("period")
    assert sec.loc["2024", "member_count"] == 25
    assert sec.loc["2024", "order_document_count"] == 5
    assert sec.loc["total", "member_count"] == 42
    eth = counts[counts["family"].eq("ethereum_protocol")].set_index("period")
    assert eth.loc["2025", "member_count"] == 5
    assert eth.loc["2025", "activation_clock_count"] == 3


def test_source_specs_are_bounded_official_or_labelled_secondary() -> None:
    specs = cs.source_specs()
    assert len(specs) == 48
    assert len({spec.url for spec in specs}) == len(specs)
    assert all(cs.approved_source_url(spec.url) for spec in specs)
    secondary = [spec for spec in specs if spec.source_class == "canonical_chain_derived_secondary"]
    assert len(secondary) == 3
    assert all("blockstream.info/api/block/" in spec.url for spec in secondary)
    chain_explorer = [
        spec
        for spec in specs
        if spec.source_class == "canonical_chain_explorer_secondary"
    ]
    assert len(chain_explorer) == 6
    assert all("etherscan.io/block/" in spec.url for spec in chain_explorer)
    assert cs.MAX_RETRIEVAL_WORKERS == 4


def test_fetch_rejects_changed_url_and_missing_verification_token() -> None:
    class Response:
        status_code = 200
        url = "https://example.com/changed"
        content = b"expected"

    spec = cs.SourceSpec(
        "https://bitcoincore.org/test", "official_project_publication", "text", ("expected",)
    )

    def requester(*args: Any, **kwargs: Any) -> Response:
        assert kwargs["allow_redirects"] is False
        return Response()

    with pytest.raises(ValueError, match="response URL changed"):
        cs.fetch_source(spec, requester=requester)
    Response.url = spec.url
    Response.content = b"other"
    with pytest.raises(ValueError, match="tokens missing"):
        cs.fetch_source(spec, requester=requester)


def test_source_hashes_and_retrieval_times_bind_to_every_row() -> None:
    frame = cs.build_catalogue()
    urls = set(frame["official_source_url"]) | set(frame["schedule_source_url"]) | set(
        frame["actual_time_source_url"]
    )
    sources = [
        {
            "url": url,
            "content_sha256": "a" * 64,
            "retrieved_at_utc": "2026-09-13T12:00:00Z",
        }
        for url in urls
        if url
    ]
    bound = cs.bind_source_evidence(frame, sources)
    assert bound["official_source_sha256"].eq("a" * 64).all()
    assert bound["official_source_retrieved_at_utc"].eq(
        "2026-09-13T12:00:00Z"
    ).all()


def test_partial_output_refuses_before_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = tuple(tmp_path / name for name in ["catalogue.csv", "counts.csv", "manifest.json", "freeze.json", "report.md", "result.json"])
    monkeypatch.setattr(cs, "EXPECTED_OUTPUTS", paths)
    monkeypatch.setattr(cs, "RESULT_PATH", paths[-1])
    monkeypatch.setattr(cs, "OUTPUT_ROOT", tmp_path)
    paths[0].write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="before retrieval"):
        cs._check_output_state(overwrite=False)


def test_validate_existing_result_rejects_source_only_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cs, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(cs, "EXPECTED_OUTPUTS", tuple(tmp_path / name for name in ["a", "b", "c", "d", "e", "f"]))
    result = {
        "status": "completed_crypto_structural_source_freeze",
        "member_rows": 63,
        "source_documents": 48,
        "market_outcomes_opened": True,
        "profit_used": False,
        "direction_tested": False,
        "inferred_surprise_used": False,
        "artifacts": {},
    }
    with pytest.raises(ValueError, match="market_outcomes_opened"):
        cs.validate_existing_result(result)


def test_validate_existing_result_rejects_artifact_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    paths = tuple(
        output / name
        for name in ["catalogue.csv", "counts.csv", "manifest.json", "freeze.json", "report.md", "result.json"]
    )
    for path in paths:
        path.write_text("same", encoding="utf-8")
    analysis_path = tmp_path / "analysis.py"
    analysis_path.write_text("same", encoding="utf-8")
    monkeypatch.setattr(cs, "OUTPUT_ROOT", output)
    monkeypatch.setattr(cs, "EXPECTED_OUTPUTS", paths)
    monkeypatch.setattr(cs, "ANALYSIS_PATH", analysis_path)
    artifacts = {
        key: {"path": str(path), "sha256": cs.g0.sha256_file(path)}
        for key, path in cs._expected_artifact_paths().items()
    }
    substitute = tmp_path / "substitute.csv"
    substitute.write_text("same", encoding="utf-8")
    artifacts["catalogue"]["path"] = str(substitute)
    result = {
        "status": "completed_crypto_structural_source_freeze",
        "member_rows": 63,
        "source_documents": 48,
        "market_outcomes_opened": False,
        "profit_used": False,
        "direction_tested": False,
        "inferred_surprise_used": False,
        "artifacts": artifacts,
    }
    with pytest.raises(ValueError, match="artifact path changed"):
        cs.validate_existing_result(result)


def test_ready_mode_is_explicitly_outcome_blind(capsys: pytest.CaptureFixture[str]) -> None:
    assert cs.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["market_outcomes_will_be_opened"] is False
    assert result["direction_will_be_tested"] is False
