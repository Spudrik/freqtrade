# ruff: noqa: S101

"""Outcome-blind tests for the official SEC hyperscaler 8-K catalogue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_sec_hyperscaler_8k_catalogue as sec,
)


def filing_document(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {column: [row[column] for row in rows] for column in sec.SUBMISSION_COLUMNS}


def filing_row(
    cik: str,
    serial: int,
    *,
    form: str = "8-K",
    accepted: str = "2021-02-01T21:02:03.000Z",
    items: str = "2.02,9.01",
) -> dict[str, str]:
    return {
        "accessionNumber": f"{cik}-{accepted[2:4]}-{serial:06d}",
        "filingDate": accepted[:10],
        "reportDate": "2020-12-31",
        "acceptanceDateTime": accepted,
        "form": form,
        "primaryDocument": f"form8k{serial}.htm",
        "items": items,
    }


def main_document(cik: str) -> dict[str, Any]:
    company = sec.COMPANIES[cik]
    recent = filing_document(
        [
            filing_row(cik, 1),
            filing_row(cik, 2, form="10-Q", items=""),
        ]
    )
    return {
        "cik": cik,
        "name": company["company_name"],
        "filings": {
            "recent": recent,
            "files": [
                {
                    "name": f"CIK{cik}-submissions-001.json",
                    "filingFrom": "2019-01-01",
                    "filingTo": "2020-01-15",
                },
                {
                    "name": f"CIK{cik}-submissions-002.json",
                    "filingFrom": "2010-01-01",
                    "filingTo": "2019-12-31",
                },
            ],
        },
    }


def supplement_document(cik: str) -> dict[str, list[str]]:
    return filing_document(
        [
            filing_row(
                cik,
                3,
                form="8-K/A",
                accepted="2020-01-02T13:14:15.000Z",
                items="2.02",
            )
        ]
    )


def test_fixed_identity_and_overlapping_supplement_selection() -> None:
    cik = "0000789019"
    main = main_document(cik)
    sec.validate_main_identity(main, cik)
    selected = sec.supplement_references(main, cik)
    assert [row["name"] for row in selected] == [
        "CIK0000789019-submissions-001.json"
    ]
    assert selected[0]["url"] == (
        "https://data.sec.gov/submissions/CIK0000789019-submissions-001.json"
    )
    main["name"] = "Wrong company"
    with pytest.raises(ValueError, match="name disagrees"):
        sec.validate_main_identity(main, cik)


def test_catalogue_keeps_required_source_fields_and_deduplicates() -> None:
    records: list[dict[str, Any]] = []
    for cik in sec.COMPANIES:
        main_url = sec.main_submission_url(cik)
        rows = sec.submission_rows(
            main_document(cik)["filings"]["recent"],
            cik=cik,
            source_url=main_url,
            source_kind="main",
        )
        records.extend(rows)
        records.extend(
            sec.submission_rows(
                supplement_document(cik),
                cik=cik,
                source_url=sec.supplemental_submission_url(
                    cik, f"CIK{cik}-submissions-001.json"
                ),
                source_kind="supplement",
            )
        )
    duplicate = dict(records[0])
    duplicate["submission_source_url"] = (
        "https://data.sec.gov/submissions/CIK0000789019-submissions-001.json"
    )
    duplicate["source_kind"] = "supplement"
    records.append(duplicate)
    catalogue = sec.build_catalogue(records)
    assert len(catalogue) == 8
    assert set(catalogue["form"]) == {"8-K", "8-K/A"}
    assert catalogue["acceptance_datetime_utc"].str.endswith("Z").all()
    assert catalogue["source_url"].str.startswith("https://www.sec.gov/").all()
    assert catalogue["earnings_like_candidate"].all()
    assert set(catalogue["whole_event_partition"]) == {
        "context_only_2020",
        "development_2021_2023",
    }
    assert catalogue.duplicated(["cik", "accession_number"]).sum() == 0


def test_duplicate_accession_disagreement_is_rejected() -> None:
    cik = next(iter(sec.COMPANIES))
    records = sec.submission_rows(
        filing_document([filing_row(cik, 1)]),
        cik=cik,
        source_url=sec.main_submission_url(cik),
        source_kind="main",
    )
    disagreeing = dict(records[0])
    disagreeing["primaryDocument"] = "different8k.htm"

    with pytest.raises(ValueError, match="records disagree"):
        sec.build_catalogue([*records, disagreeing])


def test_invalid_item_and_timestamp_are_rejected() -> None:
    with pytest.raises(ValueError, match="item code"):
        sec.parse_item_codes("2.2")
    with pytest.raises(ValueError, match="exact UTC"):
        sec.exact_utc_timestamp("2020-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="exact UTC"):
        sec.exact_utc_timestamp("2020-01-01T00:00:00.000+01:00")


def test_execute_writes_complete_manifest_and_revalidates_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "sec_catalogue"
    monkeypatch.setattr(sec, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(sec, "CATALOGUE_PATH", output_root / "catalogue.csv")
    monkeypatch.setattr(sec, "COUNTS_PATH", output_root / "counts.csv")
    monkeypatch.setattr(sec, "SOURCE_MANIFEST_PATH", output_root / "manifest.json")
    monkeypatch.setattr(sec, "FREEZE_PATH", output_root / "freeze.json")
    monkeypatch.setattr(sec, "REPORT_PATH", output_root / "report.md")
    monkeypatch.setattr(sec, "RESULT_PATH", output_root / "result.json")
    calls: list[str] = []

    def fake_fetch(url: str) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(url)
        filename = url.rsplit("/", maxsplit=1)[-1]
        cik = filename[3:13]
        document = (
            supplement_document(cik)
            if "-submissions-" in filename
            else main_document(cik)
        )
        return document, {"url": url, "http_status": 200, "content_sha256": "a" * 64}

    monkeypatch.setattr(sec, "fetch_json", fake_fetch)
    result = sec.execute()
    assert result["event_rows"] == 8
    assert result["source_json_documents"] == 8
    manifest = json.loads(sec.SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["sources"]) == 8
    assert len(calls) == 8
    assert all(url.startswith("https://data.sec.gov/submissions/") for url in calls)
    assert sec.execute() == result


def test_official_urls_and_no_execute_is_offline(capsys: Any) -> None:
    filing_url = sec.filing_source_url(
        "0000789019", "0000789019-20-000001", "form8k.htm"
    )
    assert sec.official_submission_url(sec.main_submission_url("0000789019"))
    assert sec.official_submission_url(
        "https://data.sec.gov/submissions/CIK0000789019-submissions-001.json"
    )
    assert not sec.official_submission_url(
        "https://data.sec.gov/submissions/CIK0000789019.json?query=yes"
    )
    assert sec.official_filing_url(filing_url, "0000789019", "0000789019-20-000001")
    assert not sec.official_filing_url(
        filing_url.replace("https://", "http://"),
        "0000789019",
        "0000789019-20-000001",
    )
    assert sec.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False


def test_validate_catalogue_rejects_non_official_source_url() -> None:
    cik = next(iter(sec.COMPANIES))
    records = []
    for selected_cik in sec.COMPANIES:
        records.extend(
            sec.submission_rows(
                filing_document([filing_row(selected_cik, 1)]),
                cik=selected_cik,
                source_url=sec.main_submission_url(selected_cik),
                source_kind="main",
            )
        )
    catalogue = sec.build_catalogue(records)
    catalogue.loc[catalogue["cik"].eq(cik), "source_url"] = "https://example.com/x"
    with pytest.raises(ValueError, match="non-official filing URL"):
        sec.validate_catalogue(catalogue)


def test_fetch_rejects_non_official_final_response_url() -> None:
    class Response:
        content = b"{}"
        status_code = 200
        url = "https://example.com/redirected.json"

    def requester(*args: Any, **kwargs: Any) -> Response:
        assert kwargs["allow_redirects"] is False
        return Response()

    with pytest.raises(ValueError, match="approved official URL"):
        sec.fetch_json(sec.main_submission_url("0000789019"), requester=requester)
