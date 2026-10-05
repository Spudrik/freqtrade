# ruff: noqa: S101

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_regional_nominal_clock_freeze as freeze,
)


WEEKDAYS = "一二三四五六日"


def _date_cell(year: int, month: int, day: int = 1) -> str:
    weekday = WEEKDAYS[date(year, month, day).weekday()]
    return f"{day}/{weekday}"


def _calendar_html(year: int, *, ppi_minute: str = "30") -> str:
    months = [f"{month}月" for month in range(1, 13)]
    rows: list[list[object]] = [["序号", "内容", *months]]

    def add_pair(label: str, *, minute: str = "30", skip_february: bool = False) -> None:
        dates = [
            "" if skip_february and month == 2 else _date_cell(year, month)
            for month in range(1, 13)
        ]
        times = ["" if not value else f"09:{minute}" for value in dates]
        rows.extend([[len(rows), label, *dates], [len(rows) + 1, label, *times]])

    add_pair("国民经济运行情况新闻发布会", skip_february=True)
    add_pair("采购经理指数月度报告")
    add_pair("居民消费价格指数月度报告")
    add_pair("工业生产者价格指数月度报告", minute=ppi_minute)
    rows.append(
        [
            99,
            "注: 国民经济运行情况和采购经理指数月度报告的日期可能调整。",
            *("" for _ in range(12)),
        ]
    )
    return pd.DataFrame(rows).to_html(index=False, header=False)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_nbs_calendar_keeps_three_whole_event_families() -> None:
    result = freeze.parse_nbs_calendar(
        _calendar_html(2021),
        year=2021,
        source_url="https://www.stats.gov.cn/xxgk/sjfb/fbrcb/example.html",
    )

    assert len(result) == 35
    assert result["event_family"].value_counts().to_dict() == {
        "china_nbs_price_release": 12,
        "china_nbs_pmi_release": 12,
        "china_nbs_national_economy_release": 11,
    }
    price_rows = result[result["event_family"].eq("china_nbs_price_release")]
    assert price_rows["companion_clock_verified"].all()
    assert result["strict_intraday_attribution_eligible"].eq(False).all()
    assert result["anchor_utc"].str.endswith("+00:00").all()


def test_parse_nbs_calendar_rejects_different_cpi_ppi_clocks() -> None:
    with pytest.raises(ValueError, match="companion clock differs"):
        freeze.parse_nbs_calendar(
            _calendar_html(2022, ppi_minute="31"),
            year=2022,
            source_url="https://www.stats.gov.cn/xxgk/sjfb/fbrcb/example.html",
        )


def test_cell_events_applies_one_printed_clock_to_two_dates_in_one_month() -> None:
    assert freeze._cell_events(
        "1/二注5 31/四", "9:30", context="2022 PMI March"
    ) == [(1, "二", 9, 30), (31, "四", 9, 30)]


def test_parse_eurostat_rows_counts_only_exact_major_titles() -> None:
    content = json.dumps(
        [
            {"title": "Flash estimate inflation euro area"},
            {"title": "Unemployment"},
            {"title": "Unemployment - duplicate-looking but different"},
        ]
    ).encode()

    rows, counts = freeze.parse_eurostat_rows(content, year=2025)

    assert len(rows) == 3
    assert counts == {
        "eurostat_flash_inflation": 1,
        "eurostat_preliminary_gdp": 0,
        "eurostat_unemployment": 1,
    }


def test_execute_refuses_partial_outputs_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "CATALOGUE_PATH": tmp_path / "catalogue.csv",
        "COUNTS_PATH": tmp_path / "counts.csv",
        "SOURCE_MANIFEST_PATH": tmp_path / "manifest.json",
        "FREEZE_PATH": tmp_path / "freeze.json",
        "REPORT_PATH": tmp_path / "report.md",
        "RESULT_PATH": tmp_path / "result.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(freeze, name, path)
    paths["CATALOGUE_PATH"].write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Partial regional source-freeze outputs"):
        freeze.execute()


def test_validate_existing_result_requires_and_checks_every_artifact(tmp_path: Path) -> None:
    artifact_names = {
        "catalogue",
        "counts",
        "source_manifest",
        "freeze",
        "report",
        "analysis_script",
    }
    artifacts = {}
    for name in artifact_names:
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        artifacts[name] = {"path": str(path), "sha256": _hash(path)}
    result = {
        "status": "completed_regional_nominal_clock_source_freeze",
        "artifacts": artifacts,
    }

    freeze.validate_existing_result(result)
    Path(artifacts["counts"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        freeze.validate_existing_result(result)
