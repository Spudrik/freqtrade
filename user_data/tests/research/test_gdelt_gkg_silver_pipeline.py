from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from user_data.Custom_Launcher.research.context_features import gdelt_gkg_silver_pipeline as pipeline


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def test_build_steps_plans_chunks_and_phases_without_execution(tmp_path):
    steps = pipeline.build_steps(
        python_exe=Path("python"),
        db_path=tmp_path / "gdelt_context.sqlite",
        feature_db=tmp_path / "context_features.sqlite",
        output_dir=tmp_path / "reports",
        start=_dt("2022-01-01T00:00:00+00:00"),
        end=_dt("2022-01-03T00:00:00+00:00"),
        chunk_days=1,
        phases=("inventory", "gkg_documents", "quality_gates"),
        tag_prefix="unit",
        validate_raw_zip=True,
        limit_files=2,
        limit_rows=5,
    )

    assert len(steps) == 6
    assert [step.phase for step in steps[:3]] == ["inventory", "gkg_documents", "quality_gates"]
    assert steps[0].start == "2022-01-01T00:00:00+00:00"
    assert steps[3].start == "2022-01-02T00:00:00+00:00"
    assert "--validate-zip" in steps[0].command
    assert "--limit-files" in steps[1].command
    assert "--limit-rows" in steps[1].command
    assert "--feature-db" in steps[2].command


def test_parse_phases_rejects_unknown_phase():
    try:
        pipeline.parse_phases("inventory,unknown")
    except SystemExit as exc:
        assert "Unknown phase" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_iter_chunks_is_end_exclusive_and_capped():
    chunks = list(
        pipeline.iter_chunks(
            _dt("2022-01-01T00:00:00+00:00"),
            _dt("2022-01-03T12:00:00+00:00"),
            1,
        )
    )

    assert chunks == [
        (_dt("2022-01-01T00:00:00+00:00"), _dt("2022-01-02T00:00:00+00:00")),
        (_dt("2022-01-02T00:00:00+00:00"), _dt("2022-01-03T00:00:00+00:00")),
        (_dt("2022-01-03T00:00:00+00:00"), _dt("2022-01-03T12:00:00+00:00")),
    ]
