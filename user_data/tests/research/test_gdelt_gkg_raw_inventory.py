import json
import sqlite3
import zipfile

from user_data.Custom_Launcher.research.context_features import gdelt_gkg_raw_inventory as inventory
from user_data.Custom_Launcher.research.context_features import gdelt_gkg_normalized_schema as schema


def _write_zip(path, member_name="member.csv", payload="one\ntwo\n"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member_name, payload)


def test_infer_file_identity_accepts_standard_archive_names():
    assert inventory.infer_file_identity("20220101000000.export.CSV.zip") == ("export", "20220101000000")
    assert inventory.infer_file_identity("20220101001500.gkg.csv.zip") == ("gkg", "20220101001500")
    assert inventory.infer_file_identity("20220101001500.translation.export.CSV.zip") is None
    assert inventory.infer_file_identity("not-a-gdelt.zip") is None


def test_dry_run_scans_both_dirs_with_filters_without_creating_db(tmp_path):
    export_dir = tmp_path / "export"
    gkg_dir = tmp_path / "gkg"
    export_dir.mkdir()
    gkg_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"

    _write_zip(export_dir / "20220101000000.export.CSV.zip")
    _write_zip(export_dir / "20220101010000.export.CSV.zip")
    _write_zip(gkg_dir / "20220101011500.gkg.csv.zip")
    _write_zip(gkg_dir / "20220101030000.gkg.csv.zip")
    _write_zip(gkg_dir / "ignore-me.zip")

    summary = inventory.inventory_raw_archives(
        db_path=db_path,
        export_raw_dir=export_dir,
        gkg_raw_dir=gkg_dir,
        start_stamp="20220101010000",
        end_stamp="20220101030000",
        dry_run=True,
    )

    assert summary.files_seen == 5
    assert summary.files_ignored == 1
    assert summary.files_planned == 2
    assert summary.files_upserted == 0
    assert summary.kind_counts == {"export": 1, "gkg": 1}
    assert [row["gdelt_stamp"] for row in summary.files] == ["20220101010000", "20220101011500"]
    assert not db_path.exists()


def test_validated_inventory_upserts_raw_file_metadata(tmp_path):
    export_dir = tmp_path / "export"
    gkg_dir = tmp_path / "gkg"
    export_dir.mkdir()
    gkg_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"

    _write_zip(export_dir / "20220101000000.export.CSV.zip", payload="abc")
    (gkg_dir / "20220101001500.gkg.csv.zip").write_bytes(b"not a zip")

    summary = inventory.inventory_raw_archives(
        db_path=db_path,
        export_raw_dir=export_dir,
        gkg_raw_dir=gkg_dir,
        validate_zip=True,
    )

    assert summary.files_planned == 2
    assert summary.files_upserted == 2
    assert summary.status_counts == {"zip_error": 1, "zip_valid": 1}

    with schema.connect_db(db_path) as conn:
        rows = schema.list_raw_file_metadata(conn)

    assert [(row["file_kind"], row["gdelt_stamp"], row["status"]) for row in rows] == [
        ("export", "20220101000000", "zip_valid"),
        ("gkg", "20220101001500", "zip_error"),
    ]
    assert rows[0]["compressed_bytes"] > 0
    assert rows[0]["uncompressed_bytes"] == 3
    assert rows[1]["parse_error"].startswith("bad zip:")
    assert json.loads(rows[0]["metadata_json"])["zip_validation"] == "ok"


def test_main_writes_json_summary_to_stdout_and_honors_limit(tmp_path, monkeypatch, capsys):
    export_dir = tmp_path / "export"
    gkg_dir = tmp_path / "gkg"
    export_dir.mkdir()
    gkg_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"

    _write_zip(export_dir / "20220101000000.export.CSV.zip")
    _write_zip(gkg_dir / "20220101001500.gkg.csv.zip")

    monkeypatch.setattr(
        "sys.argv",
        [
            "gdelt_gkg_raw_inventory",
            "--export-raw-dir",
            str(export_dir),
            "--gkg-raw-dir",
            str(gkg_dir),
            "--db",
            str(db_path),
            "--limit-files",
            "1",
            "--dry-run",
        ],
    )

    assert inventory.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert payload["files_planned"] == 1
    assert payload["files_upserted"] == 0
    assert payload["files"][0]["file_kind"] == "export"
    assert not db_path.exists()


def test_limit_applies_before_zip_validation(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    gkg_dir = tmp_path / "gkg"
    export_dir.mkdir()
    gkg_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    calls = []

    _write_zip(export_dir / "20220101000000.export.CSV.zip")
    _write_zip(export_dir / "20220101010000.export.CSV.zip")
    _write_zip(gkg_dir / "20220101001500.gkg.csv.zip")

    def fake_validate(path):
        calls.append(path.name)
        return {
            "status": "zip_valid",
            "parse_error": None,
            "uncompressed_bytes": 1,
            "metadata": {"zip_validation": "ok", "zip_members": 1},
        }

    monkeypatch.setattr(inventory, "validate_zip_file", fake_validate)

    summary = inventory.inventory_raw_archives(
        db_path=db_path,
        export_raw_dir=export_dir,
        gkg_raw_dir=gkg_dir,
        limit_files=1,
        dry_run=True,
        validate_zip=True,
    )

    assert summary.files_planned == 1
    assert calls == ["20220101000000.export.CSV.zip"]
