import sqlite3

from user_data.Custom_Launcher.research.context_features import gdelt_gkg_normalized_schema as schema


def test_init_db_creates_normalized_tables_and_indexes(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    schema.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert {
        "gdelt_raw_files",
        "gdelt_events_silver",
        "gkg_documents_silver",
        "story_clusters",
        "story_members",
        "gdelt_gkg_hourly_features_v1",
    }.issubset(tables)
    assert {
        "idx_gdelt_raw_files_kind_stamp",
        "idx_gdelt_events_silver_event_time",
        "idx_gdelt_events_silver_available_at",
        "idx_gkg_documents_silver_url_hash",
        "idx_gkg_documents_silver_available_at",
        "idx_story_members_member",
        "idx_gdelt_gkg_hourly_features_v1_generated_at",
    }.issubset(indexes)
    with sqlite3.connect(db_path) as conn:
        gdelt_columns = {row[1] for row in conn.execute("PRAGMA table_info(gdelt_events_silver)").fetchall()}
        gkg_columns = {row[1] for row in conn.execute("PRAGMA table_info(gkg_documents_silver)").fetchall()}

    assert {"available_at", "raw_ref_json", "topic", "impact_channel", "severity_proxy"}.issubset(gdelt_columns)
    assert {"available_at", "published_at", "raw_ref_json", "v2_themes_json", "extras_json"}.issubset(gkg_columns)


def test_url_canonicalization_and_hashing_are_stable():
    left = "HTTPS://www.Example.com:443/news/path/?b=2&utm_source=x&a=1#fragment"
    right = "https://example.com/news/path?a=1&b=2"

    assert schema.canonicalize_url(left) == schema.canonicalize_url(right)
    assert schema.url_hash(left) == schema.url_hash(right)
    assert schema.raw_file_id("GKG", "20220728000000", left) == schema.raw_file_id(
        "gkg",
        "20220728000000",
        right,
    )
    assert schema.canonicalize_url("example.com/news/path/") == "http://example.com/news/path"


def test_raw_file_metadata_upsert_is_idempotent(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    schema.init_db(db_path)

    with schema.connect_db(db_path) as conn:
        raw_id = schema.upsert_raw_file_metadata(
            conn,
            file_kind="gkg",
            gdelt_stamp="20220728000000",
            source_url="HTTPS://www.Example.com/file.gkg.csv.zip?utm_campaign=test",
            local_path=tmp_path / "20220728000000.gkg.csv.zip",
            file_sha256="abc123",
            compressed_bytes=100,
            row_count=5,
            fetched_at="2022-07-28T00:00:00+00:00",
            status="downloaded",
            metadata={"worker": "A"},
        )
        raw_id_again = schema.upsert_raw_file_metadata(
            conn,
            file_kind="GKG",
            gdelt_stamp="20220728000000",
            source_url="https://example.com/file.gkg.csv.zip",
            local_path=tmp_path / "20220728000000.gkg.csv.zip",
            file_sha256="def456",
            compressed_bytes=120,
            row_count=6,
            parsed_at="2022-07-28T00:01:00+00:00",
            status="parsed",
            metadata={"worker": "A", "attempt": 2},
        )

        rows = schema.list_raw_file_metadata(conn)

    assert raw_id_again == raw_id
    assert len(rows) == 1
    assert rows[0]["raw_file_id"] == raw_id
    assert rows[0]["file_kind"] == "gkg"
    assert rows[0]["canonical_source_url"] == "https://example.com/file.gkg.csv.zip"
    assert rows[0]["file_sha256"] == "def456"
    assert rows[0]["compressed_bytes"] == 120
    assert rows[0]["row_count"] == 6
    assert rows[0]["status"] == "parsed"
    assert rows[0]["metadata_json"] == '{"attempt":2,"worker":"A"}'
