import ast
import json
import zipfile

from user_data.Custom_Launcher.launcher_v2.services import sieve3_v2_migration as migration


def _write_strategy(path, *, source="", hypothesis="test hypothesis", body=""):
    metadata = f'SOURCE_STRATEGY = "{source}"\n' if source else ""
    path.write_text(
        metadata
        + f'EXIT_HYPOTHESIS = "{hypothesis}"\n'
        + "ENTRY_TAG = \"entry_test\"\n"
        + "class TestStrategy:\n"
        + "    def populate_entry_trend(self, dataframe, metadata):\n"
        + "        return dataframe\n"
        + body,
        encoding="utf-8",
    )


def _write_result(runtime_root, rows):
    result_dir = runtime_root / "entry_sieve" / "results"
    result_dir.mkdir(parents=True)
    path = result_dir / "job.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_filename_wins_when_metadata_would_collapse_a_distinct_source(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_filename_source.py"
    _write_strategy(path, source="sieve3/sieve3_exit_metadata_source.py:MetadataSource")

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["canonical_source"] == "filename_source"
    assert record["source_resolution"] == "filename_metadata_mismatch"
    assert "metadata_filename_source_mismatch" in record["ambiguity_flags"]
    assert record["deletion_eligible"] is False
    assert record["deletion_status"] == "blocked_ambiguous"


def test_windows_source_path_is_parsed_without_collapsing_to_drive_letter():
    value = "C:/FreqTradeStuff/user_data/strategies/sieve3_candidates/sieve3_from_entry_source.py:EntrySource"
    assert migration.canonical_source_slug(value) == "entry_source"


def test_source_entry_stem_has_primary_lineage_precedence(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_filename_source.py"
    _write_strategy(
        path,
        source="sieve3/sieve3_exit_metadata_source.py:MetadataSource",
        body='    SOURCE_ENTRY_STEM = "sieve2_canonical_entry_source"\n',
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["canonical_source"] == "canonical_entry_source"
    assert record["source_resolution"] == "source_entry_stem"
    assert "source_entry_stem_filename_mismatch" in record["ambiguity_flags"]


def test_entry_signature_ignores_generated_branch_class_name(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    first = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    second = strategy_dir / "sieve3_exit_trailing_from_entry_source.py"
    shared_method = (
        "    def populate_entry_trend(self, dataframe, metadata):\n"
        "        dataframe.loc[dataframe['close'] > dataframe['open'], 'enter_long'] = 1\n"
        "        return dataframe\n"
    )
    first.write_text(
        "class FirstGeneratedBranch:\n" + shared_method,
        encoding="utf-8",
    )
    second.write_text(
        "class SecondGeneratedBranch:\n" + shared_method,
        encoding="utf-8",
    )

    first_signature = migration.parse_strategy(first)["entry_signature_sha256"]
    second_signature = migration.parse_strategy(second)["entry_signature_sha256"]

    assert first_signature
    assert first_signature == second_signature


def test_entry_signature_includes_wrapped_buy_parameter_and_lowercase_timeframe(tmp_path):
    def signature(name, *, default, timeframe):
        path = tmp_path / name
        path.write_text(
            "def tagged_parameter(value):\n"
            "    return value\n"
            "class Strategy:\n"
            f"    timeframe = {timeframe!r}\n"
            "    threshold = tagged_parameter("
            f"BooleanParameter(default={default!r}, space='buy', optimize=False))\n"
            "    def populate_entry_trend(self, dataframe, metadata):\n"
            "        return dataframe\n",
            encoding="utf-8",
        )
        return migration.parse_strategy(path)["entry_signature_sha256"]

    base = signature("base.py", default=True, timeframe="1h")
    changed_default = signature("changed_default.py", default=False, timeframe="1h")
    changed_timeframe = signature("changed_timeframe.py", default=True, timeframe="4h")

    assert base != changed_default
    assert base != changed_timeframe


def test_local_dependency_closure_records_current_hash_as_nonhistorical():
    dependency = migration._local_dependency_closure(
        b"from user_data.strategies.sieve_guard_helpers import add_sieve2_guard_indicators\n"
    )

    assert dependency["status"] == "current_dependencies_recorded_historical_versions_unavailable"
    assert len(dependency["fingerprint_sha256"]) == 64
    helper = next(
        row
        for row in dependency["dependencies"]
        if row["path"].endswith("user_data/strategies/sieve_guard_helpers.py")
    )
    assert len(helper["current_sha256"]) == 64
    assert helper["historical_version_status"] == "not_archived_with_backtest"

    missing = migration._local_dependency_closure(
        b"from user_data.strategies.entry_sieve_tools import entry_sieve_minimal_roi\n"
    )
    assert missing["status"] == "current_dependencies_missing_historical_versions_unavailable"
    assert missing["unresolved_local_modules"] == ["user_data.strategies.entry_sieve_tools"]


def test_any_successful_physical_row_preserves_file_after_later_error(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    _write_strategy(path, source="sieve3/sieve3_exit_entry_source.py:EntrySource")
    base = {
        "job_id": "job",
        "strategy": path.stem,
        "strategy_class": "TestStrategy",
        "strategy_file": str(path),
        "training_window": "train",
        "validation_window": "validate",
        "random_state": "42",
        "sampling_seed": "42",
    }
    _write_result(
        tmp_path / "runtime",
        [
            {**base, "finished_at": "2026-01-01T00:00:00+00:00", "status": "ok"},
            {**base, "finished_at": "2026-01-02T00:00:00+00:00", "status": "error"},
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["tested_successfully"] is True
    assert record["category"] == "tested_legacy"
    assert record["deletion_status"] == "preserve_tested_original"
    assert record["successful_row_count"] == 1
    assert record["successful_result_ids"] == ["job"]


def test_basename_only_result_is_protected_alias_not_exact_test(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    _write_strategy(path, source="sieve3/sieve3_exit_entry_source.py:EntrySource")
    archive = tmp_path / "result.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("result_TestStrategy.py", path.read_bytes())
        handle.writestr(
            "result_TestStrategy.json",
            json.dumps({"strategy_name": "TestStrategy", "params": {}}),
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "job_id": "job",
                "finished_at": "2026-01-01T00:00:00+00:00",
                "status": "ok",
                "strategy": path.stem,
                "strategy_class": "TestStrategy",
                "strategy_file": str(tmp_path / "old" / path.name),
                "training_window": "train",
                "validation_window": "validate",
                "random_state": "42",
                "sampling_seed": "1337",
                "backtest_file": str(archive),
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["category"] == "protected_alias"
    assert record["tested_successfully"] is False
    assert record["protected_by_alias_evidence"] is True
    assert record["exact_successful_row_count"] == 0
    assert record["alias_successful_row_count"] == 1
    assert record["deletion_status"] == "preserve_ambiguous_alias"
    assert ledger["summary"]["protected_alias_files"] == 1
    evidence = record["successful_evidence_rows"][0]
    assert evidence["resolution"] == "basename_alias"
    assert evidence["line_number"] == 1
    assert len(evidence["row_sha256"]) == 64
    assert len(evidence["archive_sha256"]) == 64
    assert evidence["source_member"] == "result_TestStrategy.py"
    assert len(evidence["source_sha256"]) == 64
    assert evidence["params_member"] == "result_TestStrategy.json"
    assert len(evidence["params_sha256"]) == 64


def test_latest_backtest_snapshot_detects_current_file_mismatch(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    _write_strategy(path, source="sieve3/sieve3_exit_entry_source.py:EntrySource")
    archive = tmp_path / "result.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("result_TestStrategy.py", "historical strategy bytes")
        handle.writestr(
            "result_TestStrategy.json",
            json.dumps({"strategy_name": "TestStrategy", "params": {}}),
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "job_id": "job",
                "finished_at": "2026-01-01T00:00:00+00:00",
                "status": "ok",
                "strategy": path.stem,
                "strategy_class": "TestStrategy",
                "strategy_file": str(path),
                "training_window": "train",
                "validation_window": "validate",
                "random_state": "42",
                "sampling_seed": "42",
                "backtest_file": str(archive),
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["snapshot_status"] == "mismatch"
    assert "current_differs_from_latest_tested_snapshot" in record["ambiguity_flags"]
    assert record["deletion_status"] == "preserve_tested_original"


def test_all_successful_snapshots_are_checked_for_historical_divergence(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    _write_strategy(path, source="sieve3/sieve3_exit_entry_source.py:EntrySource")
    old_archive = tmp_path / "old.zip"
    current_archive = tmp_path / "current.zip"
    with zipfile.ZipFile(old_archive, "w") as handle:
        handle.writestr("result_TestStrategy.py", "historical strategy bytes")
        handle.writestr(
            "result_TestStrategy.json",
            json.dumps({"strategy_name": "TestStrategy", "params": {}}),
        )
    with zipfile.ZipFile(current_archive, "w") as handle:
        handle.writestr("result_TestStrategy.py", path.read_bytes())
        handle.writestr(
            "result_TestStrategy.json",
            json.dumps({"strategy_name": "TestStrategy", "params": {}}),
        )
    base = {
        "job_id": "job",
        "status": "ok",
        "strategy": path.stem,
        "strategy_class": "TestStrategy",
        "strategy_file": str(path),
        "training_window": "train",
        "validation_window": "validate",
        "random_state": "42",
        "sampling_seed": "42",
    }
    _write_result(
        tmp_path / "runtime",
        [
            {**base, "finished_at": "2026-01-01T00:00:00+00:00", "backtest_file": str(old_archive)},
            {**base, "finished_at": "2026-01-02T00:00:00+00:00", "backtest_file": str(current_archive)},
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["snapshot_status"] == "match"
    assert record["historical_snapshot_status"] == "historical_divergence"
    assert record["historical_snapshot_archives_checked"] == 2
    assert record["historical_snapshot_match_count"] == 1
    assert record["historical_snapshot_mismatch_count"] == 1
    assert len(record["historical_snapshot_hashes"]) == 2
    assert "historical_tested_snapshot_divergence" in record["ambiguity_flags"]
    assert record["deletion_status"] == "preserve_tested_original"


def test_group_summary_separates_tested_untested_pilot_and_pattern(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    tested = strategy_dir / "sieve3_exit_breakeven_from_triangle_breakout_long.py"
    untested = strategy_dir / "sieve3_exit_trailing_from_triangle_breakout_long.py"
    pilot = strategy_dir / "sieve3_rework_exit_layered_from_triangle_breakout_long.py"
    for path in (tested, untested, pilot):
        _write_strategy(path)
    _write_result(
        tmp_path / "runtime",
        [
            {
                "job_id": "job",
                "finished_at": "2026-01-01T00:00:00+00:00",
                "status": "ok",
                "strategy": tested.stem,
                "strategy_file": str(tested),
                "training_window": "train",
                "validation_window": "validate",
                "random_state": "42",
                "sampling_seed": "42",
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    assert ledger["summary"]["tested_legacy_files"] == 1
    assert ledger["summary"]["untested_legacy_files"] == 1
    assert ledger["summary"]["refined_pilot_files"] == 1
    assert ledger["summary"]["canonical_source_groups"] == 1
    group = ledger["source_groups"][0]
    assert group["pattern_source"] is True
    assert group["v2_replacement"] == "sieve3_V2_pattern_integrated_from_triangle_breakout_long.py"
    assert group["tested_legacy_files"] == [tested.name]
    assert group["untested_legacy_files"] == [untested.name]
    assert group["refined_pilot_files"] == [pilot.name]
    assert group["deletion_approved"] is False


def test_source_group_ambiguity_blocks_untested_member(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    path = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    _write_strategy(path)

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    record = ledger["records"][0]
    assert record["category"] == "untested_legacy"
    assert "baseline_result_not_found_for_current_entry_surface" in record["source_group_ambiguity_flags"]
    assert "source_group_ambiguity" in record["ambiguity_flags"]
    assert record["deletion_status"] == "blocked_ambiguous"


def test_baseline_lookup_prefers_two_two_then_winrate(tmp_path):
    path = tmp_path / "lookup.md"
    path.write_text(
        "| sieve2_source | s3_files | tp/sl | ret% | wr% | trades | dd% | pf | baseline_window | baseline_job |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
        "| `sieve2_entry_source` | 2 | `3/3` | 10.00 | 90.00 | 10 | 1.00 | 2.00 | `window` | `job_high_wr` |\n"
        "| `sieve2_entry_source` | 2 | `2/2` | 2.00 | 55.00 | 20 | 1.00 | 1.20 | `window` | `job_two_two` |\n",
        encoding="utf-8",
    )

    selected = migration.parse_baseline_lookup(path)

    assert selected["entry_source"]["job_id"] == "job_two_two"
    assert selected["entry_source"]["lookup_candidate_count"] == 2


def test_baseline_lineage_recovers_exact_buy_params_from_backtest_snapshot(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    strategy = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    _write_strategy(
        strategy,
        body=(
            "    threshold = IntParameter(1, 10, default=7, space='buy', optimize=False)\n"
            "    use_guard = BooleanParameter(default=True, space='buy', optimize=False)\n"
        ),
    )
    archive = tmp_path / "baseline.zip"
    params = {
        "strategy_name": "Sieve2EntrySource",
        "params": {"buy": {"threshold": 7, "use_guard": True}},
        "ft_stratparam_v": 1,
    }
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("result_Sieve2EntrySource.json", json.dumps(params))
        handle.writestr(
            "result_Sieve2EntrySource.py",
                "ENTRY_MODE = 'entry_source'\n"
                "class Sieve2EntrySource:\n"
                "    threshold = IntParameter(1, 10, default=7, space='buy', optimize=False)\n"
                "    use_guard = BooleanParameter(default=True, space='buy', optimize=False)\n"
                "    def populate_indicators(self, dataframe, metadata):\n"
            "        dataframe['source_value'] = dataframe['close']\n"
            "        return dataframe\n"
            "    def populate_entry_trend(self, dataframe, metadata):\n"
            "        return dataframe\n",
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "job_id": "baseline_job",
                "finished_at": "2026-01-01T00:00:00+00:00",
                "status": "ok",
                "strategy": "sieve2_entry_source",
                "strategy_class": "Sieve2EntrySource",
                "strategy_file": str(tmp_path / "sieve2_entry_source.py"),
                "training_window": "window",
                "take_profit_pct": "2",
                "stoploss_pct": "2",
                "profit_total": 0.02,
                "trade_count": 20,
                "winrate": 0.55,
                "profit_factor": 1.2,
                "max_drawdown_pct": 0.01,
                "backtest_file": str(archive),
            }
        ],
    )
    lookup = tmp_path / "lookup.md"
    lookup.write_text(
        "| `sieve2_entry_source` | 1 | `2/2` | 2.00 | 55.00 | 20 | 1.00 | 1.20 | `window` | `baseline_job` |\n",
        encoding="utf-8",
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=lookup,
    )

    baseline = ledger["source_groups"][0]["sieve2_baseline"]
    assert baseline["status"] == "verified"
    assert baseline["locked_buy_params"] == {"threshold": 7, "use_guard": True}
    assert baseline["snapshot_member"] == "result_Sieve2EntrySource.json"
    assert baseline["baseline_entry_signature_sha256"]
    assert ledger["summary"]["verified_sieve2_baselines"] == 1


def test_identical_baseline_rows_copied_between_ledgers_are_not_ambiguous(tmp_path):
    runtime_root = tmp_path / "runtime"
    result_dir = runtime_root / "entry_sieve" / "results"
    result_dir.mkdir(parents=True)
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "result_Sieve2EntrySource.json",
            json.dumps(
                {
                    "strategy_name": "Sieve2EntrySource",
                    "params": {"buy": {"threshold": 7}},
                }
            ),
        )
        handle.writestr(
            "result_Sieve2EntrySource.py",
            "class Sieve2EntrySource:\n"
            "    def populate_entry_trend(self, dataframe, metadata):\n"
            "        return dataframe\n",
        )
    row = {
        "job_id": "baseline_job",
        "finished_at": "2026-01-01T00:00:00+00:00",
        "status": "ok",
        "strategy": "sieve2_entry_source",
        "strategy_class": "Sieve2EntrySource",
        "strategy_file": str(tmp_path / "sieve2_entry_source.py"),
        "training_window": "window",
        "take_profit_pct": "2",
        "stoploss_pct": "2",
        "profit_total": 0.02,
        "trade_count": 20,
        "winrate": 0.55,
        "profit_factor": 1.2,
        "max_drawdown_pct": 0.01,
        "backtest_file": str(archive),
    }
    payload = json.dumps(row) + "\n"
    (result_dir / "canonical.jsonl").write_text(payload, encoding="utf-8")
    (result_dir / "copied.jsonl").write_text(payload, encoding="utf-8")
    lookup = tmp_path / "lookup.md"
    lookup.write_text(
        "| `sieve2_entry_source` | 1 | `2/2` | 2.00 | 55.00 | 20 | 1.00 | 1.20 | `window` | `baseline_job` |\n",
        encoding="utf-8",
    )

    resolved, stats = migration.resolve_baseline_lineage(runtime_root, lookup)

    assert resolved["entry_source"]["status"] == "verified"
    assert "duplicate_matching_baseline_rows" not in resolved["entry_source"]["ambiguity_flags"]
    assert stats["duplicate_result_row_copies"] == 1
    assert stats["ambiguous_result_rows"] == 0


def test_current_entry_params_override_stale_quick_lookup_hint(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    strategy = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    strategy.write_text(
        "ENTRY_MODE = 'entry_source'\n"
        "class Strategy:\n"
        "    threshold = IntParameter(1, 10, default=9, space='buy', optimize=False)\n"
        "    def populate_entry_trend(self, dataframe, metadata):\n"
        "        return dataframe\n",
        encoding="utf-8",
    )

    def archive(path, class_name, threshold):
        with zipfile.ZipFile(path, "w") as handle:
            handle.writestr(
                f"result_{class_name}.json",
                json.dumps(
                    {
                        "strategy_name": class_name,
                        "params": {"buy": {"threshold": threshold}},
                    }
                ),
            )
            handle.writestr(
                f"result_{class_name}.py",
                "ENTRY_MODE = 'entry_source'\n"
                f"class {class_name}:\n"
                f"    threshold = IntParameter(1, 10, default={threshold}, space='buy', optimize=False)\n"
                "    def populate_entry_trend(self, dataframe, metadata):\n"
                "        return dataframe\n",
            )

    stale_archive = tmp_path / "stale.zip"
    current_archive = tmp_path / "current.zip"
    archive(stale_archive, "StaleSource", 7)
    archive(current_archive, "CurrentSource", 9)
    base = {
        "status": "ok",
        "strategy": "sieve2_entry_source",
        "strategy_file": str(tmp_path / "sieve2_entry_source.py"),
        "training_window": "window",
        "take_profit_pct": "3",
        "stoploss_pct": "3",
        "profit_factor": 2.0,
    }
    _write_result(
        tmp_path / "runtime",
        [
            {
                **base,
                "job_id": "stale_lookup_job",
                "finished_at": "2026-01-01T00:00:00+00:00",
                "strategy_class": "StaleSource",
                "profit_total": 0.01,
                "trade_count": 4,
                "winrate": 0.75,
                "max_drawdown_pct": 0.002,
                "backtest_file": str(stale_archive),
            },
            {
                **base,
                "job_id": "promoted_job",
                "finished_at": "2026-01-02T00:00:00+00:00",
                "strategy_class": "CurrentSource",
                "profit_total": 0.03,
                "trade_count": 97,
                "winrate": 0.57,
                "max_drawdown_pct": 0.01,
                "backtest_file": str(current_archive),
            },
        ],
    )
    lookup = tmp_path / "lookup.md"
    lookup.write_text(
        "| `sieve2_entry_source` | 1 | `3/3` | 1.00 | 75.00 | 4 | 0.20 | 2.00 | `window` | `stale_lookup_job` |\n",
        encoding="utf-8",
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=lookup,
    )

    baseline = ledger["source_groups"][0]["sieve2_baseline"]
    assert baseline["status"] == "verified"
    assert baseline["job_id"] == "promoted_job"
    assert baseline["locked_buy_params"] == {"threshold": 9}
    assert baseline["selected_by"] == "exact_current_params_and_entry_signature"
    assert baseline["baseline_lookup_hint"]["job_id"] == "stale_lookup_job"


def test_complete_pattern_source_resolves_sieve2_name_without_prefix(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    strategy = strategy_dir / "sieve3_exit_breakeven_from_complete_pattern_avwap_reject_short.py"
    strategy.write_text(
        "ENTRY_MODE = 'avwap_reject_short'\n"
        "class Strategy:\n"
        "    threshold = IntParameter(1, 10, default=9, space='buy', optimize=False)\n"
        "    def populate_entry_trend(self, dataframe, metadata):\n"
        "        return dataframe\n",
        encoding="utf-8",
    )
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "result_Sieve2AvwapRejectShort.json",
            json.dumps(
                {
                    "strategy_name": "Sieve2AvwapRejectShort",
                    "params": {"buy": {"threshold": 9}},
                }
            ),
        )
        handle.writestr(
            "result_Sieve2AvwapRejectShort.py",
            "ENTRY_MODE = 'avwap_reject_short'\n"
            "class Sieve2AvwapRejectShort:\n"
            "    threshold = IntParameter(1, 10, default=9, space='buy', optimize=False)\n"
            "    def populate_entry_trend(self, dataframe, metadata):\n"
            "        return dataframe\n",
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "job_id": "promotion_job",
                "finished_at": "2026-01-01T00:00:00+00:00",
                "status": "ok",
                "strategy": "sieve2_avwap_reject_short",
                "strategy_class": "Sieve2AvwapRejectShort",
                "strategy_file": str(tmp_path / "sieve2_avwap_reject_short.py"),
                "training_window": "window",
                "take_profit_pct": "2",
                "stoploss_pct": "2",
                "profit_total": 0.02,
                "trade_count": 30,
                "winrate": 0.60,
                "profit_factor": 1.5,
                "max_drawdown_pct": 0.01,
                "backtest_file": str(archive),
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    baseline = ledger["source_groups"][0]["sieve2_baseline"]
    assert baseline["status"] == "verified"
    assert baseline["sieve2_source"] == "sieve2_avwap_reject_short"
    assert "avwap_reject_short" in baseline["source_context"]["sieve2_source_aliases"]


def test_baseline_entry_signature_includes_entry_mode():
    first = ast.parse(
        "ENTRY_MODE = 'first'\n"
        "class Strategy:\n"
        "    def populate_indicators(self, dataframe, metadata):\n"
        "        return dataframe\n"
    )
    second = ast.parse(
        "ENTRY_MODE = 'second'\n"
        "class Strategy:\n"
        "    def populate_indicators(self, dataframe, metadata):\n"
        "        return dataframe\n"
    )

    first_signature = migration._entry_signature(
        first,
        migration._literal_assignments(first.body),
        migration.BASELINE_ENTRY_METHODS,
    )
    second_signature = migration._entry_signature(
        second,
        migration._literal_assignments(second.body),
        migration.BASELINE_ENTRY_METHODS,
    )

    assert first_signature != second_signature


def test_sparse_params_overlay_archived_source_defaults_into_effective_lock(tmp_path):
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "result_Sieve2EntrySource.json",
            json.dumps(
                {
                    "strategy_name": "Sieve2EntrySource",
                    "params": {"buy": {"threshold": 9}},
                }
            ),
        )
        handle.writestr(
            "result_Sieve2EntrySource.py",
            "class Sieve2EntrySource:\n"
            "    threshold = IntParameter(1, 10, default=7, space='buy')\n"
            "    use_guard = BooleanParameter(default=True, space='buy')\n",
        )

    snapshot = migration._baseline_snapshot(
        {"backtest_file": str(archive), "strategy_class": "Sieve2EntrySource"}
    )

    assert snapshot["snapshot_status"] == "verified"
    assert snapshot["archived_source_buy_defaults"] == {
        "threshold": 7,
        "use_guard": True,
    }
    assert snapshot["selected_buy_params"] == {"threshold": 9}
    assert snapshot["effective_buy_lock"] == {"threshold": 9, "use_guard": True}
    assert snapshot["locked_buy_params"] == snapshot["effective_buy_lock"]


def test_explicit_source_members_outvote_filename_only_predecessors(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    (strategy_dir / "sieve3_exit_breakeven_from_entry_source.py").write_text(
        "class OldBranch:\n"
        "    threshold = IntParameter(1, 10, default=7, space='buy')\n",
        encoding="utf-8",
    )
    (strategy_dir / "sieve3_exit_trailing_from_entry_source.py").write_text(
        "SOURCE_ENTRY_STEM = 'sieve2_entry_source'\n"
        "class MaintainedBranch:\n"
        "    threshold = IntParameter(1, 10, default=9, space='buy')\n",
        encoding="utf-8",
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )
    context = ledger["source_groups"][0]["sieve2_baseline"]["source_context"]

    assert context["foundation_member_policy"] == "explicit_source_entry_stem_members"
    assert context["selected_buy_defaults"] == {"threshold": 9}
    assert len(context["excluded_filename_predecessors"]) == 1
    assert context["excluded_filename_predecessors"][0].replace("\\", "/").endswith(
        "/strategies/sieve3_exit_breakeven_from_entry_source.py"
    )


def test_unknown_complete_pattern_alias_is_not_inferred(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    (strategy_dir / "sieve3_exit_breakeven_from_complete_pattern_unknown.py").write_text(
        "class Strategy:\n"
        "    threshold = IntParameter(1, 10, default=9, space='buy')\n",
        encoding="utf-8",
    )
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "result_Sieve2Unknown.json",
            json.dumps(
                {"strategy_name": "Sieve2Unknown", "params": {"buy": {"threshold": 9}}}
            ),
        )
        handle.writestr(
            "result_Sieve2Unknown.py",
            "class Sieve2Unknown:\n"
            "    threshold = IntParameter(1, 10, default=9, space='buy')\n",
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "status": "ok",
                "strategy": "sieve2_unknown",
                "strategy_class": "Sieve2Unknown",
                "strategy_file": str(tmp_path / "sieve2_unknown.py"),
                "backtest_file": str(archive),
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )

    assert ledger["source_groups"][0]["sieve2_baseline"]["status"] == (
        "baseline_result_not_found"
    )


def test_explicit_promotion_snapshot_reports_unlocked_current_defaults(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    strategy = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    strategy.write_text(
        "SOURCE_ENTRY_STEM = 'sieve2_entry_source'\n"
        "SOURCE_SIEVE2_STRATEGY = 'sieve2_entry_source'\n"
        "SOURCE_RESULT_BATCH = 'promotion_job'\n"
        "class Strategy:\n"
        "    threshold = IntParameter(1, 10, default=7, space='buy')\n",
        encoding="utf-8",
    )
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "result_Sieve2EntrySource.json",
            json.dumps(
                {
                    "strategy_name": "Sieve2EntrySource",
                    "params": {"buy": {"threshold": 9}},
                }
            ),
        )
        handle.writestr(
            "result_Sieve2EntrySource.py",
            "class Sieve2EntrySource:\n"
            "    threshold = IntParameter(1, 10, default=7, space='buy')\n",
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "job_id": "promotion_job",
                "status": "ok",
                "strategy": "sieve2_entry_source",
                "strategy_class": "Sieve2EntrySource",
                "strategy_file": str(tmp_path / "sieve2_entry_source.py"),
                "backtest_file": str(archive),
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )
    baseline = ledger["source_groups"][0]["sieve2_baseline"]

    assert baseline["snapshot_lineage_status"] == "verified"
    assert baseline["status"] == "promotion_params_not_locked"
    assert baseline["effective_buy_lock"] == {"threshold": 9}


def test_promoted_value_outside_archived_domain_blocks_static_verification(tmp_path):
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    strategy = strategy_dir / "sieve3_exit_breakeven_from_entry_source.py"
    strategy.write_text(
        "class Strategy:\n"
        "    tolerance = DecimalParameter(0.0, 0.03, default=0.04, space='buy')\n",
        encoding="utf-8",
    )
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "result_Sieve2EntrySource.json",
            json.dumps(
                {
                    "strategy_name": "Sieve2EntrySource",
                    "params": {"buy": {"tolerance": 0.04}},
                }
            ),
        )
        handle.writestr(
            "result_Sieve2EntrySource.py",
            "class Sieve2EntrySource:\n"
            "    tolerance = DecimalParameter(0.0, 0.03, default=0.02, space='buy')\n",
        )
    _write_result(
        tmp_path / "runtime",
        [
            {
                "status": "ok",
                "strategy": "sieve2_entry_source",
                "strategy_class": "Sieve2EntrySource",
                "strategy_file": str(tmp_path / "sieve2_entry_source.py"),
                "backtest_file": str(archive),
            }
        ],
    )

    ledger = migration.build_migration_ledger(
        strategy_dir,
        tmp_path / "runtime",
        inspect_snapshots=False,
        baseline_lookup=tmp_path / "missing.md",
    )
    baseline = ledger["source_groups"][0]["sieve2_baseline"]

    assert baseline["status"] == "promoted_params_outside_declared_domain"
    assert baseline["effective_buy_lock_domain_violations"] == [
        {
            "parameter": "tolerance",
            "value": 0.04,
            "reason": "value_outside_archived_parameter_domain",
            "domain": {"type": "numeric", "low": 0.0, "high": 0.03},
        }
    ]


def test_result_scan_conserves_orphans_and_logical_duplicate_occurrences(tmp_path):
    runtime_root = tmp_path / "runtime"
    result_dir = runtime_root / "entry_sieve" / "results"
    result_dir.mkdir(parents=True)
    row = {
        "status": "ok",
        "strategy": "sieve3_missing_strategy",
        "strategy_class": "MissingStrategy",
        "strategy_file": str(tmp_path / "missing" / "sieve3_missing_strategy.py"),
    }
    payload = json.dumps(row) + "\n"
    (result_dir / "first.jsonl").write_text(payload, encoding="utf-8")
    (result_dir / "copy.jsonl").write_text(payload, encoding="utf-8")

    _, stats = migration.scan_successful_results(runtime_root, [])

    assert stats["successful_rows"] == 2
    assert stats["logical_successful_rows"] == 1
    assert stats["duplicate_successful_row_occurrences"] == 1
    assert stats["classified_successful_rows"] == 2
    assert len(stats["orphan_successful_evidence"]) == 2
    assert len({row["occurrence_id"] for row in stats["orphan_successful_evidence"]}) == 2


def test_archive_snapshot_rejects_wrong_class_fallback(tmp_path):
    archive = tmp_path / "result.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("result_OtherStrategy.py", "class OtherStrategy: pass\n")
        handle.writestr(
            "result_OtherStrategy.json",
            json.dumps({"strategy_name": "OtherStrategy", "params": {}}),
        )

    snapshot = migration._read_archive_snapshot(str(archive), "ExpectedStrategy")

    assert snapshot["archive_status"] == "snapshot_ambiguous"
