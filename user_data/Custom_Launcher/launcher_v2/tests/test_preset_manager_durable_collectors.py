from __future__ import annotations

import json

import pytest

from launcher_v2.preset_manager import (
    AUTO_PRESET_NAME,
    collector_desired,
    read_presets,
    set_collector_desired,
    update_auto_preset,
)


def test_collector_intent_round_trip_and_stale_gui_update(tmp_path):
    path = tmp_path / "presets.json"
    path.write_text(
        json.dumps({"other-preset": {"keep": 1}, AUTO_PRESET_NAME: {"extra": "keep"}}),
        encoding="utf-8",
    )

    assert collector_desired(tmp_path, "news", path) is True
    set_collector_desired(tmp_path, "news", False, path)
    assert collector_desired(tmp_path, "news", path) is False

    def stale_gui_save(auto_preset):
        auto_preset.update({"timeframe": "5m", "extra": "stale-ui-value"})
        return auto_preset

    update_auto_preset(path, stale_gui_save)
    payload = read_presets(path)
    assert payload["other-preset"] == {"keep": 1}
    assert payload[AUTO_PRESET_NAME]["collector_desired"] == {"news": False}
    assert payload[AUTO_PRESET_NAME]["extra"] == "stale-ui-value"
    assert payload[AUTO_PRESET_NAME]["timeframe"] == "5m"


def test_preset_reader_rejects_malformed_and_non_object_files(tmp_path):
    path = tmp_path / "presets.json"
    path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read presets"):
        read_presets(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_presets(path)
