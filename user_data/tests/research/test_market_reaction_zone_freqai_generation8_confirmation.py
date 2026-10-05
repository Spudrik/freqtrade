# ruff: noqa: S101

from __future__ import annotations

import json

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8_confirmation as confirmation,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


def test_load_frozen_confirmation_rejects_registry_drift(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    decisions_path = tmp_path / "decisions.csv"
    result_path.write_text("{}", encoding="utf-8")
    decisions_path.write_text("x\n", encoding="utf-8")
    frozen = {
        "status": "frozen_before_generation8_confirmation_outcomes",
        "confirmation_seeds": list(g8.CONFIRMATION_SEEDS),
        "source_runs": {
            "normal": {
                "result_path": str(result_path),
                "result_sha256": g0.sha256_file(result_path),
                "decisions_path": str(decisions_path),
                "decisions_sha256": g0.sha256_file(decisions_path),
            }
        },
        "cohorts": {
            "normal": {
                "profile_ids": ["not-a-profile"],
                "comparison_ids": [],
            }
        },
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")

    try:
        confirmation.load_frozen_confirmation(path, "normal")
    except (KeyError, ValueError) as error:
        assert "profile" in str(error).lower()
    else:
        raise AssertionError("Registry drift should fail closed.")
