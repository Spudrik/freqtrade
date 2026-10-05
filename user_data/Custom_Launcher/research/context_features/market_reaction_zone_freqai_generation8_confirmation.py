from __future__ import annotations

# Bound numerical libraries before importing pandas/FreqAI helpers.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_initial_review as review,
)


DEFAULT_FREEZE = review.DEFAULT_FREEZE_PATH


def load_frozen_confirmation(
    path: Path, cohort: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation8_confirmation_outcomes":
        raise ValueError("Generation 8 confirmation selection is not frozen.")
    if tuple(frozen.get("confirmation_seeds", ())) != g8.CONFIRMATION_SEEDS:
        raise ValueError("Generation 8 confirmation seeds drifted.")
    for source in frozen["source_runs"].values():
        for stem in ("result", "decisions"):
            source_path = Path(source[f"{stem}_path"])
            if (
                not source_path.is_file()
                or g0.sha256_file(source_path) != source[f"{stem}_sha256"]
            ):
                raise ValueError(f"Frozen Generation 8 {stem} source changed.")
    selection = frozen["cohorts"].get(cohort)
    if not selection:
        raise ValueError(f"Frozen confirmation lacks cohort {cohort!r}.")
    profiles, comparisons = g8.build_registry(g8.CONFIRMATION_SEEDS)
    selected_profiles = set(selection["profile_ids"])
    selected_comparisons = {
        item["comparison_id"]
        for item in comparisons
        if item["candidate"] in selected_profiles
        and item["baseline"] in selected_profiles
    }
    if selected_comparisons != set(selection["comparison_ids"]):
        raise ValueError("Frozen confirmation profile/comparison registry drifted.")
    if any(profiles[item]["cohort"] != cohort for item in selected_profiles):
        raise ValueError("Frozen confirmation profile crosses cohort boundaries.")
    return frozen, profiles, comparisons


def run_confirmation(
    *,
    run_id: str,
    cohort: str,
    freeze_path: Path,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
) -> dict[str, Any]:
    frozen, profiles, comparisons = load_frozen_confirmation(freeze_path, cohort)
    selection = frozen["cohorts"][cohort]
    pairs = g8.allowed_pairs(cohort)
    manifest, manifest_path, artifact_dir = g8.build_manifest(
        run_id=run_id,
        cohort=cohort,
        pairs=pairs,
        profile_ids=tuple(selection["profile_ids"]),
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=max(1, min(int(profile_workers), g8.MAX_WORKERS)),
        technical_smoke=False,
        registry_profiles=profiles,
        registry_comparisons=comparisons,
        run_stage="seed_confirmation",
        frozen_selection_path=freeze_path,
    )
    active_ids = {item["comparison_id"] for item in manifest["comparisons"]}
    if active_ids != set(selection["comparison_ids"]):
        raise ValueError("Prepared confirmation comparisons differ from the frozen set.")
    if tuple(manifest["seeds"]) != g8.CONFIRMATION_SEEDS:
        raise ValueError("Prepared confirmation manifest has the wrong seeds.")
    audit = g8.preflight_run(manifest, python_exe=python_exe)
    audit_path = manifest_path.parent / "g8_freqai_preflight.json"
    g0.atomic_write_json(audit, audit_path)
    if not audit["passed"]:
        manifest["status"] = "blocked_preflight"
        manifest["preflight"] = audit
        g0.atomic_write_json(manifest, manifest_path)
        return {"status": "blocked_preflight", "preflight": audit}
    returncode = g7f.run_manifest(manifest, manifest_path)
    if returncode:
        return {"status": "profile_failure", "returncode": returncode}
    manifest["status"] = "profiles_completed"
    manifest["finished_at_utc"] = g0.utc_now()
    g0.atomic_write_json(manifest, manifest_path)
    result = g8.score_run(
        manifest,
        record_dir=manifest_path.parent,
        artifact_dir=artifact_dir,
    )
    manifest["status"] = result["status"]
    manifest["result"] = result
    g0.atomic_write_json(manifest, manifest_path)
    return {**result, "result_path": str(manifest_path.parent / "g8_freqai_result.json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the exact Generation 8 seed-17/73 profiles frozen after the joint "
            "seed-42 review."
        )
    )
    parser.add_argument("--cohort", choices=("normal", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--base-config", type=Path, default=g8.DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=g8.DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=g8.MAX_WORKERS)
    args = parser.parse_args(argv)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    result = run_confirmation(
        run_id=args.run_id,
        cohort=args.cohort,
        freeze_path=args.freeze_path,
        base_config=args.base_config,
        python_exe=args.python_exe,
        profile_workers=args.profile_workers,
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0 if str(result["status"]).startswith("completed_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
