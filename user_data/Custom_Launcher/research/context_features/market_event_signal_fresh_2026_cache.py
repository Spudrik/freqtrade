"""Adapt the frozen event cache engine to the compact 2026 confirmation."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_cache as engine,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_fresh_2026_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = frozen.OUTPUT_ROOT / "freqai_cache"
SUPPORT_PATH = RECORD_ROOT / "event_signal_fresh_outcome_blind_support.json"
CACHE_MANIFEST_PATH = RECORD_ROOT / "event_signal_fresh_cache_manifest.json"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/event_hierarchy/"
    "event_signal_fresh_2026_20260909a/cache"
)
READY_BLOCK = frozen.READY_BLOCK


@contextmanager
def _engine_scope() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "frozen": frozen,
        "ANALYSIS_PATH": ANALYSIS_PATH,
        "RECORD_ROOT": RECORD_ROOT,
        "SUPPORT_PATH": SUPPORT_PATH,
        "CACHE_MANIFEST_PATH": CACHE_MANIFEST_PATH,
        "ARTIFACT_ROOT": ARTIFACT_ROOT,
        "READY_BLOCK": READY_BLOCK,
    }
    previous = {name: getattr(engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def outcome_surface(frame: DataFrame) -> DataFrame:
    with _engine_scope():
        return engine.outcome_surface(frame)


def event_sample_context(samples: DataFrame, events: DataFrame) -> DataFrame:
    with _engine_scope():
        return engine.event_sample_context(samples, events)


def _level_context(parent: DataFrame) -> DataFrame:
    with _engine_scope():
        return engine._level_context(parent)


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    with _engine_scope():
        result = engine.freeze_support(overwrite=overwrite)
        engine._verify_support()
        return result


def materialize_outcomes(*, overwrite: bool = False) -> dict[str, Any]:
    with _engine_scope():
        return engine.materialize_outcomes(overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = (
        freeze_support(overwrite=args.overwrite)
        if args.freeze_only
        else materialize_outcomes(overwrite=args.overwrite)
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
