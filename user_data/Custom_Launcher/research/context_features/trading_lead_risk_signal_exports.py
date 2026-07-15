from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas import DataFrame, Series

from trading_lead_risk_overlay import (
    DEFAULT_FEATURES,
    REPORTS_DIR,
    btc_sieve_quality_orderbook_guard,
    btc_sieve_quality_story_multiplier,
    extreme_orderbook_invalidation_multiplier,
    load_features,
    story_specific_structure_volume_multiplier,
)


@dataclass(frozen=True)
class RiskSignalExport:
    export_id: str
    source_signal_file: str
    output_signal_file: str
    risk_overlay_id: str
    multiplier_builder: Callable[[DataFrame], Series]


EXPORTS = (
    RiskSignalExport(
        export_id="refined_story_sv",
        source_signal_file="trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah.parquet",
        output_signal_file="trading_lead_signals_20260606_refined_squeeze_priority_story_sv_risk.parquet",
        risk_overlay_id="story_specific_structure_volume_size",
        multiplier_builder=story_specific_structure_volume_multiplier,
    ),
    RiskSignalExport(
        export_id="refined_story_sv_extreme_ob",
        source_signal_file="trading_lead_signals_20260606_refined_multi_scenario_squeeze_priority_no_vah.parquet",
        output_signal_file="trading_lead_signals_20260606_refined_squeeze_priority_story_sv_extreme_ob_risk.parquet",
        risk_overlay_id="structure_volume_with_extreme_ob_guard",
        multiplier_builder=lambda f: (
            story_specific_structure_volume_multiplier(f) * extreme_orderbook_invalidation_multiplier(f)
        ).clip(0.40, 1.35),
    ),
    RiskSignalExport(
        export_id="mtf_story_sv",
        source_signal_file="trading_lead_signals_20260606_mtf_validated_confluence_dedup_fix_best_per_rule_selected_confluence.parquet",
        output_signal_file="trading_lead_signals_20260606_mtf_best_story_sv_risk.parquet",
        risk_overlay_id="story_specific_structure_volume_size",
        multiplier_builder=story_specific_structure_volume_multiplier,
    ),
    RiskSignalExport(
        export_id="mtf_story_sv_extreme_ob",
        source_signal_file="trading_lead_signals_20260606_mtf_validated_confluence_dedup_fix_best_per_rule_selected_confluence.parquet",
        output_signal_file="trading_lead_signals_20260606_mtf_best_story_sv_extreme_ob_risk.parquet",
        risk_overlay_id="structure_volume_with_extreme_ob_guard",
        multiplier_builder=lambda f: (
            story_specific_structure_volume_multiplier(f) * extreme_orderbook_invalidation_multiplier(f)
        ).clip(0.40, 1.35),
    ),
    RiskSignalExport(
        export_id="btc_sieve_quality_story",
        source_signal_file="trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_positive_families.parquet",
        output_signal_file="trading_lead_signals_20260606_btc_sieve_quality_story_risk.parquet",
        risk_overlay_id="btc_sieve_quality_story_size",
        multiplier_builder=btc_sieve_quality_story_multiplier,
    ),
    RiskSignalExport(
        export_id="btc_sieve_quality_story_ob_guard",
        source_signal_file="trading_lead_signals_20260606_btc_sieve_top30_v2_compression_plus_range_break_positive_families.parquet",
        output_signal_file="trading_lead_signals_20260606_btc_sieve_quality_story_ob_guard_risk.parquet",
        risk_overlay_id="btc_sieve_quality_story_with_orderbook_guard",
        multiplier_builder=lambda f: (
            btc_sieve_quality_story_multiplier(f) * btc_sieve_quality_orderbook_guard(f)
        ).clip(0.40, 1.45),
    ),
)


def main() -> int:
    features = load_features(DEFAULT_FEATURES)
    rows: list[dict[str, object]] = []
    for spec in EXPORTS:
        rows.append(export_risk_signals(spec, features))
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": str(DEFAULT_FEATURES),
        "exports": rows,
    }
    meta_path = REPORTS_DIR / "trading_lead_risk_signal_exports_20260606_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def export_risk_signals(spec: RiskSignalExport, features: DataFrame) -> dict[str, object]:
    source_path = REPORTS_DIR / spec.source_signal_file
    output_path = REPORTS_DIR / spec.output_signal_file
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    signals = pd.read_parquet(source_path)
    signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce")
    signals = signals.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    merged = signals.merge(features, on="date", how="left", suffixes=("", "_feature"))
    multiplier = spec.multiplier_builder(merged).replace([float("inf"), float("-inf")], pd.NA)
    signals["risk_multiplier"] = pd.to_numeric(multiplier, errors="coerce").fillna(1.0).clip(0.25, 1.75)
    signals["risk_overlay_id"] = spec.risk_overlay_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_path, index=False)

    changed = signals["risk_multiplier"].ne(1.0)
    return {
        "export_id": spec.export_id,
        "risk_overlay_id": spec.risk_overlay_id,
        "source": str(source_path),
        "output": str(output_path),
        "rows": int(len(signals)),
        "changed_rows": int(changed.sum()),
        "avg_multiplier": float(signals["risk_multiplier"].mean()) if len(signals) else 0.0,
        "min_multiplier": float(signals["risk_multiplier"].min()) if len(signals) else 0.0,
        "max_multiplier": float(signals["risk_multiplier"].max()) if len(signals) else 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
