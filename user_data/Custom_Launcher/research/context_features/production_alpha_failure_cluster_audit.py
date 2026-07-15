from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SNAPSHOT = ROOT / "user_data" / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_REPORT_DIR = ROOT / "user_data" / "research_news_data" / "context_features" / "reports"


FEATURE_HINTS = (
    "st_failed_breakdown_structure_risk",
    "st_failed_breakout_structure_risk",
    "st_range_position_24h",
    "st_range_position_72h",
    "st_volume_pressure_6h",
    "st_volume_pressure_24h",
    "conf_failed_breakdown_exhaustion_score",
    "conf_failed_breakdown_exhaustion_trigger",
    "conf_structure_bullish_state_score",
    "conf_structure_bearish_state_score",
    "conf_volume_bullish_confirmation",
    "conf_volume_bearish_confirmation",
    "conf_ob_bullish_pressure_agreement",
    "conf_ob_bearish_pressure_agreement",
    "conf_ob_pressure_divergence",
    "conf_ob_bid_absorption",
    "conf_ob_breakdown_failure",
    "conf_ob_support_removed_strength",
    "conf_ob_downside_vacuum_after_support_removed",
    "conf_ob_support_persistence_weak",
    "ob_spot_obts_breakdown_failure_score",
    "ob_spot_obts_bid_absorption_score",
    "ob_spot_obts_price_bounced_support_3h",
    "ob_spot_obts_price_accepted_below_support_3h",
    "ob_spot_obts_pressure_price_agreement",
    "ob_spot_obts_pressure_divergence",
    "ob_spot_obts_support_zone_removed_strength_1h",
    "ob_spot_obts_support_zone_evaporation_1h",
    "ob_spot_obts_downside_vacuum_after_support_removed",
    "ob_linear_obts_breakdown_failure_score",
    "ob_linear_obts_bid_absorption_score",
    "ob_linear_obts_pressure_divergence",
    "ob_inverse_obts_breakdown_failure_score",
    "ob_inverse_obts_bid_absorption_score",
    "ob_inverse_obts_pressure_divergence",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a time/family failure cluster against frozen confluence features.")
    parser.add_argument("--backtest-zip", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--direction", choices=("long", "short", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({
            "mode": "setup-only unless --execute is supplied",
            "backtest_zip": str(args.backtest_zip),
            "snapshot": str(args.snapshot),
            "window": [args.start, args.end],
            "direction": args.direction,
        }, indent=2))
        return 0

    trades = load_trades(args.backtest_zip)
    snapshot = pd.read_parquet(args.snapshot)
    snapshot["date"] = pd.to_datetime(snapshot["date"], utc=True, errors="coerce")
    snapshot = snapshot.dropna(subset=["date"]).sort_values("date")
    feature_cols = [col for col in FEATURE_HINTS if col in snapshot.columns]
    joined = join_features(trades, snapshot[["date", *feature_cols]], feature_cols)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    cluster = joined.loc[(joined["open_date"] >= start) & (joined["open_date"] < end)].copy()
    if args.direction != "both":
        cluster = cluster.loc[cluster["direction"] == args.direction].copy()
    same_direction = joined if args.direction == "both" else joined.loc[joined["direction"] == args.direction].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    cluster_path = args.output_dir / f"production_alpha_{tag}_cluster_trades.csv"
    comparison_path = args.output_dir / f"production_alpha_{tag}_feature_comparison.csv"
    md_path = args.output_dir / f"production_alpha_{tag}_failure_cluster.md"
    cluster.to_csv(cluster_path, index=False)
    comparison = feature_comparison(cluster, same_direction, feature_cols)
    comparison.to_csv(comparison_path, index=False)
    md_path.write_text(markdown_report(cluster, comparison, args), encoding="utf-8")
    print(json.dumps({"cluster": str(cluster_path), "comparison": str(comparison_path), "markdown": str(md_path), "cluster_trades": int(len(cluster))}, indent=2))
    return 0


def load_trades(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        json_names = [name for name in archive.namelist() if name.endswith(".json") and not name.endswith("_config.json")]
        if len(json_names) != 1:
            raise ValueError(f"Expected one result JSON in {path}, found {json_names}")
        payload = json.loads(archive.read(json_names[0]))
    result = next(iter(payload["strategy"].values()))
    trades = pd.DataFrame(result["trades"])
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True, errors="coerce")
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True, errors="coerce")
    trades["direction"] = trades["is_short"].map({True: "short", False: "long"}).fillna("unknown")
    trades["win"] = trades["profit_abs"] > 0
    return trades.dropna(subset=["open_date"]).sort_values("open_date").reset_index(drop=True)


def join_features(trades: pd.DataFrame, snapshot: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    frame = trades.copy()
    frame["decision_date"] = frame["open_date"].dt.floor("h")
    return frame.merge(snapshot.rename(columns={"date": "decision_date"}), on="decision_date", how="left")


def feature_comparison(cluster: pd.DataFrame, reference: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    winners = reference.loc[reference["profit_abs"] > 0]
    losers = reference.loc[reference["profit_abs"] <= 0]
    for col in feature_cols:
        rows.append({
            "feature": col,
            "cluster_mean": mean(cluster[col]),
            "reference_mean": mean(reference[col]),
            "winner_mean": mean(winners[col]),
            "loser_mean": mean(losers[col]),
            "cluster_minus_winner": mean(cluster[col]) - mean(winners[col]),
            "cluster_minus_loser": mean(cluster[col]) - mean(losers[col]),
            "cluster_non_null": int(cluster[col].notna().sum()),
            "reference_non_null": int(reference[col].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("cluster_minus_winner", key=lambda s: s.abs(), ascending=False)


def mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if pd.notna(value) else float("nan")


def markdown_report(cluster: pd.DataFrame, comparison: pd.DataFrame, args: argparse.Namespace) -> str:
    losses = int((cluster["profit_abs"] <= 0).sum())
    wins = int((cluster["profit_abs"] > 0).sum())
    total_profit = float(cluster["profit_abs"].sum()) if not cluster.empty else 0.0
    worst = cluster.sort_values("profit_abs").head(1).to_dict("records")
    worst_row = worst[0] if worst else {}
    top_features = comparison.head(8)
    lines = [
        "# Production Alpha Failure Cluster Audit",
        "",
        "## Trader Question",
        "",
        "What did the market look like at the decision hour for this failure cluster, and does it suggest a better future filter or exit?",
        "",
        "## Cluster",
        "",
        f"- Window: `{args.start}` to `{args.end}`",
        f"- Direction: `{args.direction}`",
        f"- Trades: `{len(cluster)}`",
        f"- Wins/losses: `{wins}/{losses}`",
        f"- Total profit: `{total_profit:.2f}`",
        f"- Worst trade: `{worst_row.get('enter_tag')}` opened `{worst_row.get('open_date')}` with `{float(worst_row.get('profit_abs', 0.0)):.2f}` profit.",
        "",
        "## Biggest Feature Differences",
        "",
    ]
    for _, row in top_features.iterrows():
        lines.append(
            f"- `{row['feature']}`: cluster mean `{row['cluster_mean']:.4f}`, winner mean `{row['winner_mean']:.4f}`, loser mean `{row['loser_mean']:.4f}`."
        )
    lines.extend([
        "",
        "## Initial Read",
        "",
        "Use this as diagnostic evidence only. A filter or exit still needs a separate backtest before promotion.",
    ])
    return "\n".join(lines)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
