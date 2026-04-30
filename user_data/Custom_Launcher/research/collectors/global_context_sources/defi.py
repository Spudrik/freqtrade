from __future__ import annotations

from typing import Any

from .common import clamp, compact_usd, float_or_none, fmt_pct, nested_float, pct_change, result, score_signal


def normalize_defillama_stablecoins(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> list[dict[str, Any]]:
    assets = payload.get("peggedAssets") if isinstance(payload, dict) else []
    totals = {"current": 0.0, "day": 0.0, "week": 0.0, "month": 0.0}
    top_assets: list[tuple[str, float]] = []
    for asset in assets if isinstance(assets, list) else []:
        if not isinstance(asset, dict):
            continue
        current = nested_float(asset, "circulating", "peggedUSD")
        day = nested_float(asset, "circulatingPrevDay", "peggedUSD")
        week = nested_float(asset, "circulatingPrevWeek", "peggedUSD")
        month = nested_float(asset, "circulatingPrevMonth", "peggedUSD")
        totals["current"] += current or 0.0
        totals["day"] += day or 0.0
        totals["week"] += week or 0.0
        totals["month"] += month or 0.0
        name = str(asset.get("symbol") or asset.get("name") or "")
        if name and current:
            top_assets.append((name, current))
    change_1d = pct_change(totals["current"], totals["day"])
    change_7d = pct_change(totals["current"], totals["week"])
    score = clamp(50.0 + (change_7d or 0.0) * 10.0 + (change_1d or 0.0) * 5.0)
    top = ", ".join(f"{name} ${compact_usd(value)}" for name, value in sorted(top_assets, key=lambda item: item[1], reverse=True)[:3])
    notes = f"supply ${compact_usd(totals['current'])}, 1d {fmt_pct(change_1d)}, 7d {fmt_pct(change_7d)}, top {top}"
    primary = result(
        source,
        metric_key="stablecoin_supply_change_7d",
        score=score,
        calc_score=score,
        signal=score_signal(score),
        value=change_7d,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )
    rows = [primary]
    rows.append(
        result(
            source,
            metric_key="stablecoin_supply_usd",
            score=None,
            signal="",
            value=totals["current"],
            unit="usd",
            notes="Current aggregate stablecoin supply from DeFiLlama.",
            raw=None,
        )
    )
    if change_1d is not None:
        rows.append(
            result(
                source,
                metric_key="stablecoin_supply_change_1d",
                score=None,
                signal="",
                value=change_1d,
                unit="percent",
                notes="One-day aggregate stablecoin supply change from DeFiLlama.",
                raw=None,
            )
        )
    return rows


def normalize_defillama_chains(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload if isinstance(payload, list) else []
    total_tvl = 0.0
    weighted_1d = 0.0
    weighted_7d = 0.0
    top_chains: list[tuple[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tvl = float_or_none(row.get("tvl")) or 0.0
        if tvl <= 0:
            continue
        change_1d = float_or_none(row.get("change_1d")) or 0.0
        change_7d = float_or_none(row.get("change_7d")) or 0.0
        total_tvl += tvl
        weighted_1d += tvl * change_1d
        weighted_7d += tvl * change_7d
        name = str(row.get("name") or "")
        if name:
            top_chains.append((name, tvl))
    avg_1d = (weighted_1d / total_tvl) if total_tvl > 0 else None
    avg_7d = (weighted_7d / total_tvl) if total_tvl > 0 else None
    score = clamp(50.0 + (avg_7d or 0.0) * 2.0 + (avg_1d or 0.0) * 3.0)
    top = ", ".join(f"{name} ${compact_usd(value)}" for name, value in sorted(top_chains, key=lambda item: item[1], reverse=True)[:3])
    notes = f"TVL ${compact_usd(total_tvl)}, weighted 1d {fmt_pct(avg_1d)}, 7d {fmt_pct(avg_7d)}, top {top}"
    return result(
        source,
        metric_key="defi_tvl_weighted_change_7d",
        score=score,
        calc_score=score,
        signal=score_signal(score),
        value=avg_7d,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )
