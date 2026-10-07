from __future__ import annotations

import math
import re
from typing import Any

from .common import (
    clamp,
    compact_usd,
    float_or_none,
    fmt_pct,
    nested_float,
    pct_change,
    result,
    score_signal,
)


def normalize_defillama_stablecoin_chains(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else []
    by_chain = {
        str(row.get("name") or "").strip().casefold(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    configured = source.get("chains")
    chains = (
        [str(chain).strip() for chain in configured if str(chain).strip()]
        if isinstance(configured, list)
        else []
    )
    if not chains:
        chains = [
            str(row.get("name") or "").strip()
            for row in rows[: int(source.get("max_chains", 20))]
            if isinstance(row, dict)
        ]
    output: list[dict[str, Any]] = []
    for chain in chains[: int(source.get("max_chains", 20))]:
        row = by_chain.get(chain.casefold(), {})
        components = row.get("totalCirculatingUSD")
        values = [float_or_none(value) for value in components.values()] if isinstance(components, dict) else []
        complete = bool(values) and all(value is not None and math.isfinite(value) and value >= 0 for value in values)
        amount = sum(values) if complete else None
        if not chain:
            continue
        valid = amount is not None and math.isfinite(amount) and amount >= 0
        chain_key = re.sub(r"[^a-z0-9]+", "_", chain.casefold()).strip("_") or "unknown"
        output.append(
            result(
                source,
                metric_key=f"stablecoin_supply_by_chain_usd_{chain_key}",
                score=None,
                signal="",
                value=amount if valid else None,
                unit="usd",
                notes=(
                    f"Current stablecoin supply USD value for chain {chain}, summed across provider totalCirculatingUSD peg components; captured at collector fetch time because this endpoint has no per-chain observation clock. Supply level is not a net capital-flow measure."
                    if valid
                    else f"Unknown: malformed stablecoin supply for chain {chain}."
                ),
                raw=row if store_raw else None,
            )
        )
    if not output:
        raise ValueError("DeFiLlama stablecoin-chain response contained no valid chain rows")
    return output


def normalize_defillama_dex_volume(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    envelope = payload if isinstance(payload, dict) else {}
    protocols = envelope.get("protocols")
    if not isinstance(protocols, list):
        raise ValueError("Malformed DeFiLlama DEX-volume response")
    # Provider totals avoid re-summing overlapping parent/child or double-counted protocols.
    total_24h = float_or_none(envelope.get("total24h"))
    previous_24h = float_or_none(envelope.get("total48hto24h"))
    complete = total_24h is not None and math.isfinite(total_24h) and total_24h >= 0
    previous_valid = previous_24h is not None and math.isfinite(previous_24h) and previous_24h > 0
    change = pct_change(total_24h, previous_24h) if complete and previous_valid else None
    if change is not None and not math.isfinite(change):
        change = None
    notes = (
        "Provider aggregate DEX volume (not re-summed protocol rows); this activity measure is not a direct capital-flow measure. "
        "The provider response has no shared observation clock, so collected-at is availability."
        if complete
        else "Unknown: DEX volume coverage is incomplete or malformed."
    )
    return [
        result(
            source,
            metric_key="dex_volume_24h_usd",
            score=None,
            signal="",
            value=total_24h if complete else None,
            unit="usd",
            notes=notes,
            raw=payload if store_raw else None,
        ),
        result(
            source,
            metric_key="dex_volume_change_24h_pct",
            score=None,
            signal="",
            value=change,
            unit="percent",
            notes=(
                "Change versus the preceding 24h DEX-volume bucket; descriptive activity, not net capital flow. "
                "The provider response has no shared observation clock, so collected-at is availability."
                if change is not None
                else "Unknown: missing or incomplete matched 24h DEX-volume buckets."
            ),
            raw=None,
        ),
    ]


def normalize_defillama_stablecoins(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
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
    top = ", ".join(
        f"{name} ${compact_usd(value)}"
        for name, value in sorted(top_assets, key=lambda item: item[1], reverse=True)[:3]
    )
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
        day_score = clamp(50.0 + change_1d * 15.0)
        rows.append(
            result(
                source,
                metric_key="stablecoin_supply_change_1d",
                score=day_score,
                calc_score=day_score,
                signal=score_signal(day_score),
                value=change_1d,
                unit="percent",
                notes="One-day aggregate stablecoin supply change from DeFiLlama.",
                raw=None,
            )
        )
    return rows


def normalize_defillama_chains(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> dict[str, Any]:
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
    top = ", ".join(
        f"{name} ${compact_usd(value)}"
        for name, value in sorted(top_chains, key=lambda item: item[1], reverse=True)[:3]
    )
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
