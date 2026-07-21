from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import blake2b
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - optional fallback for standalone notebooks
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]

LineSide = Literal["resistance", "support"]

_STATE_ATTR = "_trendline_projection_v2_state"
_CONSTRUCTION_FIELDS = (
    "timeframe",
    "pivot_strength",
    "min_anchor_bars",
    "max_anchor_bars",
    "min_pivot_prominence_atr",
    "max_slope_atr_per_bar",
    "max_projection_bars",
    "absorb_width_scale",
    "duplicate_line_proximity_atr_mult",
    "anchor_break_edge_bars",
    "anchor_break_max_run",
    "anchor_break_max_atr_mult",
    "projection_break_candles",
    "projection_break_atr_mult",
    "join_angle_tolerance",
    "max_joined_pivots",
    "family_touch_tolerance_atr_mult",
    "family_fit_tolerance_atr_mult",
    "family_seed_pivot_lookback",
    "max_active_seeds_per_side",
    "max_active_families_per_side",
)

_PROFILE_FIELDS = (
    "max_anchor_bars",
    "max_projection_bars",
    "max_active_line_distance_atr_mult",
    "max_slope_atr_per_bar",
    "min_output_active_bars",
)
_TIMEFRAME_PROFILES: dict[str, dict[str, object]] = {
    "1h": {
        "max_anchor_bars": 50,
        "max_projection_bars": 50,
        "max_active_line_distance_atr_mult": 2.5,
        "max_slope_atr_per_bar": 0.35,
        "min_output_active_bars": 1,
    },
    "4h": {
        "max_anchor_bars": 50,
        "max_projection_bars": 50,
        "max_active_line_distance_atr_mult": 2.5,
        "max_slope_atr_per_bar": 0.30,
        "min_output_active_bars": 1,
    },
    "8h": {
        "max_anchor_bars": 80,
        "max_projection_bars": 80,
        "max_active_line_distance_atr_mult": 2.5,
        "max_slope_atr_per_bar": 0.18,
        "min_output_active_bars": 1,
    },
    "1d": {
        "max_anchor_bars": 80,
        "max_projection_bars": 90,
        "max_active_line_distance_atr_mult": 2.5,
        "max_slope_atr_per_bar": 0.10,
        "min_output_active_bars": 1,
    },
    "3d": {
        "max_anchor_bars": 90,
        "max_projection_bars": 110,
        "max_active_line_distance_atr_mult": 2.5,
        "max_slope_atr_per_bar": 0.035,
        "min_output_active_bars": 1,
    },
}
_TIMEFRAME_ALIASES = {
    "1h": "1h",
    "60m": "1h",
    "4h": "4h",
    "240m": "4h",
    "8h": "8h",
    "480m": "8h",
    "1d": "1d",
    "1day": "1d",
    "3d": "3d",
    "3day": "3d",
}


@dataclass(frozen=True)
class TrendlineProjectionV2Config:
    """Causal proximity-family trendline generator.

    Confirmed pivot pairs are bounded internal seeds. A third nearby confirmed
    pivot promotes a seed cluster to a three-to-six-pivot least-squares family.
    Later nearby pivots create new causal versions from their confirmation row;
    they never refit earlier output. Pair outputs are separately named and
    opt-in, while the canonical candidate table contains confirmed families.

    Principal levers:
    - ``pivot_strength``: confirmed body pivot strength. A pivot is emitted only
      after this many candles have confirmed it.
    - ``max_slope_atr_per_bar``: removes extreme angle lines using the local
      ATR/body scale over the p1->p2 span. The value means "price units moved
      per candle, divided by the local market scale."
    - ``min_anchor_bars``: removes pivot pairs that are too close together.
    - ``max_anchor_bars``: removes seed lines whose p1->p2 span is too long to
      be a clean initial trendline definition.
    - ``max_projection_bars``: caps a seed/family version after the candle on
      which it becomes knowable. A later confirmed touch starts a new version.
    - ``max_active_line_distance_atr_mult``: suppresses projected lines once price has
      moved too far away for the line to be useful on the active timeframe.
    - ``family_touch_tolerance_atr_mult`` and
      ``family_fit_tolerance_atr_mult``: admit rough human-style touches, then
      require all retained explicit pivots to remain close to the best-fit line.
    - ``family_seed_pivot_lookback``, ``max_active_seeds_per_side`` and
      ``max_active_families_per_side``: hard bounds that keep runtime linear in
      candle count rather than candidate count squared.
    - ``anchor_break_*``: nuanced p1->p2 body-break handling. Minor body breaks
      near either anchor, or short/shallow interior breaks, can be allowed so
      valid human-looking lines are not deleted by pivot granularity.
    - ``projection_break_*``: require consecutive closes beyond an ATR-scaled
      tolerance. The break-confirmation candle remains visible; later candles
      do not revise earlier output.
    - ``min_output_*``: direct-output quality controls for the ranked slots.
    - ``proximity_rank_weight``: strategy-facing ranking tilt. Candidate
      quality remains the main gate, but a positive value lets nearer lines
      outrank equally credible distant lines. A useful first strategy test is
      "score above X and distance_atr below Y" rather than treating TLV2 as a
      direct long/short signal.
    - ``min_candidate_line_score``: optional upstream candidate-table floor for
      consumers such as geometry v2 that already reject weak source lines. Keep
      it lower than or equal to downstream pair-level score gates.

    Strategy-facing outputs:
    - ``*_resistance_line_rankN`` / ``*_support_line_rankN``: compact ranked
      confirmed trendline slots with at least ``confirmed_min_pivots`` explicit
      pivots. Channels and higher-level patterns are separate consumers.
    - ``*_provisional_*``: optional two-anchor hypotheses for other indicators.
      They never compete with the confirmed ranks.
    - ``*_score_rankN`` is the underlying line quality score, while
      ``*_distance_atr_rankN`` is the current candle's distance from the line in
      ATR units. Strategies can hyperopt these together, for example by using
      nearby high-score support as a guard or using nearby high-score resistance
      as a take-profit/avoid-long context.
    """

    output_prefix: str = "tlv2"
    timeframe: str = "4h"
    pivot_strength: int = 2
    raw_line_output_count: int = 3
    include_diagnostics: bool = False
    confirmed_min_pivots: int = 3
    include_provisional_pair_outputs: bool = False
    provisional_line_output_count: int = 1

    min_anchor_bars: int = 10
    max_anchor_bars: int = 50
    min_pivot_prominence_atr: float = 0.35
    max_slope_atr_per_bar: float = 0.35
    max_projection_bars: int = 50
    max_active_line_distance_atr_mult: float = 2.5

    absorb_width_scale: float = 0.30
    duplicate_line_proximity_atr_mult: float = 0.50
    anchor_break_edge_bars: int = 3
    anchor_break_max_run: int = 2
    anchor_break_max_atr_mult: float = 0.40
    projection_break_candles: int = 2
    projection_break_atr_mult: float = 0.35

    join_angle_tolerance: float = 0.15
    max_joined_pivots: int = 6
    family_touch_tolerance_atr_mult: float = 0.60
    family_fit_tolerance_atr_mult: float = 0.55
    family_seed_pivot_lookback: int = 12
    max_active_seeds_per_side: int = 24
    max_active_families_per_side: int = 6
    min_candidate_line_score: float = 0.0
    min_output_line_score: float = 0.50
    min_output_active_bars: int = 1
    proximity_rank_weight: float = 0.05
    proximity_rank_distance_cap_atr: float = 6.0


@dataclass(frozen=True, eq=False)
class TrendlineProjectionV2State:
    """One causal TLV2 candidate build reusable by ranked lines and geometry."""

    config: TrendlineProjectionV2Config
    frame_signature: str
    base: dict[str, Series]
    candidates: DataFrame
    provisional_candidates: DataFrame

    def matches(self, dataframe: DataFrame, config: TrendlineProjectionV2Config) -> bool:
        return self.frame_signature == _frame_signature(dataframe) and _construction_signature(
            self.config
        ) == _construction_signature(config)

    def attach_to(self, dataframe: DataFrame) -> None:
        dataframe.attrs[_STATE_ATTR] = self


def build_trendline_projection_v2_state(
    dataframe: DataFrame,
    config: TrendlineProjectionV2Config | None = None,
    **overrides: object,
) -> TrendlineProjectionV2State:
    """Build the canonical unranked TLV2 state once for downstream consumers."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)
    return _build_state(dataframe, cfg)


def resolve_trendline_projection_v2_state(
    dataframe: DataFrame,
    config: TrendlineProjectionV2Config | None = None,
    state: TrendlineProjectionV2State | None = None,
    **overrides: object,
) -> TrendlineProjectionV2State:
    """Reuse a compatible explicit/attached state, otherwise build it once."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)
    candidate = state
    if candidate is None:
        attached = dataframe.attrs.get(_STATE_ATTR)
        if isinstance(attached, TrendlineProjectionV2State):
            candidate = attached
    if candidate is not None:
        if not candidate.matches(dataframe, cfg):
            if state is not None:
                raise ValueError(
                    "TrendlineProjectionV2State does not match the dataframe or construction config"
                )
        else:
            return candidate
    return _build_state(dataframe, cfg)


def add_trendline_projection_v2(
    dataframe: DataFrame,
    config: TrendlineProjectionV2Config | None = None,
    *,
    state: TrendlineProjectionV2State | None = None,
    **overrides: object,
) -> DataFrame:
    """Append confirmed trendlines and optional provisional pair lines."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    tlv2_state = resolve_trendline_projection_v2_state(frame, cfg, state=state)
    base = tlv2_state.base
    p = cfg.output_prefix

    new_cols: dict[str, Series] = {}
    if bool(cfg.include_diagnostics):
        new_cols.update(
            {
                f"{p}_atr": base["atr"],
                f"{p}_line_scale": base["line_scale"],
                f"{p}_body_high": base["body_high"],
                f"{p}_body_low": base["body_low"],
                f"{p}_pivot_high": base["pivot_high"],
                f"{p}_pivot_low": base["pivot_low"],
                f"{p}_pivot_high_index": base["pivot_high_index"],
                f"{p}_pivot_low_index": base["pivot_low_index"],
                f"{p}_pivot_high_available_index": base["pivot_high_available_index"],
                f"{p}_pivot_low_available_index": base["pivot_low_available_index"],
                f"{p}_pivot_high_prominence": base["pivot_high_prominence"],
                f"{p}_pivot_low_prominence": base["pivot_low_prominence"],
                f"{p}_pivot_high_score": base["pivot_high_score"],
                f"{p}_pivot_low_score": base["pivot_low_score"],
            }
        )
    sequence_candidates = _filter_candidate_score(
        tlv2_state.candidates,
        float(cfg.min_candidate_line_score),
    )
    raw_resistance_cols = _rank_sequence_candidate_columns(
        sequence_candidates,
        base,
        "resistance",
        "resistance",
        cfg,
        slots=int(cfg.raw_line_output_count),
        min_explicit_pivots=int(cfg.confirmed_min_pivots),
    )
    raw_support_cols = _rank_sequence_candidate_columns(
        sequence_candidates,
        base,
        "support",
        "support",
        cfg,
        slots=int(cfg.raw_line_output_count),
        min_explicit_pivots=int(cfg.confirmed_min_pivots),
    )
    new_cols.update(raw_resistance_cols)
    new_cols.update(raw_support_cols)
    if bool(cfg.include_provisional_pair_outputs):
        provisional_prefix = f"{p}_provisional"
        provisional_candidates = _filter_candidate_score(
            tlv2_state.provisional_candidates,
            float(cfg.min_candidate_line_score),
        )
        new_cols.update(
            _rank_sequence_candidate_columns(
                provisional_candidates,
                base,
                "resistance",
                "resistance",
                cfg,
                slots=int(cfg.provisional_line_output_count),
                output_prefix=provisional_prefix,
                min_explicit_pivots=2,
                max_explicit_pivots=2,
            )
        )
        new_cols.update(
            _rank_sequence_candidate_columns(
                provisional_candidates,
                base,
                "support",
                "support",
                cfg,
                slots=int(cfg.provisional_line_output_count),
                output_prefix=provisional_prefix,
                min_explicit_pivots=2,
                max_explicit_pivots=2,
            )
        )
    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    clean = frame.drop(columns=existing).copy() if existing else frame.copy()
    result = pd.concat([clean, pd.DataFrame(new_cols, index=frame.index)], axis=1)
    tlv2_state.attach_to(result)
    return result


def _build_state(
    dataframe: DataFrame, cfg: TrendlineProjectionV2Config
) -> TrendlineProjectionV2State:
    frame = dataframe.copy()
    base = _base_inputs(frame, cfg)
    candidate_cfg = replace(cfg, min_candidate_line_score=0.0)
    candidates, provisional_candidates = _build_chronological_family_candidate_tables(
        base, candidate_cfg
    )
    return TrendlineProjectionV2State(
        config=cfg,
        frame_signature=_frame_signature(dataframe),
        base=base,
        candidates=candidates,
        provisional_candidates=provisional_candidates,
    )


def _frame_signature(frame: DataFrame) -> str:
    values = pd.util.hash_pandas_object(
        frame.loc[:, ["open", "high", "low", "close"]],
        index=True,
    ).to_numpy(dtype="uint64", copy=False)
    return blake2b(values.tobytes(), digest_size=16).hexdigest()


def _construction_signature(cfg: TrendlineProjectionV2Config) -> tuple[object, ...]:
    return tuple(getattr(cfg, field) for field in _CONSTRUCTION_FIELDS)


def _base_inputs(frame: DataFrame, cfg: TrendlineProjectionV2Config) -> dict[str, Series]:
    open_ = _num(frame, "open")
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(frame, 14)
    body_height = (body_high - body_low).abs()
    body_scale = body_height.rolling(14, min_periods=3).mean()
    line_scale = pd.concat([atr, body_scale], axis=1).max(axis=1).replace(0.0, np.nan)
    bar_index = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    pivots = build_clean_pivot_source(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        bar_index=bar_index,
        strength=int(cfg.pivot_strength),
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "body_high": body_high,
        "body_low": body_low,
        "atr": atr,
        "line_scale": line_scale,
        "bar_index": bar_index,
        **pivots,
    }


@dataclass(frozen=True)
class _PivotPoint:
    anchor_index: int
    available_index: int
    price: float
    prominence: float
    quality: float
    scale: float


@dataclass(frozen=True)
class _LineFit:
    slope: float
    intercept: float
    line_scale: float
    slope_atr_per_bar: float
    max_residual: float
    max_residual_atr: float


@dataclass
class _ProvisionalSeed:
    side: LineSide
    points: tuple[_PivotPoint, _PivotPoint]
    slope: float
    intercept: float
    line_scale: float
    slope_atr_per_bar: float
    score: float
    live_start: int
    projection_cap: int
    record: dict[str, object]
    break_run: int = 0


@dataclass
class _TrendlineFamily:
    family_id: str
    side: LineSide
    points: list[_PivotPoint]
    slope: float
    intercept: float
    line_scale: float
    slope_atr_per_bar: float
    score: float
    total_touch_count: int
    live_start: int
    projection_cap: int
    version_record: dict[str, object]
    break_run: int = 0


def _build_chronological_family_candidate_tables(
    base: dict[str, Series],
    cfg: TrendlineProjectionV2Config,
) -> tuple[DataFrame, DataFrame]:
    """Build bounded provisional seeds and causal three-plus-pivot families.

    A pair is only an internal hypothesis.  A third confirmed pivot must land
    near it before a strategy-facing family exists.  Later nearby pivots update
    that family from their own confirmation rows; earlier versions are never
    refitted or backfilled.
    """

    family_records: list[dict[str, object]] = []
    provisional_records: list[dict[str, object]] = []
    for side in ("resistance", "support"):
        side_families, side_provisional = _chronological_side_candidates(base, side, cfg)
        family_records.extend(side_families)
        provisional_records.extend(side_provisional)
    return _candidate_records_frame(family_records), _candidate_records_frame(provisional_records)


def _chronological_side_candidates(
    base: dict[str, Series],
    side: LineSide,
    cfg: TrendlineProjectionV2Config,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = len(base["close"])
    points = _confirmed_pivot_points(base, side, cfg)
    events: dict[int, list[_PivotPoint]] = {}
    for point in points:
        if 0 <= point.available_index < rows:
            events.setdefault(point.available_index, []).append(point)

    family_records: list[dict[str, object]] = []
    provisional_records: list[dict[str, object]] = []
    active_seeds: list[_ProvisionalSeed] = []
    active_families: list[_TrendlineFamily] = []
    known_points: list[_PivotPoint] = []

    # Candidate birth, confirmation, refit, and break state depend on the prior
    # candle's state, so this event sweep cannot be vectorized without first
    # constructing future-aware histories. Per-row work remains hard-bounded by
    # the active seed/family limits above.
    for row in range(rows):
        active_seeds = [seed for seed in active_seeds if row <= seed.projection_cap]
        active_families = [family for family in active_families if row <= family.projection_cap]

        for point in events.get(row, []):
            updated = _update_nearest_family(active_families, point, row, family_records, cfg)
            if not updated:
                family, used_seeds = _confirm_family_from_seeds(active_seeds, point, row, cfg)
                if family is not None:
                    for seed in used_seeds:
                        _end_candidate_before(seed.record, row)
                    used_ids = {id(seed) for seed in used_seeds}
                    active_seeds = [seed for seed in active_seeds if id(seed) not in used_ids]
                    active_families.append(family)
                    family_records.append(family.version_record)

            new_seeds = _new_seeds_for_point(base, side, known_points, point, row, cfg)
            for seed in new_seeds:
                provisional_records.append(seed.record)
            active_seeds.extend(new_seeds)
            known_points.append(point)

            active_seeds = _limit_active_seeds(active_seeds, row, cfg)
            active_families = _limit_active_families(active_families, row, cfg)

        active_seeds = _advance_seed_breaks(active_seeds, base, row, cfg)
        active_families = _advance_family_breaks(active_families, base, row, cfg)

    return family_records, provisional_records


def _confirmed_pivot_points(
    base: dict[str, Series],
    side: LineSide,
    cfg: TrendlineProjectionV2Config,
) -> list[_PivotPoint]:
    stem = "high" if side == "resistance" else "low"
    price = base[f"pivot_{stem}"].to_numpy(dtype="float64")
    anchor = base[f"pivot_{stem}_index"].to_numpy(dtype="float64")
    available = base[f"pivot_{stem}_available_index"].to_numpy(dtype="float64")
    prominence = base[f"pivot_{stem}_prominence"].to_numpy(dtype="float64")
    quality = base[f"pivot_{stem}_score"].to_numpy(dtype="float64")
    scale_values = base["line_scale"].to_numpy(dtype="float64")
    valid_rows = np.flatnonzero(
        np.isfinite(price)
        & np.isfinite(anchor)
        & np.isfinite(available)
        & np.isfinite(prominence)
        & (prominence >= float(cfg.min_pivot_prominence_atr))
    )
    points: list[_PivotPoint] = []
    for event_row in valid_rows:
        anchor_index = round(float(anchor[event_row]))
        available_index = max(int(event_row), round(float(available[event_row])))
        scale_index = min(max(anchor_index, 0), len(scale_values) - 1)
        scale = float(scale_values[scale_index]) if len(scale_values) else np.nan
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(scale_values[event_row]) if np.isfinite(scale_values[event_row]) else 1e-9
        points.append(
            _PivotPoint(
                anchor_index=anchor_index,
                available_index=available_index,
                price=float(price[event_row]),
                prominence=float(prominence[event_row]),
                quality=float(quality[event_row]) if np.isfinite(quality[event_row]) else 0.0,
                scale=max(float(scale), 1e-9),
            )
        )
    return sorted(points, key=lambda point: (point.available_index, point.anchor_index))


def _new_seeds_for_point(
    base: dict[str, Series],
    side: LineSide,
    known_points: list[_PivotPoint],
    point: _PivotPoint,
    row: int,
    cfg: TrendlineProjectionV2Config,
) -> list[_ProvisionalSeed]:
    lookback = max(int(cfg.family_seed_pivot_lookback), 1)
    candidates: list[_ProvisionalSeed] = []
    for old in reversed(known_points[-lookback:]):
        span = point.anchor_index - old.anchor_index
        if span < int(cfg.min_anchor_bars) or span > int(cfg.max_anchor_bars):
            continue
        line_scale = _span_market_scale(base, float(old.anchor_index), float(point.anchor_index))
        slope = (point.price - old.price) / float(span)
        slope_atr = abs(slope) / max(line_scale, 1e-9)
        if not np.isfinite(slope_atr) or slope_atr > float(cfg.max_slope_atr_per_bar):
            continue
        intercept = point.price - slope * float(point.anchor_index)
        if not _anchor_span_is_clear(
            base,
            side,
            old.anchor_index,
            point.anchor_index,
            float(slope),
            float(intercept),
            cfg,
        ):
            continue
        score = _seed_quality(old, point, slope_atr, cfg)
        cap = row + int(cfg.max_projection_bars)
        record = _candidate_record(
            side=side,
            points=[old, point],
            slope=float(slope),
            intercept=float(intercept),
            line_scale=float(line_scale),
            slope_atr_per_bar=float(slope_atr),
            score=float(score),
            live_start=row,
            projection_cap=cap,
            total_touch_count=2,
            half_width=0.0,
            line_kind="provisional",
            family_id=f"{side}|seed|{old.anchor_index}|{point.anchor_index}",
        )
        candidates.append(
            _ProvisionalSeed(
                side=side,
                points=(old, point),
                slope=float(slope),
                intercept=float(intercept),
                line_scale=float(line_scale),
                slope_atr_per_bar=float(slope_atr),
                score=float(score),
                live_start=row,
                projection_cap=cap,
                record=record,
            )
        )
    return _cluster_same_event_seeds(candidates, point, cfg)


def _cluster_same_event_seeds(
    seeds: list[_ProvisionalSeed],
    point: _PivotPoint,
    cfg: TrendlineProjectionV2Config,
) -> list[_ProvisionalSeed]:
    kept: list[_ProvisionalSeed] = []
    for seed in sorted(
        seeds,
        key=lambda item: (item.score, item.points[1].anchor_index - item.points[0].anchor_index),
        reverse=True,
    ):
        duplicate = next(
            (other for other in kept if _seeds_share_family(seed, other, point, cfg)), None
        )
        if duplicate is None:
            kept.append(seed)
            continue
        duplicate.record["absorbed_line_count"] = (
            float(duplicate.record.get("absorbed_line_count", 0.0)) + 1.0
        )
        absorbed = {pivot.anchor_index for pivot in (*duplicate.points, *seed.points)}
        duplicate.record["absorbed_pivot_count"] = float(len(absorbed))
        distance = _max_point_distance_to_line(seed.points, duplicate.slope, duplicate.intercept)
        duplicate.record["half_width"] = max(
            float(duplicate.record.get("half_width", 0.0)),
            float(distance) * float(cfg.absorb_width_scale),
        )
    return kept


def _confirm_family_from_seeds(
    seeds: list[_ProvisionalSeed],
    point: _PivotPoint,
    row: int,
    cfg: TrendlineProjectionV2Config,
) -> tuple[_TrendlineFamily | None, list[_ProvisionalSeed]]:
    matches: list[tuple[float, _ProvisionalSeed]] = []
    for seed in seeds:
        if point.anchor_index <= seed.points[-1].anchor_index or row > seed.projection_cap:
            continue
        distance_atr = abs(point.price - (seed.intercept + seed.slope * point.anchor_index)) / max(
            point.scale, 1e-9
        )
        if distance_atr > float(cfg.family_touch_tolerance_atr_mult):
            continue
        fit = _fit_family_points([*seed.points, point], cfg)
        if fit is not None:
            matches.append((float(distance_atr), seed))
    if not matches:
        return None, []

    matches.sort(
        key=lambda item: (
            item[0],
            -item[1].score,
            -(item[1].points[-1].anchor_index - item[1].points[0].anchor_index),
        )
    )
    best = matches[0][1]
    cluster = [seed for _, seed in matches if _seeds_share_family(seed, best, point, cfg)]
    family_points: list[_PivotPoint] = [*best.points, point]
    for seed in cluster:
        for candidate_point in seed.points:
            if any(
                existing.anchor_index == candidate_point.anchor_index for existing in family_points
            ):
                continue
            proposed = _bounded_family_points(
                [*family_points, candidate_point], int(cfg.max_joined_pivots)
            )
            if _fit_family_points(proposed, cfg) is not None:
                family_points = proposed
    family_points = sorted(family_points, key=lambda item: item.anchor_index)
    fit = _fit_family_points(family_points, cfg)
    if fit is None:
        return None, []

    total_touches = len(
        {pivot.anchor_index for seed in cluster for pivot in seed.points} | {point.anchor_index}
    )
    score = _family_quality(family_points, fit, total_touches, cfg)
    cap = row + int(cfg.max_projection_bars)
    family_id = f"{best.side}|family|{best.points[0].anchor_index}|{best.points[1].anchor_index}"
    record = _candidate_record(
        side=best.side,
        points=family_points,
        slope=fit.slope,
        intercept=fit.intercept,
        line_scale=fit.line_scale,
        slope_atr_per_bar=fit.slope_atr_per_bar,
        score=score,
        live_start=row,
        projection_cap=cap,
        total_touch_count=total_touches,
        half_width=fit.max_residual * float(cfg.absorb_width_scale),
        line_kind="family",
        family_id=family_id,
    )
    return (
        _TrendlineFamily(
            family_id=family_id,
            side=best.side,
            points=family_points,
            slope=fit.slope,
            intercept=fit.intercept,
            line_scale=fit.line_scale,
            slope_atr_per_bar=fit.slope_atr_per_bar,
            score=score,
            total_touch_count=total_touches,
            live_start=row,
            projection_cap=cap,
            version_record=record,
        ),
        cluster,
    )


def _update_nearest_family(
    families: list[_TrendlineFamily],
    point: _PivotPoint,
    row: int,
    records: list[dict[str, object]],
    cfg: TrendlineProjectionV2Config,
) -> bool:
    matches: list[tuple[float, float, _TrendlineFamily, list[_PivotPoint], _LineFit]] = []
    for family in families:
        if point.anchor_index <= family.points[-1].anchor_index or row > family.projection_cap:
            continue
        distance_atr = abs(
            point.price - (family.intercept + family.slope * point.anchor_index)
        ) / max(point.scale, 1e-9)
        if distance_atr > float(cfg.family_touch_tolerance_atr_mult):
            continue
        proposed = _bounded_family_points([*family.points, point], int(cfg.max_joined_pivots))
        fit = _fit_family_points(proposed, cfg)
        if fit is None:
            continue
        matches.append((float(distance_atr), -float(family.score), family, proposed, fit))
    if not matches:
        return False

    _, _, family, proposed, fit = min(matches, key=lambda item: (item[0], item[1]))
    _end_candidate_before(family.version_record, row)
    family.points = proposed
    family.slope = fit.slope
    family.intercept = fit.intercept
    family.line_scale = fit.line_scale
    family.slope_atr_per_bar = fit.slope_atr_per_bar
    family.total_touch_count += 1
    family.score = _family_quality(proposed, fit, family.total_touch_count, cfg)
    family.live_start = row
    family.projection_cap = row + int(cfg.max_projection_bars)
    family.break_run = 0
    family.version_record = _candidate_record(
        side=family.side,
        points=family.points,
        slope=family.slope,
        intercept=family.intercept,
        line_scale=family.line_scale,
        slope_atr_per_bar=family.slope_atr_per_bar,
        score=family.score,
        live_start=row,
        projection_cap=family.projection_cap,
        total_touch_count=family.total_touch_count,
        half_width=fit.max_residual * float(cfg.absorb_width_scale),
        line_kind="family_update",
        family_id=family.family_id,
    )
    records.append(family.version_record)
    return True


def _fit_family_points(
    points: list[_PivotPoint], cfg: TrendlineProjectionV2Config
) -> _LineFit | None:
    if len(points) < 3:
        return None
    ordered = sorted(points, key=lambda point: point.anchor_index)
    x = np.asarray([point.anchor_index for point in ordered], dtype="float64")
    y = np.asarray([point.price for point in ordered], dtype="float64")
    scales = np.asarray([point.scale for point in ordered], dtype="float64")
    x_center = x - float(x.mean())
    denominator = float(x_center.dot(x_center))
    if denominator <= 0.0:
        return None
    slope = float(x_center.dot(y - float(y.mean())) / denominator)
    intercept = float(y.mean() - slope * x.mean())
    residuals = np.abs(y - (intercept + slope * x))
    residual_atr = residuals / np.maximum(scales, 1e-9)
    max_residual_atr = float(np.max(residual_atr))
    if not np.isfinite(max_residual_atr) or max_residual_atr > float(
        cfg.family_fit_tolerance_atr_mult
    ):
        return None
    line_scale = float(np.median(scales[np.isfinite(scales) & (scales > 0.0)]))
    if not np.isfinite(line_scale) or line_scale <= 0.0:
        return None
    slope_atr = abs(slope) / line_scale
    if not np.isfinite(slope_atr) or slope_atr > float(cfg.max_slope_atr_per_bar):
        return None
    return _LineFit(
        slope=slope,
        intercept=intercept,
        line_scale=line_scale,
        slope_atr_per_bar=float(slope_atr),
        max_residual=float(np.max(residuals)),
        max_residual_atr=max_residual_atr,
    )


def _bounded_family_points(points: list[_PivotPoint], maximum: int) -> list[_PivotPoint]:
    unique = {point.anchor_index: point for point in points}
    ordered = sorted(unique.values(), key=lambda point: point.anchor_index)
    limit = max(int(maximum), 3)
    if len(ordered) <= limit:
        return ordered
    return [ordered[0], *ordered[-(limit - 1) :]]


def _seed_quality(
    old: _PivotPoint, new: _PivotPoint, slope_atr: float, cfg: TrendlineProjectionV2Config
) -> float:
    span = float(new.anchor_index - old.anchor_index)
    span_score = min(span / max(float(cfg.min_anchor_bars) * 5.0, 1.0), 1.0)
    slope_score = max(0.0, 1.0 - slope_atr / max(float(cfg.max_slope_atr_per_bar), 1e-9))
    prominence = (old.prominence + new.prominence) / 2.0
    prominence_score = min(prominence / max(float(cfg.min_pivot_prominence_atr) * 5.0, 1.0), 1.0)
    quality = (old.quality + new.quality) / 2.0
    return float(
        np.clip(
            0.38 * span_score + 0.24 * slope_score + 0.26 * prominence_score + 0.12 * quality,
            0.0,
            1.0,
        )
    )


def _family_quality(
    points: list[_PivotPoint],
    fit: _LineFit,
    total_touch_count: int,
    cfg: TrendlineProjectionV2Config,
) -> float:
    span = float(points[-1].anchor_index - points[0].anchor_index)
    span_score = min(span / max(float(cfg.min_anchor_bars) * 5.0, 1.0), 1.0)
    slope_score = max(
        0.0, 1.0 - fit.slope_atr_per_bar / max(float(cfg.max_slope_atr_per_bar), 1e-9)
    )
    prominence = float(np.mean([point.prominence for point in points]))
    prominence_score = min(prominence / max(float(cfg.min_pivot_prominence_atr) * 5.0, 1.0), 1.0)
    pivot_quality = float(np.mean([point.quality for point in points]))
    fit_score = max(
        0.0, 1.0 - fit.max_residual_atr / max(float(cfg.family_fit_tolerance_atr_mult), 1e-9)
    )
    touch_score = min(
        max(float(total_touch_count - 2), 0.0) / max(float(cfg.max_joined_pivots - 2), 1.0), 1.0
    )
    return float(
        np.clip(
            0.25 * span_score
            + 0.18 * slope_score
            + 0.20 * prominence_score
            + 0.12 * pivot_quality
            + 0.20 * fit_score
            + 0.05 * touch_score,
            0.0,
            1.0,
        )
    )


def _candidate_record(
    *,
    side: LineSide,
    points: list[_PivotPoint],
    slope: float,
    intercept: float,
    line_scale: float,
    slope_atr_per_bar: float,
    score: float,
    live_start: int,
    projection_cap: int,
    total_touch_count: int,
    half_width: float,
    line_kind: str,
    family_id: str,
) -> dict[str, object]:
    ordered = sorted(points, key=lambda point: point.anchor_index)
    first = ordered[0]
    last = ordered[-1]
    return {
        "side": side,
        "x_old": float(first.anchor_index),
        "x_new": float(last.anchor_index),
        "x_end": float(last.anchor_index),
        "y_old": float(first.price),
        "y_new": float(last.price),
        "y_end": float(last.price),
        "slope": float(slope),
        "intercept": float(intercept),
        "projection_end": float(projection_cap),
        "projection_cap": float(projection_cap),
        "live_start": float(live_start),
        "span": float(last.anchor_index - first.anchor_index),
        "slope_atr_per_bar": float(slope_atr_per_bar),
        "line_scale": float(line_scale),
        "prominence": float(np.mean([point.prominence for point in ordered])),
        "pivot_quality": float(np.mean([point.quality for point in ordered])),
        "score": float(score),
        "pivot_count": float(len(ordered)),
        "pivot_path": "|".join(str(point.anchor_index) for point in ordered),
        "price_path": "|".join(str(float(point.price)) for point in ordered),
        "source_count": float(max(len(ordered) - 1, 1)),
        "absorbed_pivot_count": float(max(total_touch_count, len(ordered))),
        "absorbed_line_count": float(max(total_touch_count - len(ordered), 0)),
        "half_width": float(max(half_width, 0.0)),
        "line_kind": line_kind,
        "family_id": family_id,
    }


def _candidate_records_frame(records: list[dict[str, object]]) -> DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    live_start = pd.to_numeric(frame["live_start"], errors="coerce")
    projection_end = pd.to_numeric(frame["projection_end"], errors="coerce")
    frame = frame[live_start.notna() & projection_end.ge(live_start)].copy()
    if frame.empty:
        return frame.reset_index(drop=True)
    return frame.sort_values(
        ["side", "live_start", "pivot_count", "score", "span"],
        ascending=[True, True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _seeds_share_family(
    left: _ProvisionalSeed,
    right: _ProvisionalSeed,
    point: _PivotPoint,
    cfg: TrendlineProjectionV2Config,
) -> bool:
    left_norm = left.slope / max(left.line_scale, 1e-9)
    right_norm = right.slope / max(right.line_scale, 1e-9)
    denominator = max(abs(left_norm), abs(right_norm), 0.02)
    if abs(left_norm - right_norm) / denominator > float(cfg.join_angle_tolerance):
        return False
    left_value = left.intercept + left.slope * point.anchor_index
    right_value = right.intercept + right.slope * point.anchor_index
    return abs(left_value - right_value) / max(point.scale, 1e-9) <= float(
        cfg.duplicate_line_proximity_atr_mult
    )


def _max_point_distance_to_line(
    points: tuple[_PivotPoint, ...], slope: float, intercept: float
) -> float:
    if not points:
        return 0.0
    return float(
        max(abs(point.price - (intercept + slope * point.anchor_index)) for point in points)
    )


def _end_candidate_before(record: dict[str, object], row: int) -> None:
    record["projection_end"] = float(min(float(record["projection_end"]), float(row - 1)))


def _limit_active_seeds(
    seeds: list[_ProvisionalSeed],
    row: int,
    cfg: TrendlineProjectionV2Config,
) -> list[_ProvisionalSeed]:
    limit = int(cfg.max_active_seeds_per_side)
    if len(seeds) <= limit:
        return seeds
    ranked = sorted(
        seeds,
        key=lambda seed: (
            seed.score,
            seed.points[-1].anchor_index - seed.points[0].anchor_index,
            seed.live_start,
        ),
        reverse=True,
    )
    kept = ranked[:limit]
    kept_ids = {id(seed) for seed in kept}
    for seed in seeds:
        if id(seed) not in kept_ids:
            _end_candidate_before(seed.record, row)
    return kept


def _limit_active_families(
    families: list[_TrendlineFamily],
    row: int,
    cfg: TrendlineProjectionV2Config,
) -> list[_TrendlineFamily]:
    limit = int(cfg.max_active_families_per_side)
    if len(families) <= limit:
        return families
    ranked = sorted(
        families,
        key=lambda family: (
            len(family.points),
            family.score,
            family.total_touch_count,
            family.live_start,
        ),
        reverse=True,
    )
    kept = ranked[:limit]
    kept_ids = {id(family) for family in kept}
    for family in families:
        if id(family) not in kept_ids:
            _end_candidate_before(family.version_record, row)
    return kept


def _advance_seed_breaks(
    seeds: list[_ProvisionalSeed],
    base: dict[str, Series],
    row: int,
    cfg: TrendlineProjectionV2Config,
) -> list[_ProvisionalSeed]:
    active: list[_ProvisionalSeed] = []
    for seed in seeds:
        seed.break_run = (
            seed.break_run + 1
            if _line_broken(base, seed.side, row, seed.slope, seed.intercept, cfg)
            else 0
        )
        if seed.break_run >= int(cfg.projection_break_candles):
            seed.record["projection_end"] = float(min(float(seed.record["projection_end"]), row))
        else:
            active.append(seed)
    return active


def _advance_family_breaks(
    families: list[_TrendlineFamily],
    base: dict[str, Series],
    row: int,
    cfg: TrendlineProjectionV2Config,
) -> list[_TrendlineFamily]:
    active: list[_TrendlineFamily] = []
    for family in families:
        family.break_run = (
            family.break_run + 1
            if _line_broken(base, family.side, row, family.slope, family.intercept, cfg)
            else 0
        )
        if family.break_run >= int(cfg.projection_break_candles):
            family.version_record["projection_end"] = float(
                min(float(family.version_record["projection_end"]), row)
            )
        else:
            active.append(family)
    return active


def _line_broken(
    base: dict[str, Series],
    side: LineSide,
    row: int,
    slope: float,
    intercept: float,
    cfg: TrendlineProjectionV2Config,
) -> bool:
    atr = float(base["atr"].iloc[row])
    if not np.isfinite(atr) or atr <= 0.0:
        return False
    line = float(intercept) + float(slope) * float(row)
    tolerance = atr * float(cfg.projection_break_atr_mult)
    close = float(base["close"].iloc[row])
    if not np.isfinite(close) or not np.isfinite(line):
        return False
    return (
        bool(close > line + tolerance) if side == "resistance" else bool(close < line - tolerance)
    )


def _filter_candidate_score(candidates: DataFrame, minimum: float) -> DataFrame:
    if candidates.empty or float(minimum) <= 0.0:
        return candidates.copy()
    score = pd.to_numeric(candidates["score"], errors="coerce").fillna(0.0)
    return candidates[score.ge(float(minimum))].reset_index(drop=True)


def _rank_sequence_candidate_columns(
    candidates: DataFrame,
    base: dict[str, Series],
    side: LineSide,
    output_side: str,
    cfg: TrendlineProjectionV2Config,
    *,
    slots: int,
    output_prefix: str | None = None,
    min_explicit_pivots: int = 2,
    max_explicit_pivots: int | None = None,
) -> dict[str, Series]:
    index = base["close"].index
    rows = len(index)
    prefix = cfg.output_prefix if output_prefix is None else str(output_prefix)
    if rows == 0 or candidates.empty:
        return _empty_ranked_line_columns(index, prefix, output_side, int(slots))

    side_candidates = candidates[candidates["side"].eq(side)].copy()
    if side_candidates.empty:
        return _empty_ranked_line_columns(index, prefix, output_side, int(slots))
    explicit_pivots = pd.to_numeric(side_candidates["pivot_count"], errors="coerce").fillna(0.0)
    pivot_mask = explicit_pivots.ge(float(min_explicit_pivots))
    if max_explicit_pivots is not None:
        pivot_mask &= explicit_pivots.le(float(max_explicit_pivots))
    side_candidates = side_candidates[pivot_mask].copy()
    if side_candidates.empty:
        return _empty_ranked_line_columns(index, prefix, output_side, int(slots))
    side_candidates = side_candidates[
        pd.to_numeric(side_candidates["score"], errors="coerce")
        .fillna(0.0)
        .ge(float(cfg.min_output_line_score))
    ].copy()
    if side_candidates.empty:
        return _empty_ranked_line_columns(index, prefix, output_side, int(slots))

    side_candidates = side_candidates.sort_values(
        ["pivot_count", "score", "span", "absorbed_pivot_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    top = _empty_top_line_arrays(rows, int(slots))
    x_values = np.arange(rows, dtype="float64")
    close_values = base["close"].to_numpy(dtype="float64")
    atr_values = base["atr"].to_numpy(dtype="float64")
    for _, candidate in side_candidates.iterrows():
        live_start = int(np.ceil(float(candidate.get("live_start", candidate["x_new"]))))
        end = int(np.floor(float(candidate["projection_end"])))
        start = max(live_start, 0)
        end = min(end, rows - 1)
        if end < start:
            continue
        xs = x_values[start : end + 1]
        line = float(candidate["intercept"]) + float(candidate["slope"]) * xs
        close_segment = close_values[start : end + 1]
        atr_segment = atr_values[start : end + 1]
        active_distance = atr_segment * float(cfg.max_active_line_distance_atr_mult)
        valid_line = (
            np.isfinite(line)
            & np.isfinite(active_distance)
            & (np.abs(line - close_segment) <= active_distance)
        )
        valid_line = _keep_long_true_runs(valid_line, int(cfg.min_output_active_bars))
        if not np.any(valid_line):
            continue
        distance_atr_segment = np.abs(line - close_segment) / np.maximum(atr_segment, 1e-9)
        # Ranking is allowed to prefer nearby rails, but the exported score
        # remains the raw line-quality score. Strategy agents should test
        # score and distance_atr separately, for example:
        # high score + low distance_atr as actionable support/resistance,
        # high score + high distance_atr as context only.
        proximity_penalty = float(cfg.proximity_rank_weight) * np.minimum(
            distance_atr_segment,
            float(cfg.proximity_rank_distance_cap_atr),
        )
        rank_score = np.where(valid_line, float(candidate["score"]) - proximity_penalty, -np.inf)
        segment_rows = end - start + 1
        line_id = _stable_line_id(candidate)
        metrics = {
            "line": np.where(valid_line, line, np.nan),
            "score": np.where(valid_line, float(candidate["score"]), np.nan),
            "distance_atr": np.where(valid_line, distance_atr_segment, np.nan),
            "slope": np.full(segment_rows, float(candidate["slope"]), dtype="float64"),
            "slope_atr_per_bar": np.full(
                segment_rows,
                float(candidate["slope_atr_per_bar"]),
                dtype="float64",
            ),
            "anchor_old_index": np.full(segment_rows, float(candidate["x_old"]), dtype="float64"),
            "anchor_new_index": np.full(segment_rows, float(candidate["x_new"]), dtype="float64"),
            # The configured projection cap is knowable when the line becomes
            # live.  The actual break row is intentionally not backfilled into
            # earlier output because that would expose future candles.
            "projection_end_index": np.full(
                segment_rows,
                float(candidate.get("projection_cap", candidate["projection_end"])),
                dtype="float64",
            ),
            "line_id": np.full(segment_rows, line_id, dtype="float64"),
            "pivot_count": np.full(
                segment_rows,
                float(candidate.get("pivot_count", 2.0)),
                dtype="float64",
            ),
            "absorbed_pivot_count": np.full(
                segment_rows,
                float(candidate.get("absorbed_pivot_count", candidate.get("pivot_count", 2.0))),
                dtype="float64",
            ),
            "line_width": np.full(
                segment_rows,
                float(candidate.get("half_width", 0.0)),
                dtype="float64",
            ),
            "last_confirm_index": np.full(
                segment_rows,
                float(candidate.get("live_start", candidate.get("x_end", candidate["x_new"]))),
                dtype="float64",
            ),
        }
        _insert_top_line_candidate_slice(top, start, rank_score, metrics)

    _gap_top_line_identity_switches(top)
    return _top_line_arrays_to_columns(top, index, prefix, output_side)


def _stable_line_id(candidate: Series) -> float:
    """Return an identity that cannot be renumbered by later candidates."""

    identity = candidate.get("family_id", candidate["pivot_path"])
    key = f"{candidate['side']}|{identity}".encode("ascii")
    return float(int.from_bytes(blake2b(key, digest_size=6).digest(), "big"))


def _gap_top_line_identity_switches(top: dict[str, np.ndarray]) -> None:
    """Leave one causal gap when a ranked slot changes to another family."""

    line_ids = top["line_id"]
    if line_ids.shape[0] < 2:
        return
    for rank in range(line_ids.shape[1]):
        previous = line_ids[:-1, rank]
        current = line_ids[1:, rank]
        switches = (
            np.flatnonzero(np.isfinite(previous) & np.isfinite(current) & (previous != current)) + 1
        )
        if len(switches) == 0:
            continue
        for key, values in top.items():
            values[switches, rank] = -np.inf if key in {"score", "rank_score"} else np.nan


def _empty_top_line_arrays(rows: int, slots: int) -> dict[str, np.ndarray]:
    top = {
        "score": np.full((rows, slots), -np.inf, dtype="float64"),
        "rank_score": np.full((rows, slots), -np.inf, dtype="float64"),
        "line": np.full((rows, slots), np.nan, dtype="float64"),
        "distance_atr": np.full((rows, slots), np.nan, dtype="float64"),
        "slope": np.full((rows, slots), np.nan, dtype="float64"),
        "slope_atr_per_bar": np.full((rows, slots), np.nan, dtype="float64"),
        "anchor_old_index": np.full((rows, slots), np.nan, dtype="float64"),
        "anchor_new_index": np.full((rows, slots), np.nan, dtype="float64"),
        "projection_end_index": np.full((rows, slots), np.nan, dtype="float64"),
        "line_id": np.full((rows, slots), np.nan, dtype="float64"),
        "pivot_count": np.full((rows, slots), np.nan, dtype="float64"),
        "absorbed_pivot_count": np.full((rows, slots), np.nan, dtype="float64"),
        "line_width": np.full((rows, slots), np.nan, dtype="float64"),
        "last_confirm_index": np.full((rows, slots), np.nan, dtype="float64"),
    }
    return top


def _insert_top_line_candidate_slice(
    top: dict[str, np.ndarray],
    start: int,
    score: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> None:
    inserted = np.zeros(score.shape[0], dtype=bool)
    slots = top["rank_score"].shape[1]
    for rank in range(slots):
        segment = top["rank_score"][start : start + len(score), rank]
        mask = (~inserted) & (score > segment)
        if not np.any(mask):
            continue
        rows = start + np.flatnonzero(mask)
        if rank < slots - 1:
            for values in top.values():
                values[rows, rank + 1 :] = values[rows, rank:-1].copy()
        top["rank_score"][rows, rank] = score[mask]
        for key, candidate_values in metrics.items():
            top[key][rows, rank] = candidate_values[mask]
        inserted[mask] = True
        if np.all(inserted):
            break


def _top_line_arrays_to_columns(
    top: dict[str, np.ndarray],
    index: pd.Index,
    prefix: str,
    side: str,
) -> dict[str, Series]:
    slots = top["rank_score"].shape[1]
    out: dict[str, Series] = {}
    for rank in range(slots):
        valid = np.isfinite(top["rank_score"][:, rank]) & (top["rank_score"][:, rank] > -np.inf)
        line = np.where(valid, top["line"][:, rank], np.nan)
        line_series = pd.Series(line, index=index, dtype="float64")
        out[f"{prefix}_{side}_line_rank{rank}"] = line_series
        out[f"{prefix}_{side}_score_rank{rank}"] = pd.Series(
            np.where(valid, top["score"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        for metric in (
            "distance_atr",
            "slope",
            "slope_atr_per_bar",
            "anchor_old_index",
            "anchor_new_index",
            "projection_end_index",
            "line_id",
            "pivot_count",
            "absorbed_pivot_count",
            "line_width",
            "last_confirm_index",
        ):
            out[f"{prefix}_{side}_{metric}_rank{rank}"] = pd.Series(
                np.where(valid, top[metric][:, rank], np.nan),
                index=index,
                dtype="float64",
            )
    return out


def _empty_ranked_line_columns(
    index: pd.Index, prefix: str, side: str, slots: int
) -> dict[str, Series]:
    out: dict[str, Series] = {}
    for rank in range(slots):
        out.update(_empty_line_columns(index, prefix, side, rank))
    return out


def _empty_line_columns(index: pd.Index, prefix: str, side: str, rank: int) -> dict[str, Series]:
    nan = pd.Series(np.nan, index=index, dtype="float64")
    return {
        f"{prefix}_{side}_line_rank{rank}": nan,
        f"{prefix}_{side}_score_rank{rank}": nan,
        f"{prefix}_{side}_distance_atr_rank{rank}": nan,
        f"{prefix}_{side}_slope_rank{rank}": nan,
        f"{prefix}_{side}_slope_atr_per_bar_rank{rank}": nan,
        f"{prefix}_{side}_anchor_old_index_rank{rank}": nan,
        f"{prefix}_{side}_anchor_new_index_rank{rank}": nan,
        f"{prefix}_{side}_projection_end_index_rank{rank}": nan,
        f"{prefix}_{side}_line_id_rank{rank}": nan,
        f"{prefix}_{side}_pivot_count_rank{rank}": nan,
        f"{prefix}_{side}_absorbed_pivot_count_rank{rank}": nan,
        f"{prefix}_{side}_line_width_rank{rank}": nan,
        f"{prefix}_{side}_last_confirm_index_rank{rank}": nan,
    }


def _span_market_scale(base: dict[str, Series], x_old: float, x_new: float) -> float:
    start = max(0, round(min(float(x_old), float(x_new))))
    end = min(len(base["close"]) - 1, round(max(float(x_old), float(x_new))))
    scale_series = base.get("line_scale", base["atr"])
    values = scale_series.iloc[start : end + 1].to_numpy(dtype="float64")
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) > 0:
        return float(max(np.nanmedian(values), 1e-9))
    fallback = base["atr"].dropna()
    if not fallback.empty:
        return float(max(float(fallback.median()), 1e-9))
    return 1e-9


def _anchor_span_is_clear(
    base: dict[str, Series],
    side: LineSide,
    x_old: int,
    x_new: int,
    slope: float,
    intercept: float,
    cfg: TrendlineProjectionV2Config,
) -> bool:
    """Reject only meaningful body breaks between p1 and p2.

    Pivots are discrete approximations. A candle very close to either anchor,
    or a short/shallow interior breach, can be the same human-visible turn that
    the pivot picker represented with a neighbouring candle. Those exceptions
    prevent valid seed lines from being deleted before confirmation logic can
    assess them.
    """

    start = max(x_old + 1, 0)
    end = min(x_new, len(base["close"]))
    if end <= start:
        return True

    xs = np.arange(start, end, dtype="float64")
    line = intercept + slope * xs
    if side == "resistance":
        body = base["body_high"].iloc[start:end].to_numpy(dtype="float64")
        breach = body > line
        beyond = body - line
    else:
        body = base["body_low"].iloc[start:end].to_numpy(dtype="float64")
        breach = body < line
        beyond = line - body

    if not bool(np.any(breach)):
        return True

    return _anchor_breaches_are_allowed(
        breach=breach,
        beyond=beyond,
        atr=base["atr"].iloc[start:end].to_numpy(dtype="float64"),
        x_values=xs,
        x_old=x_old,
        x_new=x_new,
        edge_bars=int(cfg.anchor_break_edge_bars),
        max_run=int(cfg.anchor_break_max_run),
        max_atr_mult=float(cfg.anchor_break_max_atr_mult),
    )


def _anchor_breaches_are_allowed(
    *,
    breach: np.ndarray,
    beyond: np.ndarray,
    atr: np.ndarray,
    x_values: np.ndarray,
    x_old: int,
    x_new: int,
    edge_bars: int,
    max_run: int,
    max_atr_mult: float,
) -> bool:
    breach_indexes = np.flatnonzero(breach)
    if len(breach_indexes) == 0:
        return True
    breached_x = x_values[breach_indexes]
    near_anchor = (breached_x <= float(x_old + edge_bars)) | (
        breached_x >= float(x_new - edge_bars)
    )
    interior = breach_indexes[~near_anchor]
    if len(interior) == 0:
        return True

    interior_breach = np.zeros_like(breach, dtype=bool)
    interior_breach[interior] = True
    max_interior_run = _max_true_run(interior_breach)
    atr_beyond = beyond[interior] / np.maximum(atr[interior], 1e-9)
    shallow_enough = bool(np.nanmax(atr_beyond) <= float(max_atr_mult))
    return max_interior_run <= int(max_run) and shallow_enough


def _max_true_run(values: np.ndarray) -> int:
    truth = np.asarray(values, dtype=bool)
    if len(truth) == 0 or not np.any(truth):
        return 0
    positions = np.arange(len(truth), dtype="int64")
    last_false = np.maximum.accumulate(np.where(truth, -1, positions))
    run_lengths = np.where(truth, positions - last_false, 0)
    return int(run_lengths.max())


def _keep_long_true_runs(values: np.ndarray, min_run: int) -> np.ndarray:
    """Causally accept a run from its confirmation row onward.

    The first ``min_run - 1`` rows remain false because their eventual run
    length is not known yet.  Appending candles therefore cannot rewrite a
    historical output from false to true.
    """

    threshold = max(int(min_run), 1)
    truth = np.asarray(values, dtype=bool)
    if threshold <= 1 or len(truth) == 0:
        return truth
    positions = np.arange(len(truth), dtype="int64")
    last_false = np.maximum.accumulate(np.where(truth, -1, positions))
    run_lengths = positions - last_false
    return truth & (run_lengths >= threshold)


def _atr(frame: DataFrame, period: int) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(int(period), min_periods=max(2, int(period) // 2)).mean()


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _resolve_config(
    config: TrendlineProjectionV2Config | None, overrides: dict[str, object]
) -> TrendlineProjectionV2Config:
    base = config or TrendlineProjectionV2Config()
    valid = {field.name for field in fields(TrendlineProjectionV2Config)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown trendline v2 config override(s): {', '.join(unknown)}")

    requested = replace(base, **{name: clean[name] for name in clean if name in valid})
    timeframe = _normalize_timeframe(str(requested.timeframe))
    requested = replace(requested, timeframe=timeframe)
    profile = _TIMEFRAME_PROFILES[timeframe]
    if config is None:
        profiled = replace(requested, **profile)
    else:
        default = TrendlineProjectionV2Config()
        profile_updates = {
            name: profile[name]
            for name in _PROFILE_FIELDS
            if name not in clean and getattr(config, name) == getattr(default, name)
        }
        profiled = replace(requested, **profile_updates)
    if clean:
        profiled = replace(profiled, **{name: clean[name] for name in clean if name in valid})
        profiled = replace(profiled, timeframe=_normalize_timeframe(str(profiled.timeframe)))
    return profiled


def _normalize_timeframe(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _TIMEFRAME_ALIASES:
        allowed = ", ".join(sorted(_TIMEFRAME_PROFILES))
        raise ValueError(f"timeframe must be one of: {allowed}")
    return _TIMEFRAME_ALIASES[key]


def _validate_dataframe(frame: DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataframe missing required columns: {missing}")


def _validate_config(cfg: TrendlineProjectionV2Config) -> None:  # noqa: C901
    _normalize_timeframe(str(cfg.timeframe))
    if cfg.pivot_strength < 1:
        raise ValueError("pivot_strength must be positive")
    if cfg.raw_line_output_count < 1:
        raise ValueError("raw_line_output_count must be positive")
    if int(cfg.confirmed_min_pivots) < 3:
        raise ValueError("confirmed_min_pivots must be at least 3")
    if int(cfg.provisional_line_output_count) < 1:
        raise ValueError("provisional_line_output_count must be positive")
    if cfg.min_anchor_bars < 1:
        raise ValueError("min_anchor_bars must be positive")
    if cfg.max_anchor_bars < cfg.min_anchor_bars:
        raise ValueError("max_anchor_bars must be greater than or equal to min_anchor_bars")
    if cfg.min_pivot_prominence_atr < 0.0:
        raise ValueError("min_pivot_prominence_atr must be non-negative")
    if cfg.max_slope_atr_per_bar <= 0.0:
        raise ValueError("max_slope_atr_per_bar must be positive")
    if cfg.max_projection_bars < 1:
        raise ValueError("max_projection_bars must be positive")
    if cfg.max_active_line_distance_atr_mult <= 0.0:
        raise ValueError("max_active_line_distance_atr_mult must be positive")
    if cfg.absorb_width_scale < 0.0:
        raise ValueError("absorb_width_scale must be non-negative")
    if cfg.duplicate_line_proximity_atr_mult <= 0.0:
        raise ValueError("duplicate_line_proximity_atr_mult must be positive")
    if cfg.anchor_break_edge_bars < 0 or cfg.anchor_break_max_run < 0:
        raise ValueError("anchor break bars/run settings must be non-negative")
    if cfg.anchor_break_max_atr_mult < 0.0:
        raise ValueError("anchor_break_max_atr_mult must be non-negative")
    if cfg.projection_break_candles < 1:
        raise ValueError("projection_break_candles must be positive")
    if cfg.projection_break_atr_mult < 0.0:
        raise ValueError("projection_break_atr_mult must be non-negative")
    if cfg.join_angle_tolerance <= 0.0:
        raise ValueError("join_angle_tolerance must be positive")
    if cfg.max_joined_pivots < 2:
        raise ValueError("max_joined_pivots must be at least 2")
    if int(cfg.confirmed_min_pivots) > int(cfg.max_joined_pivots):
        raise ValueError("confirmed_min_pivots cannot exceed max_joined_pivots")
    if cfg.family_touch_tolerance_atr_mult <= 0.0:
        raise ValueError("family_touch_tolerance_atr_mult must be positive")
    if cfg.family_fit_tolerance_atr_mult <= 0.0:
        raise ValueError("family_fit_tolerance_atr_mult must be positive")
    if cfg.family_seed_pivot_lookback < 2:
        raise ValueError("family_seed_pivot_lookback must be at least 2")
    if cfg.max_active_seeds_per_side < 1:
        raise ValueError("max_active_seeds_per_side must be positive")
    if cfg.max_active_families_per_side < 1:
        raise ValueError("max_active_families_per_side must be positive")
    if not 0.0 <= float(cfg.min_candidate_line_score) <= 1.0:
        raise ValueError("min_candidate_line_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_output_line_score) <= 1.0:
        raise ValueError("min_output_line_score must be between 0 and 1")
    if cfg.min_output_active_bars < 1:
        raise ValueError("min_output_active_bars must be positive")
    if float(cfg.proximity_rank_weight) < 0.0:
        raise ValueError("proximity_rank_weight must be non-negative")
    if float(cfg.proximity_rank_distance_cap_atr) <= 0.0:
        raise ValueError("proximity_rank_distance_cap_atr must be positive")


__all__ = [
    "TrendlineProjectionV2Config",
    "TrendlineProjectionV2State",
    "add_trendline_projection_v2",
    "build_trendline_projection_v2_state",
    "resolve_trendline_projection_v2_state",
]
