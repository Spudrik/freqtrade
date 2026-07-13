"""
Build market-basket feature snapshots from local Freqtrade OHLCV data.

The output has two parquet files per timeframe:
- long: one row per date/basket with plain feature names.
- wide: one row per date with FreqAI-style feature/label prefixes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_data
from freqtrade.enums import CandleType


DEFAULT_DATADIR = Path("user_data/data/binance")
DEFAULT_OUTPUT_DIR = Path("user_data/market_context_data/market_basket_features")
DEFAULT_TIMEFRAMES = ["1h"]
RETURN_WINDOWS = [1, 3, 6, 12, 24]
TARGET_HORIZONS = [3, 6, 12, 24]
ROLLING_SHORT = 48
ROLLING_LONG = 168


PAIR_FILE_RE = re.compile(r"^(.+)_USDT_USDT-(?P<timeframe>[^-]+)-futures\.feather$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BTC/top-N market basket feature parquet snapshots."
    )
    parser.add_argument("--datadir", type=Path, default=DEFAULT_DATADIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional explicit Freqtrade pairs.")
    parser.add_argument("--min-members", type=int, default=1)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.8)
    parser.add_argument("--data-format", default="feather")
    return parser.parse_args()


def data_root(datadir: Path) -> Path:
    return datadir.parent if datadir.name.lower() == "futures" else datadir


def futures_dir(datadir: Path) -> Path:
    root = data_root(datadir)
    candidate = root / "futures"
    return candidate if candidate.exists() else root


def discover_pairs(datadir: Path, timeframe: str) -> list[str]:
    pairs: list[str] = []
    for path in sorted(futures_dir(datadir).glob(f"*-{timeframe}-futures.feather")):
        match = PAIR_FILE_RE.match(path.name)
        if not match:
            continue
        base = match.group(1)
        pairs.append(f"{base}/USDT:USDT")
    return pairs


def zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=max(3, window // 4))
    mean = rolling.mean()
    std = rolling.std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.set_index("date")

    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    body_pressure = ((out["close"] - out["open"]) / candle_range).clip(-1, 1).fillna(0.0)

    out["quote_volume"] = out["close"] * out["volume"]
    out["signed_quote_volume"] = out["quote_volume"] * body_pressure
    out["pressure_score"] = out["signed_quote_volume"] / out["quote_volume"].replace(0, np.nan)
    out["true_range_pct"] = true_range / out["close"].replace(0, np.nan)
    out["candle_body_pct"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    for window in RETURN_WINDOWS:
        out[f"ret_{window}"] = out["close"].pct_change(window)
        out[f"direction_{window}"] = np.sign(out[f"ret_{window}"])

    out["volume_z_48"] = zscore(out["quote_volume"], ROLLING_SHORT)
    out["realized_vol_24"] = out["ret_1"].rolling(24, min_periods=8).std(ddof=0)

    tr_median = out["true_range_pct"].rolling(ROLLING_SHORT, min_periods=12).median()
    tr_mean = out["true_range_pct"].rolling(ROLLING_SHORT, min_periods=12).mean()
    vol_median = out["realized_vol_24"].rolling(ROLLING_LONG, min_periods=24).median()
    qv_median = out["quote_volume"].rolling(ROLLING_SHORT, min_periods=12).median()

    out["compression_ratio"] = out["true_range_pct"] / tr_median.replace(0, np.nan)
    out["vol_compression_ratio"] = out["realized_vol_24"] / vol_median.replace(0, np.nan)
    out["range_expansion_ratio"] = out["true_range_pct"] / tr_mean.replace(0, np.nan)
    out["volume_expansion_ratio"] = out["quote_volume"] / qv_median.replace(0, np.nan)
    out["expansion_score"] = (
        out["range_expansion_ratio"].clip(lower=0)
        * np.log1p(out["volume_expansion_ratio"].clip(lower=0))
    )
    return out


def rank_pairs_by_quote_volume(pair_frames: dict[str, pd.DataFrame]) -> list[str]:
    ranked = []
    for pair, frame in pair_frames.items():
        ranked.append((pair, frame["quote_volume"].median(skipna=True)))
    ranked.sort(key=lambda item: (-np.nan_to_num(item[1], nan=-1.0), item[0]))
    return [pair for pair, _ in ranked]


def stack_feature(pair_frames: dict[str, pd.DataFrame], pairs: list[str], feature: str) -> pd.DataFrame:
    return pd.concat([pair_frames[pair][feature].rename(pair) for pair in pairs], axis=1)


def future_max(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]


def build_basket(
    basket: str,
    pairs: list[str],
    timeframe: str,
    pair_frames: dict[str, pd.DataFrame],
    btc_frame: pd.DataFrame | None,
    min_members: int,
    min_coverage_ratio: float,
) -> pd.DataFrame:
    close = stack_feature(pair_frames, pairs, "close")
    ret_1 = stack_feature(pair_frames, pairs, "ret_1")
    quote_volume = stack_feature(pair_frames, pairs, "quote_volume")
    signed_quote_volume = stack_feature(pair_frames, pairs, "signed_quote_volume")

    active_members = close.notna().sum(axis=1)
    required_members = max(min_members, int(np.ceil(len(pairs) * min_coverage_ratio)))
    valid = active_members >= required_members
    result = pd.DataFrame(index=close.index)
    result["timeframe"] = timeframe
    result["basket"] = basket
    result["rank_method"] = "median_quote_volume_available_pairs"
    result["configured_member_count"] = len(pairs)
    result["required_member_count"] = required_members
    result["active_member_count"] = active_members
    result["coverage_ratio"] = active_members / max(len(pairs), 1)

    for window in RETURN_WINDOWS:
        returns = stack_feature(pair_frames, pairs, f"ret_{window}")
        result[f"ret_{window}_mean"] = returns.mean(axis=1, skipna=True)
        result[f"ret_{window}_median"] = returns.median(axis=1, skipna=True)
        result[f"breadth_up_{window}"] = (returns > 0).sum(axis=1) / active_members.replace(0, np.nan)
        result[f"breadth_down_{window}"] = (returns < 0).sum(axis=1) / active_members.replace(0, np.nan)

    result["quote_volume_sum"] = quote_volume.sum(axis=1, min_count=1)
    result["quote_volume_z_48"] = zscore(result["quote_volume_sum"], ROLLING_SHORT)
    result["pressure_score"] = (
        signed_quote_volume.sum(axis=1, min_count=1) / result["quote_volume_sum"].replace(0, np.nan)
    )

    for feature in [
        "true_range_pct",
        "realized_vol_24",
        "compression_ratio",
        "vol_compression_ratio",
        "range_expansion_ratio",
        "volume_expansion_ratio",
        "expansion_score",
    ]:
        values = stack_feature(pair_frames, pairs, feature)
        result[f"{feature}_mean"] = values.mean(axis=1, skipna=True)
        result[f"{feature}_median"] = values.median(axis=1, skipna=True)

    result["market_index"] = (1.0 + result["ret_1_mean"].fillna(0.0)).cumprod() * 100.0

    if btc_frame is not None:
        btc = btc_frame.reindex(result.index)
        btc_quote_volume = btc["quote_volume"]
        for window in RETURN_WINDOWS:
            result[f"btc_ret_{window}"] = btc[f"ret_{window}"]
            result[f"basket_minus_btc_ret_{window}"] = (
                result[f"ret_{window}_mean"] - btc[f"ret_{window}"]
            )
        result["btc_pressure_score"] = btc["pressure_score"]
        result["pressure_minus_btc"] = result["pressure_score"] - btc["pressure_score"]
        result["btc_quote_volume_share"] = (
            btc_quote_volume / result["quote_volume_sum"].replace(0, np.nan)
        )
    else:
        for window in RETURN_WINDOWS:
            result[f"btc_ret_{window}"] = np.nan
            result[f"basket_minus_btc_ret_{window}"] = np.nan
        result["btc_pressure_score"] = np.nan
        result["pressure_minus_btc"] = np.nan
        result["btc_quote_volume_share"] = np.nan

    for horizon in TARGET_HORIZONS:
        result[f"fwd_ret_{horizon}"] = (
            result["market_index"].shift(-horizon) / result["market_index"] - 1.0
        )
        if btc_frame is not None:
            btc_close = btc_frame["close"].reindex(result.index)
            result[f"fwd_basket_minus_btc_ret_{horizon}"] = (
                result[f"fwd_ret_{horizon}"] - (btc_close.shift(-horizon) / btc_close - 1.0)
            )
        else:
            result[f"fwd_basket_minus_btc_ret_{horizon}"] = np.nan
        result[f"fwd_expansion_max_{horizon}"] = future_max(
            result["expansion_score_mean"], horizon
        )

    result.loc[~valid, result.columns.difference(["timeframe", "basket", "rank_method"])] = np.nan
    return result.reset_index(names="date")


def basket_members(ranked_pairs: list[str]) -> dict[str, list[str]]:
    alt_pairs = [pair for pair in ranked_pairs if not pair.startswith("BTC/")]
    btc_pair = [pair for pair in ranked_pairs if pair.startswith("BTC/")]
    return {
        "btc": btc_pair[:1],
        "top3": ranked_pairs[:3],
        "top10": ranked_pairs[:10],
        "top10_alt": alt_pairs[:10],
        "top100": ranked_pairs[:100],
        "top100_alt": alt_pairs[:100],
    }


def build_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = {
        "configured_member_count",
        "required_member_count",
        "active_member_count",
        "coverage_ratio",
    }
    id_cols = {"date", "timeframe", "basket", "rank_method"}
    value_cols = [col for col in long_df.columns if col not in id_cols]
    pieces = []
    for basket, group in long_df.groupby("basket", sort=True):
        frame = group.set_index("date")[value_cols].sort_index()
        renamed = {}
        for col in frame.columns:
            if col in meta_cols:
                renamed[col] = f"mkt_{basket}_{col}"
            elif col.startswith("fwd_"):
                renamed[col] = f"&-mkt_{basket}_{col}"
            else:
                renamed[col] = f"%-mkt_{basket}_{col}"
        pieces.append(frame.rename(columns=renamed))
    return pd.concat(pieces, axis=1).reset_index()


def build_timeframe(args: argparse.Namespace, timeframe: str) -> tuple[Path, Path, pd.DataFrame]:
    pairs = args.pairs or discover_pairs(args.datadir, timeframe)
    if not pairs:
        raise SystemExit(f"No futures OHLCV files found for timeframe {timeframe} in {args.datadir}")

    raw = load_data(
        datadir=data_root(args.datadir),
        timeframe=timeframe,
        pairs=pairs,
        data_format=args.data_format,
        candle_type=CandleType.FUTURES,
        fill_up_missing=True,
    )
    pair_frames = {pair: add_pair_features(frame) for pair, frame in raw.items() if not frame.empty}
    if not pair_frames:
        raise SystemExit(f"No usable OHLCV data loaded for timeframe {timeframe}")

    ranked = rank_pairs_by_quote_volume(pair_frames)
    btc_frame = pair_frames.get("BTC/USDT:USDT")
    baskets = {
        name: members
        for name, members in basket_members(ranked).items()
        if members and len(members) >= args.min_members
    }

    long_df = pd.concat(
        [
            build_basket(
                name,
                members,
                timeframe,
                pair_frames,
                btc_frame,
                min_members=args.min_members,
                min_coverage_ratio=args.min_coverage_ratio,
            )
            for name, members in baskets.items()
        ],
        ignore_index=True,
    ).sort_values(["date", "basket"])
    wide_df = build_wide(long_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / f"market_basket_features_{timeframe}.parquet"
    wide_path = args.output_dir / f"market_basket_features_wide_freqai_{timeframe}.parquet"
    long_df.to_parquet(long_path, index=False)
    wide_df.to_parquet(wide_path, index=False)
    return long_path, wide_path, long_df


def main() -> None:
    args = parse_args()
    for timeframe in args.timeframes:
        long_path, wide_path, long_df = build_timeframe(args, timeframe)
        baskets = ", ".join(sorted(long_df["basket"].unique()))
        start = long_df["date"].min()
        end = long_df["date"].max()
        print(f"{timeframe}: wrote {long_path}")
        print(f"{timeframe}: wrote {wide_path}")
        print(f"{timeframe}: baskets={baskets}; rows={len(long_df)}; range={start} -> {end}")


if __name__ == "__main__":
    main()
