"""Read-only, pre-cost signal review for the live PaperLeaderImpulse rule.

This is not a trade backtest: it ignores stops, targets, fills, fees, funding,
slippage, position limits, and the Luna/level stake modifiers. It reuses the
strategy's entry method and checks whether enough non-overlapping opportunities
exist and whether their later *direction* is better than same-side local breaks
made while Bitcoin also moves that way, but without a large/busy impulse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import talib.abstract as ta

from user_data.strategies.paper_leader_impulse import PaperLeaderImpulse


PAIRS = ("ETH", "SOL", "BNB", "DOGE", "1000PEPE")
PERIODS = (
    ("2024-04_to_2025-03", "2024-04-01", "2025-04-01"),
    ("2025-04_to_2026-03", "2025-04-01", "2026-04-01"),
    ("2026-04_onward", "2026-04-01", "2027-01-01"),
)
HOUR = pd.Timedelta(hours=1)


def _load(data_dir: Path, symbol: str) -> pd.DataFrame:
    path = data_dir / f"{symbol}_USDT_USDT-1h-futures.feather"
    frame = pd.read_feather(path).sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"Duplicate or unordered candles: {path}")
    return frame


def _separated(indices: pd.Index, dates: pd.Series, hours: int = 48) -> pd.Index:
    selected: list[int] = []
    last: pd.Timestamp | None = None
    for index in indices:
        at = dates.iloc[index]
        if last is None or at - last >= pd.Timedelta(hours=hours):
            selected.append(int(index))
            last = at
    return pd.Index(selected, dtype="int64")


def _outcome(frame: pd.DataFrame, indices: pd.Index, sides: pd.Series, hours: int) -> pd.Series:
    entry = frame["open"].shift(-1).iloc[indices].to_numpy()
    exit_price = frame["close"].shift(-hours).iloc[indices].to_numpy()
    return pd.Series(sides.iloc[indices].to_numpy() * (exit_price / entry - 1.0) * 100.0)


def _summarize(frame: pd.DataFrame, mask: pd.Series, sides: pd.Series) -> dict[str, float | int]:
    raw = int(mask.sum())
    episodes = _separated(frame.index[mask], frame["date"])
    result: dict[str, float | int] = {"raw": raw, "episodes": len(episodes)}
    for horizon in (6, 24, 48):
        valid = episodes[frame["date"].shift(-horizon).iloc[episodes].to_numpy()
                         - frame["date"].iloc[episodes].to_numpy()
                         == pd.Timedelta(hours=horizon)]
        returns = _outcome(frame, valid, sides, horizon)
        result[f"n{horizon}"] = len(returns)
        result[f"win{horizon}"] = round(float((returns > 0).mean() * 100), 1) if len(returns) else float("nan")
        result[f"mean{horizon}"] = round(float(returns.mean()), 3) if len(returns) else float("nan")
        if horizon == 48:
            valid_sides = sides.iloc[valid].to_numpy()
            for name, sign in (("long", 1), ("short", -1)):
                directional = returns.iloc[valid_sides == sign]
                result[f"{name}_n48"] = len(directional)
                result[f"{name}_win48"] = (
                    round(float((directional > 0).mean() * 100), 1)
                    if len(directional) else float("nan")
                )
                result[f"{name}_mean48"] = (
                    round(float(directional.mean()), 3) if len(directional) else float("nan")
                )
    return result


def review(data_dir: Path) -> pd.DataFrame:
    btc = _load(data_dir, "BTC")
    btc["btc_change"] = btc["close"].pct_change()
    btc["btc_prior_volume"] = btc["volume"].shift(1).rolling(20).median()
    btc["btc_contiguous"] = btc["date"].diff().eq(HOUR).rolling(20, min_periods=20).sum().eq(20)
    leader = btc[["date", "btc_change", "volume", "btc_prior_volume", "btc_contiguous"]].rename(
        columns={"volume": "btc_volume"}
    )
    rows: list[dict] = []
    for symbol in PAIRS:
        frame = _load(data_dir, symbol).merge(leader, on="date", how="left", validate="one_to_one")
        frame["paper_atr"] = ta.ATR(frame, timeperiod=14)
        frame["local_prior_6h_high"] = frame["high"].shift(1).rolling(6).max()
        frame["local_prior_6h_low"] = frame["low"].shift(1).rolling(6).min()
        frame["local_prior_volume"] = frame["volume"].shift(1).rolling(20).median()
        frame["coin_contiguous"] = frame["date"].diff().eq(HOUR).rolling(20, min_periods=20).sum().eq(20)
        # Reuse the live strategy's exact entry method; indicators above mirror its inputs.
        frame = PaperLeaderImpulse.populate_entry_trend(None, frame, {"pair": f"{symbol}/USDT:USDT"})
        valid = frame["coin_contiguous"] & frame["btc_contiguous"].fillna(False)
        valid &= frame["paper_atr"].notna() & frame["local_prior_volume"].gt(0)
        valid &= frame["btc_prior_volume"].gt(0)
        volume = frame["volume"].ge(frame["local_prior_volume"])
        up_break = frame["close"].gt(frame["local_prior_6h_high"])
        down_break = frame["close"].lt(frame["local_prior_6h_low"])
        btc_up = frame["btc_change"].ge(0.006) & frame["btc_volume"].ge(1.25 * frame["btc_prior_volume"])
        btc_down = frame["btc_change"].le(-0.006) & frame["btc_volume"].ge(1.25 * frame["btc_prior_volume"])
        signal_long = frame["enter_long"].fillna(0).eq(1)
        signal_short = frame["enter_short"].fillna(0).eq(1)
        if not signal_long.eq(btc_up & volume & up_break & frame["paper_atr"].notna()
                              & frame["local_prior_volume"].gt(0)
                              & frame["btc_prior_volume"].gt(0)).all():
            raise AssertionError(f"Signal mismatch for {symbol}; review live rule before using results")
        if not signal_short.eq(btc_down & volume & down_break & frame["paper_atr"].notna()
                               & frame["local_prior_volume"].gt(0)
                               & frame["btc_prior_volume"].gt(0)).all():
            raise AssertionError(f"Signal mismatch for {symbol}; review live rule before using results")
        aligned_small_long = valid & volume & up_break & frame["btc_change"].gt(0) & ~btc_up
        aligned_small_short = valid & volume & down_break & frame["btc_change"].lt(0) & ~btc_down
        sides = pd.Series(0, index=frame.index)
        sides.loc[signal_long | aligned_small_long] = 1
        sides.loc[signal_short | aligned_small_short] = -1
        signal = valid & (signal_long | signal_short)
        aligned_small = aligned_small_long | aligned_small_short
        for label, start, end in PERIODS:
            in_period = frame["date"].ge(pd.Timestamp(start, tz="UTC")) & frame["date"].lt(
                pd.Timestamp(end, tz="UTC")
            )
            for family, mask in (("BTC_plus_local", signal), ("aligned_small_BTC", aligned_small)):
                rows.append({"coin": symbol, "period": label, "family": family,
                             **_summarize(frame, mask & in_period, sides)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(r"D:\FreqTradeStuffLargeData\freqtrade_data\binance\futures"))
    parser.add_argument("--split-side", action="store_true", help="Show 48h outcomes for longs and shorts separately")
    args = parser.parse_args()
    result = review(args.data_dir)
    print("Pre-cost directional check; not a trading-profit estimate. One episode per coin per 48h.")
    if args.split_side:
        columns = ["coin", "period", "family", "long_n48", "long_win48", "long_mean48",
                   "short_n48", "short_win48", "short_mean48"]
        print(result[columns].to_string(index=False))
    else:
        print(result.drop(columns=["long_n48", "long_win48", "long_mean48",
                                   "short_n48", "short_win48", "short_mean48"]).to_string(index=False))


if __name__ == "__main__":
    main()
