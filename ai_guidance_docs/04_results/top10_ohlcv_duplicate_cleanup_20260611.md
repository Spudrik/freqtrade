---
doc_status: active
default_read: routed
owner: agent
purpose: Record of top-10 OHLCV duplicate cleanup and canonical futures data location.
do_not_use_for: Strategy objective definition.
last_rebuilt: 2026-06-11
---

# Top-10 OHLCV Duplicate Cleanup - 2026-06-11

## 1. Canonical data location

Use this location for top-10 futures OHLCV:

`C:\FreqTradeStuff\user_data\data\binance\futures`

Expected futures OHLCV set:

- Pairs: `BTC`, `ETH`, `BNB`, `SOL`, `XRP`, `ADA`, `DOGE`, `TRX`, `AVAX`, `LINK`
- Timeframes: `1h`, `2h`, `4h`, `8h`, `12h`, `1d`, `3d`, `1w`
- Expected files: `80`
- Present files after cleanup: `80`
- Missing files after cleanup: `0`

## 2. Deleted duplicate/non-standard files

Deleted `35` direct `.feather` files from:

1. `C:\FreqTradeStuff\user_data\data`
2. `C:\FreqTradeStuff\user_data\data\binance`

Total deleted size: `8,858,998` bytes.

No files were deleted from:

`C:\FreqTradeStuff\user_data\data\binance\futures`

## 3. Why deleted

The deleted files were non-standard root/direct-binance OHLCV copies that conflicted with the launcher/freqtrade futures cache layout. They caused earlier audit confusion where only BTC/ETH/SOL appeared to exist because the check looked in the direct binance folder instead of the futures subfolder.

## 4. Current rule

For futures backtesting/research, use the canonical futures files only:

`C:\FreqTradeStuff\user_data\data\binance\futures\*_USDT_USDT-*-futures.feather`
