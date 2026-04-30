# Free Market Data Inventory

## Implemented in Order Book Lab

- Binance spot order book partial-depth streams: public WebSocket, no key required.
- Binance USD-M futures order book partial-depth streams: public WebSocket, no key required.
- Bybit spot order book streams: public V5 WebSocket, no key required.
- Bybit linear futures order book streams: public V5 WebSocket, no key required.
- Binance USD-M context: funding, open interest, global long/short account ratio, and taker long/short volume from public REST endpoints.
- Bybit linear context: funding, open interest, account ratio, and recent-trade taker volume approximation from public REST endpoints. The shared `5m` context period is translated to Bybit's `5min` parameter format at request time.

## Implemented in Global Context

- Alternative.me Fear & Greed index: stored as a 0-100 sentiment score with risk-on/risk-off labeling.
- CoinGecko global market endpoint: total crypto market cap, volume, BTC dominance, and 24h market-cap change.
- CoinGecko BTC/ETH markets endpoint: BTC/ETH price and short-term price-change context.
- DeFiLlama stablecoins endpoint: stablecoin supply trend as a crypto-liquidity proxy.
- DeFiLlama chains endpoint: aggregate DeFi TVL and weighted chain TVL change.

## Good Next Candidates

- OKX public order book, funding, open interest, long/short ratio, and taker flow. This is the most natural next exchange because the public market-data API is broad and derivatives context is strong.
- Coinbase public spot order book and trades. Useful as a US spot venue reference, but it does not provide perp context.
- Kraken public spot order book and trades. Useful as a second non-Binance spot venue reference.
- Deribit public BTC/ETH options and futures data. Useful for volatility, funding, and options positioning context rather than broad altcoin coverage.
- Exchange announcement RSS/API sources for listings, delistings, launches, maintenance, and leverage/margin changes.
- Official macro RSS feeds: Federal Reserve, Treasury, BLS, BEA, ECB, BOE, and economic calendar sources with permissive terms.

## Rules For Adding Sources

- Prefer official public API, WebSocket, or RSS endpoints before HTML scraping.
- Store every source with `market_key`, `canonical_pair`, `symbol`, and timestamp fields so it can join to order book bars later.
- Keep this layer data-only. Strategy use belongs in a separate research card after the data has enough history to validate.

## Official References Checked

- Binance USD-M Futures market data REST docs: [funding rate](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History), [open interest](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest), [long/short ratio](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Long-Short-Ratio), and [taker buy/sell volume](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume).
- Bybit V5 market data docs: [funding history](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate), [open interest](https://bybit-exchange.github.io/docs/v5/market/open-interest), [long/short ratio](https://bybit-exchange.github.io/docs/v5/market/long-short-ratio), and [recent public trades](https://bybit-exchange.github.io/docs/v5/market/recent-trade).
